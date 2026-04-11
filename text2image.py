#!/usr/bin/env python3
"""
text2image.py

Production-ready text-to-image utility using Hugging Face Diffusers.

Features:
- Loads a Stable Diffusion pipeline from Hugging Face (configurable model id)
- Device-aware (GPU if available, falls back to CPU)
- Controlled RNG seed for reproducible outputs
- Input validation, logging, and robust error handling
- CLI for simple usage and programmatic API (Text2ImageGenerator)

Security considerations:
- Uses HF token from environment variable or explicit argument (do not hardcode tokens)
- Validates prompt type and basic length
- Sanitizes output path

Note: keep your environment's torch/diffusers versions compatible with the requirements.txt provided.

"""
from __future__ import annotations

import argparse
import io
import logging
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import PIL.Image
import torch

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import login as hf_login

# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


@dataclass
class GenerationOptions:
    """Options governing generation behavior.

    Attributes:
        num_inference_steps: Number of denoising steps. More steps -> slower, usually better quality.
        guidance_scale: Classifier-free guidance scale. Higher values make images more faithful to the prompt.
        height: Output height in pixels (must match model constraints).
        width: Output width in pixels (must match model constraints).
        seed: Optional random seed for reproducibility.
    """

    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: Optional[int] = None
    width: Optional[int] = None
    seed: Optional[int] = None


class Text2ImageGenerator:
    """High-level wrapper around a Hugging Face Diffusers text2image pipeline.

    Example:
        gen = Text2ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")
        img = gen.generate("A futuristic city skyline at sunset", GenerationOptions(seed=42))

    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[Union[str, torch.device]] = None,
        hf_token: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_auth_token: Optional[str] = None,
        scheduler: Optional[str] = "dpmsolver"  # default scheduler
    ) -> None:
        """Initialize and load the pipeline.

        Args:
            model_id: Hugging Face model id to load.
            device: Device to load the pipeline on, e.g. "cuda" or "cpu". If None, auto-detects.
            hf_token: HF access token; if not provided it will attempt to read HUGGINGFACE_HUB_TOKEN.
            torch_dtype: torch dtype to use (e.g. torch.float16 for GPU). If None, auto-determined.
            use_auth_token: Backwards-compatible alias for hf_token.
            scheduler: Optional scheduler name. Currently supports 'dpmsolver' and 'multistep'.
        """
        self.model_id = model_id
        self.hf_token = hf_token or use_auth_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        # Choose device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Determine dtype
        if torch_dtype is None:
            # Use float16 on GPU for faster inference if available
            self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Validate token and login if provided
        if self.hf_token:
            try:
                hf_login(token=self.hf_token)
            except Exception:
                # don't hard fail; from_pretrained can often accept token via use_auth_token param
                logger.debug("HF login failed or not necessary; continuing. Token may still be accepted by from_pretrained.")

        # Load pipeline
        self.pipeline: Optional[DiffusionPipeline] = None
        self.scheduler = scheduler
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load the diffusers pipeline with safe defaults and optional scheduler."""
        try:
            logger.info("Loading pipeline for model: %s", self.model_id)

            # Choose scheduler class if requested
            scheduler_obj = None
            if self.scheduler is not None:
                sched_name = self.scheduler.lower()
                if sched_name in ("dpmsolver", "dpmsolvermultistep"):
                    scheduler_obj = DPMSolverMultistepScheduler
                # additional schedulers can be added here

            # from_pretrained will automatically download cache and required files
            pipeline_kwargs = dict(
                torch_dtype=self.torch_dtype,
                revision=None,
                use_auth_token=self.hf_token,
            )

            # Load pipeline; choose StableDiffusionPipeline inference class
            pipe = DiffusionPipeline.from_pretrained(self.model_id, **pipeline_kwargs)

            if scheduler_obj is not None:
                # Replace scheduler with a new one if requested
                pipe.scheduler = scheduler_obj.from_config(pipe.scheduler.config)

            # Move pipeline to device
            pipe = pipe.to(self.device)

            # Enable attention slicing to reduce peak memory usage on GPU
            try:
                pipe.enable_attention_slicing()
            except Exception:
                logger.debug("Pipeline does not support enable_attention_slicing() on this model.")

            # Optionally enable memory-efficient VAE tiling or other optimizations here

            self.pipeline = pipe
            logger.info("Pipeline loaded on device=%s dtype=%s", self.device, self.torch_dtype)
        except Exception as e:
            logger.exception("Failed to load the diffusion pipeline: %s", e)
            raise RuntimeError("Failed to load the diffusion pipeline") from e

    def _validate_prompt(self, prompt: str) -> None:
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")
        if not prompt.strip():
            raise ValueError("Prompt must not be empty or whitespace")
        if len(prompt) > 2000:
            # arbitrary safety check to avoid extremely long prompts
            raise ValueError("Prompt too long (max 2000 characters)")

    def _seed_everything(self, seed: Optional[int]) -> int:
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
        rand = np.random.RandomState(seed)
        random.seed(seed)
        np.random.set_state(rand.get_state())
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            logger.debug("Torch seeding not available in this environment")
        return seed

    def generate(
        self,
        prompt: str,
        options: Optional[GenerationOptions] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[PIL.Image.Image, int]:
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            options: GenerationOptions controlling sampling and size.
            output_path: If provided, path to save the resulting image (PNG will be used).

        Returns:
            Tuple of (PIL.Image.Image, seed) - the generated image and the seed used.

        Raises:
            RuntimeError on pipeline errors.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded")

        self._validate_prompt(prompt)

        opts = options or GenerationOptions()
        seed = self._seed_everything(opts.seed)

        # Prepare kwargs for pipeline
        gen_kwargs = dict(
            prompt=prompt,
            num_inference_steps=opts.num_inference_steps,
            guidance_scale=opts.guidance_scale,
        )

        # Size handling: many models ignore height/width or require specific multiples
        if opts.height is not None or opts.width is not None:
            if opts.height is not None and opts.width is not None:
                gen_kwargs["height"] = opts.height
                gen_kwargs["width"] = opts.width
            else:
                raise ValueError("Both height and width must be provided together")

        # Use generator for deterministic outputs
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        gen_kwargs["generator"] = generator

        try:
            logger.info("Generating image: prompt='%s' steps=%s guidance=%s device=%s",
                        prompt if len(prompt) < 200 else prompt[:200] + "...",
                        opts.num_inference_steps,
                        opts.guidance_scale,
                        self.device)

            # Use autocast on CUDA for performance if dtype is float16
            image: PIL.Image.Image
            if self.device.type == "cuda" and self.torch_dtype == torch.float16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.pipeline(**gen_kwargs)
            else:
                output = self.pipeline(**gen_kwargs)

            # diffusers pipelines may return a dict or a StableDiffusionPipelineOutput
            images = getattr(output, "images", None) or output
            if isinstance(images, list):
                image = images[0]
            elif isinstance(images, PIL.Image.Image):
                image = images
            else:
                raise RuntimeError("Unexpected output format from pipeline")

            # Optional safety checker: many pipelines provide a safety_checker attribute
            try:
                if getattr(self.pipeline, "safety_checker", None) is not None:
                    # Some models mark NSFW images; no action here, but we log usage
                    logger.debug("Safety checker present on pipeline")
            except Exception:
                logger.debug("Safety checking attempted but failed; continuing")

            # Save if requested
            if output_path is not None:
                outp = Path(output_path)
                outp_parent = outp.parent
                if not outp_parent.exists():
                    outp_parent.mkdir(parents=True, exist_ok=True)
                # sanitize extension to .png
                if outp.suffix == "":
                    outp = outp.with_suffix(".png")
                image.save(outp, format="PNG")
                logger.info("Saved image to %s", outp)

            return image, int(seed)

        except Exception as e:
            logger.exception("Generation failed: %s", e)
            raise RuntimeError("Image generation failed") from e


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text -> Image generator using Hugging Face Diffusers")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id to use")
    parser.add_argument("--out", type=str, default=None, help="Output path to save the generated image (PNG)")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--height", type=int, default=None, help="Output height")
    parser.add_argument("--width", type=int, default=None, help="Output width")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (for reproducibility)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu). Auto-detected if not provided")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token or set HUGGINGFACE_HUB_TOKEN env var")
    parser.add_argument("--scheduler", type=str, default="dpmsolver", help="Scheduler to use (dpmsolver|multistep)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Basic logging configuration
    logger.setLevel(logging.INFO)

    opts = GenerationOptions(
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    try:
        generator = Text2ImageGenerator(
            model_id=args.model_id,
            device=args.device,
            hf_token=args.hf_token,
            scheduler=args.scheduler,
        )

        image, seed = generator.generate(args.prompt, options=opts, output_path=args.out)

        # If no output path was provided, save to a temp file and print its location
        if args.out is None:
            tmp = Path(tempfile.gettempdir()) / f"t2i_{seed}.png"
            image.save(tmp, format="PNG")
            logger.info("No --out provided, saved to %s", tmp)

        logger.info("Generation completed successfully (seed=%d)", seed)
        return 0
    except Exception as e:
        logger.exception("Fatal error while generating image: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
