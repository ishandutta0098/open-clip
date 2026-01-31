#!/usr/bin/env python3
"""
text2image.py

A production-ready, configurable command-line tool to generate images from text prompts
using Hugging Face Diffusers (Stable Diffusion pipelines).

Features:
- CLI with robust argument parsing and validation
- Device auto-detection (CUDA/CPU) with mixed-precision where available
- HF token support via env or CLI for gated models
- Retry + OOM handling with fallback resolution reduction
- Deterministic seeding support
- Clear logging and well-typed functions with Google-style docstrings

Usage example:
    export HF_TOKEN=your_token_here
    python text2image.py --prompt "A scenic sunrise over a mountain lake" --out output.png

"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
)
logger.addHandler(_handler)


@dataclass
class GenerationConfig:
    """Configuration for image generation.

    Attributes:
        model_id: Hugging Face model identifier for a Stable Diffusion model.
        prompt: Main text prompt.
        negative_prompt: Optional negative prompt to steer away from undesired concepts.
        out_path: Output file path for the generated image.
        seed: Random seed for reproducibility. If None, randomized.
        guidance_scale: Classifier-free guidance scale (higher -> more adherence to prompt).
        num_inference_steps: Number of denoising steps.
        height: Image height in pixels. Must be multiple of 8 or 16 depending on model.
        width: Image width in pixels.
        device: Which device to run on (auto / cpu / cuda).
        hf_token: Hugging Face token for gated models. If None, will use HF_TOKEN env var.
        dtype: Torch dtype to use where applicable.
    """

    model_id: str
    prompt: str
    negative_prompt: Optional[str]
    out_path: str
    seed: Optional[int]
    guidance_scale: float
    num_inference_steps: int
    height: int
    width: int
    device: str
    hf_token: Optional[str]
    dtype: torch.dtype


def parse_args(argv: Optional[list[str]] = None) -> GenerationConfig:
    """Parse CLI args and return a GenerationConfig.

    Args:
        argv: Provide a list to parse (for testing); defaults to sys.argv[1:].
    Returns:
        A validated GenerationConfig.
    """
    parser = argparse.ArgumentParser(
        prog="text2image",
        description="Generate images from text using Hugging Face Diffusers (Stable Diffusion).",
    )

    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face model id for Stable Diffusion (default: runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the desired image")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt to avoid concepts")
    parser.add_argument("--out", "-o", dest="out_path", default="output.png",
                        help="Path to write the resulting image (png/jpg)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (default 7.5)")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps (default 30)")
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels (default 512)")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels (default 512)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Compute device: auto/cpu/cuda (default auto)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto",
                        help="Precision to run in. 'auto' chooses fp16 on CUDA where supported")

    args = parser.parse_args(argv)

    # Validate image dimensions (some models require multiples of 8 or 64; 8 is safe for many)
    if args.height <= 0 or args.width <= 0:
        parser.error("height and width must be positive integers")
    if args.height % 8 != 0 or args.width % 8 != 0:
        parser.error("height and width should be multiples of 8 for most SD models")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # resolve dtype
    if args.dtype == "auto":
        dtype = torch.float16 if (torch.cuda.is_available()) else torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    config = GenerationConfig(
        model_id=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        out_path=args.out_path,
        seed=args.seed,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        device=args.device,
        hf_token=hf_token,
        dtype=dtype,
    )

    return config


def select_device(requested: str) -> str:
    """Select best device string based on requested option and availability.

    Args:
        requested: 'auto', 'cpu', or 'cuda'
    Returns:
        Chosen device string ("cpu" or "cuda").
    """
    if requested == "cpu":
        logger.info("Using CPU as requested.")
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return "cuda"
    # auto
    if torch.cuda.is_available():
        logger.info("CUDA available. Using CUDA.")
        return "cuda"
    logger.info("CUDA not available. Using CPU.")
    return "cpu"


def load_pipeline(model_id: str, device: str, dtype: torch.dtype, hf_token: Optional[str]) -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline with sensible defaults.

    The function selects a robust scheduler and sets up the pipeline to run on the selected device.

    Args:
        model_id: Hugging Face model repo id.
        device: 'cuda' or 'cpu'.
        dtype: torch.dtype for model weights.
        hf_token: Optional Hugging Face token for private/gated models.

    Returns:
        An initialized StableDiffusionPipeline.
    """
    logger.info("Loading pipeline for model: %s", model_id)

    # Use DPMSolverMultistepScheduler for faster convergence / better sample quality in many cases
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            revision=None,
            use_auth_token=hf_token,
        )
    except TypeError:
        # older diffusers may not accept use_auth_token param name; try without it
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # set scheduler to DPMSolverMultistep for improved sampling by default
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # Move to device
    pipeline = pipeline.to(device)

    # Optimize attention / memory where supported
    try:
        pipeline.enable_attention_slicing()
        logger.debug("Enabled attention slicing to reduce memory usage.")
    except Exception:
        logger.debug("Attention slicing not supported on this pipeline version.")

    return pipeline


def save_image(img: Image.Image, out_path: str) -> None:
    """Persist the generated PIL image to disk protecting against accidental overwrite.

    Args:
        img: PIL Image to save.
        out_path: Destination filepath.
    """
    directory = os.path.dirname(out_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # If path exists, append timestamp to avoid overwriting unless user explicitly sets same path
    final_path = out_path
    if os.path.exists(final_path):
        base, ext = os.path.splitext(out_path)
        timestamp = int(time.time())
        final_path = f"{base}_{timestamp}{ext}"
        logger.warning("Output path exists. Writing to %s instead", final_path)

    img.save(final_path)
    logger.info("Saved image to %s", final_path)


def generate(
    cfg: GenerationConfig,
    max_retries: int = 2,
    reduce_factor: float = 0.75,
) -> Tuple[Image.Image, dict]:
    """Generate an image from the given configuration with OOM handling and retries.

    Args:
        cfg: GenerationConfig with all parameters.
        max_retries: How many attempts to make if we hit CUDA OOMs; each retry reduces resolution.
        reduce_factor: Multiplier to reduce height/width after an OOM.

    Returns:
        Generated PIL.Image and pipeline info dict.
    """
    device = select_device(cfg.device)

    # Load pipeline
    pipeline = load_pipeline(cfg.model_id, device, cfg.dtype, cfg.hf_token)

    # Enforce seed determinism
    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        logger.info("Using seed: %d", cfg.seed)

    attempt = 0
    height, width = cfg.height, cfg.width

    while True:
        try:
            logger.info("Generating (attempt %d) with resolution %dx%d, steps=%d, guidance=%.2f",
                        attempt + 1, height, width, cfg.num_inference_steps, cfg.guidance_scale)

            # Many pipelines accept height/width arguments; we forward them.
            # Use autocast for mixed precision on CUDA where fp16 is selected
            if device == "cuda" and cfg.dtype == torch.float16:
                context = torch.autocast("cuda")
            else:
                # no-op context manager
                class _NullCtx:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                context = _NullCtx()

            with context:
                result = pipeline(
                    prompt=cfg.prompt,
                    negative_prompt=cfg.negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    generator=generator,
                )

            image = result.images[0]
            metadata = {
                "nsfw_content_detected": getattr(result, "nsfw_content_detected", None),
                "safety_metadata": None,
            }
            logger.info("Generation successful on attempt %d", attempt + 1)
            return image, metadata

        except torch.cuda.OutOfMemoryError as oom:
            logger.error("CUDA OutOfMemoryError: %s", oom)
            attempt += 1
            if attempt > max_retries:
                logger.exception("Out of memory and exceeded max retries. Consider lowering resolution or using CPU.")
                raise
            # attempt fallback: reduce resolution
            height = max(64, int(height * reduce_factor))
            width = max(64, int(width * reduce_factor))
            logger.warning("Retrying with reduced resolution %dx%d (attempt %d/%d)",
                           height, width, attempt, max_retries)
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            logger.exception("Image generation failed: %s", exc)
            raise


def main(argv: Optional[list[str]] = None) -> int:
    """Main entrypoint for CLI.

    Returns:
        Exit code (0 on success).
    """
    cfg = parse_args(argv)

    # Basic safety: warn if no HF token provided and model is gated
    if cfg.hf_token is None:
        logger.debug("No HF token provided via env or --hf-token. Public models will work but gated models will fail.")

    try:
        image, meta = generate(cfg)
        save_image(image, cfg.out_path)

        # Warn if NSFW content detected (some pipelines provide this)
        nsfw = meta.get("nsfw_content_detected")
        if nsfw is not None and any(nsfw):
            logger.warning("Model flagged NSFW content in generation. Take care with publishing this image.")

        return 0
    except Exception as exc:
        logger.error("Failed to generate image: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
