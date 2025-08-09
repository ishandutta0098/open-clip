#!/usr/bin/env python3
"""
Text2Image: Lightweight CLI to generate images from text prompts using HuggingFace Diffusers.

This script is designed for production-like usage:
- Lazy model loading on first use
- GPU/CPU friendly with appropriate dtype selection
- Configurable prompt, negative prompt, and generation hyperparameters
- Safe defaults with input validation
- Reproducible results via seed
- Image saving with deterministic file naming based on prompt and timestamp

Usage:
  python text2image.py --prompt "A photo of a futuristic city" --output_dir outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def slugify(text: str) -> str:
    """Create a filesystem-safe slug from the given text."""
    text = text.strip().lower()
    # Replace non-alphanumeric characters with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", text)
    slug = slug.strip("-")
    if not slug:
        slug = "image"
    return slug


class TextToImageGenerator:
    """A wrapper around Diffusers StableDiffusionPipeline to generate images from prompts."""

    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", auth_token: Optional[str] = None) -> None:
        self.model_name = model_name
        self.auth_token = auth_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipe: Optional[StableDiffusionPipeline] = None
        logging.debug("TextToImageGenerator initialized with model='%s', device='%s', dtype=%s",
                      self.model_name, self.device, self.dtype)

    def _ensure_pipe_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            load_kwargs: dict = {"torch_dtype": self.dtype}
            if self.auth_token:
                load_kwargs["use_auth_token"] = self.auth_token
            self._pipe = StableDiffusionPipeline.from_pretrained(self.model_name, **load_kwargs)
            self._pipe = self._pipe.to(self.device)
            logging.info("Loaded model '%s' on %s with dtype %s", self.model_name, self.device, self.dtype)
        except Exception as exc:  # pragma: no cover
            logging.exception("Failed to load model '%s': %s", self.model_name, exc)
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: The primary text prompt describing the desired image.
            negative_prompt: Optional text guiding what to avoid.
            guidance_scale: CFG scale controlling prompt fidelity.
            num_inference_steps: Number of diffusion steps.
            width: Generated image width (pixels).
            height: Generated image height (pixels).
            seed: Optional seed for reproducibility.

        Returns:
            A PIL Image instance containing the generated image.
        """
        self._ensure_pipe_loaded()
        if self._pipe is None:
            raise RuntimeError("Model pipeline is not loaded.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # Some prompts can be long; keep a simple, robust call path
        if negative_prompt:
            result = self._pipe(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator,
            )
        else:
            result = self._pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator,
            )
        return result.images[0]

    @staticmethod
    def _make_multiple_of_8(n: int) -> int:
        """Ensure dimension is a multiple of 8 for compatibility with most SD models."""
        if n <= 0:
            return 8
        return (n // 8) * 8 if n % 8 == 0 else (n // 8) * 8


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using HuggingFace Diffusers (Stable Diffusion)."
    )
    parser.add_argument("--prompt", required=True, help="Text prompt describing the desired image.")
    parser.add_argument(
        "--negative_prompt", default=None, help="Optional negative prompt to steer content away."
    )
    parser.add_argument(
        "--model_name",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model identifier to use (e.g., runwayml/stable-diffusion-v1-5).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save the generated image(s).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps (inference steps).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width in pixels (must be multiple of 8 for most models).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height in pixels (must be multiple of 8 for most models).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for the diffusion process ( CFG ).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducibility.",
    )
    args = parser.parse_args()

    # Validate and adjust dimensions to be multiples of 8
    width = TextToImageGenerator._make_multiple_of_8(args.width)
    height = TextToImageGenerator._make_multiple_of_8(args.height)
    if width != args.width or height != args.height:
        logging.warning(
            "Adjusted image size to be multiple of 8: width=%d -> %d, height=%d -> %d",
            args.width,
            width,
            args.height,
            height,
        )

    # Prepare output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional HuggingFace auth token (env: HF_HUB_TOKEN or HUGGINGFACE_HUB_TOKEN)
    auth_token = os.environ.get("HF_HUB_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )

    prompt_slug = slugify(args.prompt)
    timestamp = int(time.time())
    filename = f"{prompt_slug}_{timestamp}.png"
    image_path = output_dir / filename

    try:
        generator = TextToImageGenerator(model_name=args.model_name, auth_token=auth_token)
        logging.info(
            "Starting image generation with model='%s', prompt='%s' (seed=%s)",
            args.model_name,
            args.prompt,
            args.seed,
        )
        image = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            width=width,
            height=height,
            seed=args.seed,
        )
        image.save(str(image_path))
        logging.info("Image saved to: %s", image_path)
    except Exception as exc:  # pragma: no cover
        logging.exception("Image generation failed: %s", exc)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
