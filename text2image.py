#!/usr/bin/env python3
"""text2image.py: A production-ready CLI tool for text-to-image generation
using HuggingFace Diffusers Stable Diffusion pipelines.

This script emphasizes:
- Robust error handling and logging
- GPU/CPU device detection with appropriate dtype selection
- Optional safety checker disablement (not recommended for production)
- Seed-based reproducibility and multiple image generation per prompt
- Clean, dependency-flexible design suitable for integration and scaling

Usage examples:
  python text2image.py --prompt "A dragon perched on a cliff at sunset" \
      --model-id "stabilityai/stable-diffusion-2-1" \
      --output-dir ./outputs --n-images 2 --steps 60 --width 768 --height 512
  python text2image.py --prompt "A sci-fi city skyline" --model-id runwayml/stable-diffusion-v1-5 \n      --disable-safety-checker --output-dir ./out --n-images 1

Note:
- The script tries to gracefully handle different versions of the Diffusers API
  by switching between forward() call patterns depending on what the installed
  version supports.
- For production deployments, keep the safety_checker enabled unless you have a
  clearly defined exception policy.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure a minimal yet informative logger for the CLI tool."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("diffusers").setLevel(logging.WARNING)


class Text2ImageService:
    """Wrapper around a Diffusers Stable Diffusion pipeline for text-to-image.

    This class encapsulates model loading, image generation, and optional
    safety-checker handling. It is designed to be GPU-friendly but gracefully
    degrades to CPU if needed.
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        use_safety_checker: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_safety_checker = use_safety_checker
        self.pipeline: Optional[StableDiffusionPipeline] = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load the pretrained pipeline with sane defaults for current device."""
        logging.info("Loading pipeline model_id=%s on device=%s", self.model_id, self.device)
        # Choose an appropriate dtype for memory/speed trade-offs
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
            )
            self.pipeline = self.pipeline.to(self.device)
            if not self.use_safety_checker:
                # Disable safety checker for production-only use-cases with policy
                # in place for content moderation outside this script.
                self.pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            logging.info("Pipeline loaded successfully.")
        except Exception as exc:  # pragma: no cover - defensive around IO/API errors
            logging.exception("Failed to load pipeline: %s", exc)
            raise

    def _generate_one(
        self,
        prompt: str,
        height: int,
        width: int,
        steps: int,
        guidance: float,
        seed: Optional[int],
    ) -> Image.Image:
        """Generate a single image for a given prompt.

        Attempts the modern forward() call first, and falls back to the older
        [prompt] list form if necessary to maintain compatibility across
        Diffusers versions.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        try:
            result = self.pipeline(
                prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            image = result.images[0]
        except TypeError:
            # Fallback for older/newer APIs that expect a list of prompts
            result = self.pipeline(
                [prompt],
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            image = result.images[0]
        return image

    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        steps: int = 50,
        guidance: float = 7.5,
        seed: Optional[int] = None,
        n_images: int = 1,
    ) -> List[Image.Image]:
        """Generate multiple images for a single prompt.

        Seeds are incremented if provided to ensure unique images across outputs.
        """
        images: List[Image.Image] = []
        for i in range(n_images):
            current_seed = seed + i if seed is not None else None
            img = self._generate_one(prompt, height, width, steps, guidance, current_seed)
            images.append(img)
        return images


def save_images(images: List[Image.Image], output_dir: str, base_name: str) -> List[str]:
    """Persist images to disk and return their file paths."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    for idx, img in enumerate(images, start=1):
        safe_name = base_name if base_name else "image"
        filename = f"{safe_name}_{idx:04d}.png"
        path = Path(output_dir) / filename
        img.save(path)
        saved_paths.append(str(path.resolve()))
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate images from a text prompt using HuggingFace Diffusers Stable Diffusion."
        ),
        epilog=(
            "Examples:\n"
            "  python text2image.py --prompt 'A futuristic city' --model-id stabilityai/stable-diffusion-2-1 --n-images 2\n"
            "  python text2image.py --prompt 'A dragon in armor' --disable-safety-checker --output-dir ./out"
        ),
    )
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Model identifier to load from HuggingFace Hub.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing the image to generate.")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save generated images.")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height in pixels (must be multiple of 8).")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width in pixels (must be multiple of 8).")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of denoising steps for the diffusion process.")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducible results.")
    parser.add_argument("--n-images", type=int, default=1,
                        help="Number of images to generate for the prompt.")
    parser.add_argument("--disable-safety-checker", action="store_true",
                        help="Disable the safety checker (not recommended for production).")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device to use (e.g., 'cuda' or 'cpu'). If not provided, auto-detects.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    logging.info("Starting text2image with prompt: %s", (args.prompt[:60] if len(args.prompt) > 60 else args.prompt))

    try:
        service = Text2ImageService(
            model_id=args.model_id,
            device=args.device,
            use_safety_checker=not args.disable_safety_checker,
        )
    except Exception:
        logging.exception("Failed to initialize the text-to-image service.")
        return 1

    try:
        images = service.generate(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance=args.guidance_scale,
            seed=args.seed,
            n_images=args.n_images,
        )
    except Exception:
        logging.exception("Image generation failed.")
        return 1

    # Create a filesystem-friendly base name from the prompt
    base_name = ("_".join(c for c in args.prompt if c.isalnum() or c in " _-")).strip().lower()
    base_name = base_name.replace(" ", "_")[:60] or "image"

    try:
        paths = save_images(images, args.output_dir, base_name)
    except Exception:
        logging.exception("Failed to save generated images to disk.")
        return 1

    logging.info("Saved %d image(s) to: %s", len(paths), paths)
    for p in paths:
        print(p)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
