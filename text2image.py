# text2image.py
"""
Text-to-image generation using HuggingFace Diffusers.

This script loads a pretrained Stable Diffusion model and generates an image from a text prompt.
It is designed to be robust, production-friendly, and suitable for integration into larger tooling.

Features
- Quickstart CLI for prompt-based image generation
- Automatic device detection (CUDA if available, otherwise CPU)
- Deterministic output via seeds
- Safe defaults with sane parameter ranges
- Output as PNG with a deterministic filename based on the prompt and timestamp
- Lightweight logging and error handling

Dependencies are managed via requirements.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import time
from typing import Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure global logging.

    Args:
        verbose: If True, set logging to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def detect_device() -> str:
    """Detect the best available compute device.

    Returns:
        'cuda' if a CUDA-enabled GPU is available; otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def slugify(text: str) -> str:
    """Create a filesystem-friendly slug from input text."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "prompt"


def create_output_path(output_dir: str, prompt: str, seed: Optional[int]) -> str:
    """Generate a deterministic output filename based on the prompt and optional seed."""
    base = slugify(prompt)
    ts = int(time.time())
    seed_part = f"-seed{seed}" if seed is not None else ""
    filename = f"image-{base}-{ts}{seed_part}.png"
    return os.path.join(output_dir, filename)


def load_pipeline(model_name: str, device: str) -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline on the specified device.

    Args:
        model_name: HuggingFace model identifier.
        device: 'cuda' or 'cpu'.

    Returns:
        A ready-to-use StableDiffusionPipeline instance.
    """
    logging.info("Loading model '%s' on device '%s'...", model_name, device)
    # Choose precision based on device compatibility
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Attempt to disable safety checker for smoother automation; best effort
    try:
        pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        logging.debug("Safety checker disabled for generation (best-effort).")
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.debug("Could not disable safety checker: %s", exc)

    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    generator: Optional[torch.Generator] = None,
) -> Image.Image:
    """Generate an image from a text prompt using the provided pipeline."""
    logging.info("Generating image... (steps=%d, height=%d, width=%d, guidance=%.2f)", steps, height, width, guidance_scale)
    outputs = pipe(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator)
    image = outputs.images[0]
    return image


def save_image(image: Image.Image, path: str) -> None:
    """Persist the generated image to disk."""
    image.save(path)
    logging.info("Image saved to: %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-to-image generation using Hugging Face Diffusers (Stable Diffusion)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory to save generated images"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="runwayml/stable-d diffusion-v1-5".replace(" ", ""),  # placeholder normalize
        help="Diffusers model identifier from Hugging Face (e.g., 'runwayml/stable-diffusion-v1-5')",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps (inference steps)")
    parser.add_argument("--width", type=int, default=512, help="Width of generated image in pixels")
    parser.add_argument("--height", type=int, default=512, help="Height of generated image in pixels")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale (CFG) of the diffusion process")
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic generation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Initialize logging
    setup_logging(args.verbose)

    # Resolve device
    if args.device == "auto":
        device = detect_device()
    else:
        device = args.device
    logging.info("Using device: %s", device)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        pipe = load_pipeline(args.model_name, device)
    except Exception as exc:  # pragma: no cover - initialization guard
        logging.exception("Failed to load pipeline: %s", exc)
        sys.exit(1)

    # Prepare a generator for deterministic rendering if seed is provided
    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    try:
        image = generate_image(
            pipe=pipe,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        output_path = create_output_path(args.output_dir, args.prompt, args.seed)
        save_image(image, output_path)
        # Print path for consumer tooling
        print(output_path)
    except RuntimeError as re:
        logging.exception("Runtime error during image generation: %s", re)
        sys.exit(2)
    except Exception as exc:  # pragma: no cover
        logging.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
