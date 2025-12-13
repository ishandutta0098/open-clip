#!/usr/bin/env python3
"""
text2image.py

Command-line utility to generate images from text prompts using Hugging Face Diffusers

Features:
- Configurable model (Hugging Face model ID)
- Device selection (CUDA if available, else CPU)
- Mixed precision when GPU is available for performance
- Optional xFormers memory-efficient attention enablement
- Seeded generation for reproducibility
- Batch generation with multiple images per prompt
- Robust input validation, logging and error handling
- Saves images to output folder with deterministic filenames

Security and usage notes:
- If using a gated model, set the HF token in the environment variable HF_TOKEN or pass via CLI
- Validate prompts and resource limits before running in constrained environments

Example:
  python text2image.py --prompt "A steampunk city at sunset" --model "runwayml/stable-diffusion-v1-5" --num_images 2 --output_dir ./out --seed 42

"""
from __future__ import annotations

import argparse
import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import math

try:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
except Exception as exc:  # pragma: no cover - runtime import guard
    raise RuntimeError(
        "Missing required packages. Ensure requirements.txt dependencies are installed."
    ) from exc


# ----------------------------------------------------------------------------
# Configuration & utilities
# ----------------------------------------------------------------------------

logger = logging.getLogger("text2image")


@dataclass
class GenerationConfig:
    model_id: str
    prompt: str
    num_images: int = 1
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    output_dir: Path = Path("outputs")
    device: Optional[str] = None
    hf_token: Optional[str] = None
    enable_xformers: bool = True


def _validate_config(cfg: GenerationConfig) -> None:
    """Validate and normalize generation configuration.

    Raises ValueError on invalid inputs.
    """
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    if cfg.num_images < 1 or cfg.num_images > 20:
        # limit to avoid accidental huge runs
        raise ValueError("num_images must be between 1 and 20")

    if cfg.width <= 0 or cfg.height <= 0:
        raise ValueError("width and height must be positive integers")

    # Stable Diffusion models usually require multiples of 8
    if (cfg.width % 8) != 0 or (cfg.height % 8) != 0:
        raise ValueError("width and height must be divisible by 8")

    if cfg.guidance_scale < 1.0 or cfg.guidance_scale > 20.0:
        raise ValueError("guidance_scale should be between 1.0 and 20.0")

    if cfg.num_inference_steps < 1 or cfg.num_inference_steps > 200:
        raise ValueError("num_inference_steps should be between 1 and 200")

    if cfg.seed is not None and cfg.seed < 0:
        raise ValueError("seed must be non-negative")

    try:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ValueError(f"Could not create output directory '{cfg.output_dir}': {exc}")


# ----------------------------------------------------------------------------
# Core generation function
# ----------------------------------------------------------------------------

def generate_images(cfg: GenerationConfig) -> List[Path]:
    """
    Generate images for a text prompt using Stable Diffusion pipeline.

    Args:
        cfg: GenerationConfig with all relevant parameters.

    Returns:
        List of filesystem paths to generated images.
    """
    _validate_config(cfg)

    # Determine device and dtype
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    use_gpu = device.type == "cuda"
    torch_dtype = torch.float16 if use_gpu else torch.float32

    logger.info("Using device=%s, dtype=%s", device, torch_dtype)

    # Load model
    logger.info("Loading model '%s'", cfg.model_id)
    try:
        # Using token if provided (for gated models)
        hf_kwargs = {"use_auth_token": cfg.hf_token} if cfg.hf_token else {}

        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=torch_dtype,
            **hf_kwargs,
        )
    except Exception as exc:
        logger.exception("Failed to load model '%s': %s", cfg.model_id, exc)
        raise

    # Optimize for inference on GPU when available
    try:
        pipe = pipe.to(device)
    except Exception:
        # If conversion to device fails, release and re-raise with helpful message
        logger.exception("Failed to move pipeline to device %s", device)
        raise

    # Enable memory-efficient attention if available and requested
    if use_gpu and cfg.enable_xformers:
        try:
            # Some versions provide enable_xformers_memory_efficient_attention
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xFormers memory efficient attention")
            elif hasattr(pipe, "enable_attention_slicing"):
                # fallback optim: enable attention slicing to reduce peak memory
                pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing as fallback to xFormers")
        except Exception:
            logger.warning("Could not enable xFormers/attention optimizations; continuing without them")

    # If scheduler or safety components need setup, that would be done here (kept minimal)

    # Prepare generator for reproducibility
    generator = None
    if cfg.seed is not None:
        try:
            generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
        except Exception:
            # If device-specific generator fails (e.g., CPU float16), fall back to CPU generator
            generator = torch.Generator().manual_seed(int(cfg.seed))

    # Compose prompts list for batch generation
    prompts = [cfg.prompt] * cfg.num_images

    logger.info("Starting generation: images=%d, steps=%d, guidance=%.2f", cfg.num_images, cfg.num_inference_steps, cfg.guidance_scale)

    try:
        # The pipeline returns an object with .images (list of PIL images)
        result = pipe(
            prompts,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=int(cfg.num_inference_steps),
            guidance_scale=float(cfg.guidance_scale),
            generator=generator,
        )
        images = result.images
    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        raise

    saved_paths: List[Path] = []

    # Save images deterministically
    base_name = _sanitize_filename(cfg.prompt)[:64]
    for idx, img in enumerate(images, start=1):
        filename = f"{base_name}_{cfg.seed or "rand"}_{idx}.png"
        out_path = cfg.output_dir / filename
        try:
            # Ensure image is saved in PNG to keep quality; convert if needed
            if not isinstance(img, Image.Image):
                # Some pipelines may return numpy arrays
                img = Image.fromarray(img)
            img.save(out_path, format="PNG")
            saved_paths.append(out_path)
            logger.info("Saved image: %s", out_path)
        except Exception:
            logger.exception("Failed to save image to %s", out_path)

    return saved_paths


def _sanitize_filename(text: str) -> str:
    """Return safe string for use in filenames (simple sanitizer).

    Keeps alphanumeric, dash and underscore. Replaces others with underscore.
    """
    import re

    text = text.strip()
    # replace whitespace with underscore
    text = re.sub(r"\s+", "_", text)
    # remove characters other than alnum, dash, underscore
    text = re.sub(r"[^A-Za-z0-9_\-]", "", text)
    if not text:
        return "prompt"
    return text


# ----------------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Text -> Image generator using Hugging Face Diffusers")

    parser.add_argument("--model", "-m", dest="model_id", required=True, type=str,
                        help="Hugging Face Diffusers model id (e.g. 'runwayml/stable-diffusion-v1-5')")
    parser.add_argument("--prompt", "-p", dest="prompt", required=True, type=str,
                        help="Text prompt to generate an image for")
    parser.add_argument("--num_images", "-n", dest="num_images", default=1, type=int,
                        help="Number of images to generate for the prompt (default: 1, max 20)")
    parser.add_argument("--width", dest="width", default=512, type=int, help="Image width (multiple of 8)")
    parser.add_argument("--height", dest="height", default=512, type=int, help="Image height (multiple of 8)")
    parser.add_argument("--guidance_scale", dest="guidance_scale", default=7.5, type=float,
                        help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--steps", dest="num_inference_steps", default=50, type=int,
                        help="Number of denoising steps (default: 50)")
    parser.add_argument("--seed", dest="seed", default=None, type=int, help="Random seed (int) for reproducibility")
    parser.add_argument("--output_dir", dest="output_dir", default="outputs", type=str, help="Directory to write images")
    parser.add_argument("--device", dest="device", default=None, type=str, choices=[None, "cpu", "cuda"],
                        help="Device to run on, default auto-select (cuda if available else cpu)")
    parser.add_argument("--no_xformers", dest="no_xformers", action="store_true",
                        help="Disable xFormers auto-optimization even if available")
    parser.add_argument("--hf_token", dest="hf_token", default=None, type=str,
                        help="Hugging Face token (optional) or set HF_TOKEN env var")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    # Merge HF token from env var if not explicitly provided
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    # Normalize device argument
    device_arg = None
    if args.device == "cuda":
        device_arg = "cuda"
    elif args.device == "cpu":
        device_arg = "cpu"

    cfg = GenerationConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        device=device_arg,
        hf_token=hf_token,
        enable_xformers=(not args.no_xformers),
    )

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=level)

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    try:
        cfg = parse_args(argv)
        saved = generate_images(cfg)
        if saved:
            logger.info("Generation completed. %d images saved to: %s", len(saved), cfg.output_dir)
            for p in saved:
                print(p)
        else:
            logger.warning("Generation finished but no images were saved")
        return 0
    except Exception as exc:
        logger.exception("Failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
