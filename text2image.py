#!/usr/bin/env python3
"""
text2image.py

A production-ready CLI and library-style script to generate images from text prompts
using Hugging Face Diffusers.

Features:
- Device auto-selection (CUDA/CPU)
- Mixed precision on GPU for performance and memory savings
- Deterministic seeding support
- Input validation and secure handling of authentication tokens
- Detailed logging and error handling

Usage (example):
  export HF_TOKEN="<your_hf_token>"
  python text2image.py --prompt "A scenic painting of a futuristic city at sunset" --out_dir ./outputs

For more options run:
  python text2image.py --help

Security notes:
- Provide your Hugging Face token via the HF_TOKEN environment variable. Avoid embedding tokens
  in source code or command-line history when possible.

"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch is required. Install via 'pip install torch'.") from e

try:
    from diffusers import DiffusionPipeline
except Exception as e:
    raise RuntimeError("diffusers is required. Install via 'pip install diffusers'.") from e

from PIL import Image

# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("text2image")


@dataclass
class GenerationConfig:
    prompt: str
    model: str
    out_dir: Path
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    seed: Optional[int]
    num_images_per_prompt: int
    device: str
    torch_dtype: Optional[Any]
    auth_token: Optional[str]


def _select_device(preferred: Optional[str] = None) -> str:
    """Select a device string for torch and diffusers.

    Args:
        preferred: Optional user-preferred device string (e.g., 'cuda' or 'cpu').

    Returns:
        A device string ('cuda' if available and preferred, otherwise 'cpu').
    """
    if preferred:
        pref = preferred.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return "cuda"
        if pref == "cpu":
            return "cpu"
        # fallthrough to auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_dimensions(width: int, height: int) -> None:
    """Validate model dimension constraints (Stable Diffusion: divisible by 8).

    Raises:
        ValueError if invalid.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    # Most latent diffusion models expect multiples of 8 (or 64 for some newer ones);
    # we check for 8 which is safe/common.
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("width and height must be divisible by 8 for most diffusion models")


def _seed_generator(seed: Optional[int]) -> torch.Generator:
    """Create a torch.Generator seeded deterministically if seed provided.

    Args:
        seed: Optional seed. If None, create generator with non-deterministic seed.

    Returns:
        torch.Generator configured with the seed.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        gen.manual_seed(seed)
    else:
        # Leave generator unseeded for non-deterministic behaviour
        gen.seed()
    return gen


def load_pipeline(config: GenerationConfig) -> DiffusionPipeline:
    """Load a diffusion pipeline with sensible defaults and optimizations.

    Args:
        config: GenerationConfig with model, device, dtype, and auth token.

    Returns:
        An initialized DiffusionPipeline ready to generate images.

    Raises:
        RuntimeError for loading failures.
    """
    logger.info("Loading model '%s' on device %s", config.model, config.device)

    # Use HF token from env if provided in config; do not print tokens to logs
    use_auth_token = config.auth_token if config.auth_token else None

    # Choose dtype
    torch_dtype = config.torch_dtype

    try:
        pipeline = DiffusionPipeline.from_pretrained(
            config.model,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Explicitly disable built-in safety checker for control; callers may re-enable or sanitize.
            use_auth_token=use_auth_token,
        )
    except TypeError:
        # Older/newer diffusers versions may not accept use_auth_token param; fallback
        pipeline = DiffusionPipeline.from_pretrained(config.model, torch_dtype=torch_dtype)

    # Move to device
    try:
        pipeline = pipeline.to(config.device)
    except Exception as exc:
        logger.warning("Failed to move pipeline to device %s: %s. Continuing on CPU.", config.device, exc)
        pipeline = pipeline.to("cpu")

    # Try to enable memory efficient attention if available
    try:
        # xformers may not be installed; enable if available
        pipeline.enable_xformers_memory_efficient_attention()
        logger.debug("Enabled xFormers memory efficient attention")
    except Exception:
        logger.debug("xFormers not available or failed to enable; continuing without it")

    # Enable attention slicing to reduce peak memory at the cost of some speed
    try:
        pipeline.enable_attention_slicing()
        logger.debug("Enabled attention slicing for reduced memory usage")
    except Exception:
        logger.debug("Could not enable attention slicing")

    return pipeline


def _safe_filename(prompt: str, seed: Optional[int], ext: str = "png") -> str:
    """Create a deterministic, filesystem-safe filename from prompt and seed.

    Args:
        prompt: Text prompt.
        seed: Optional seed.
        ext: File extension without dot.

    Returns:
        Filename string.
    """
    # Keep it short: use SHA256 of prompt+seed
    h = hashlib.sha256((prompt + (str(seed) if seed is not None else "")).encode("utf-8")).hexdigest()
    short = h[:12]
    safe = f"img_{short}.{ext}"
    return safe


def generate_images(
    config: GenerationConfig,
    pipeline: DiffusionPipeline,
) -> List[Path]:
    """Generate one or more images from a text prompt using the provided pipeline.

    Args:
        config: GenerationConfig object.
        pipeline: Loaded DiffusionPipeline.

    Returns:
        List of Path objects where images were saved.
    """
    _validate_dimensions(config.width, config.height)

    # Prepare output directory
    os.makedirs(config.out_dir, exist_ok=True)

    # Prepare generator
    device_for_gen = "cpu" if config.device == "cpu" else config.device
    generator = _seed_generator(config.seed)

    # If device is cuda and generator is needed on cuda, create cuda generator
    if config.device == "cuda" and config.seed is not None:
        # Create CUDA generator for deterministic sampling on GPU
        gen = torch.Generator(device="cuda")
        gen.manual_seed(config.seed)
        generator = gen

    logger.info(
        "Generating %d image(s) with steps=%d guidance=%.2f size=%dx%d",
        config.num_images_per_prompt,
        config.num_inference_steps,
        config.guidance_scale,
        config.width,
        config.height,
    )

    images: List[Image.Image] = []

    try:
        # DiffusionPipeline.__call__ accepts generator per image when given a list.
        output = pipeline(
            prompt=[config.prompt] * config.num_images_per_prompt,
            height=config.height,
            width=config.width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        )
        pil_images = output.images
        images = pil_images
    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        raise

    saved_paths: List[Path] = []
    for idx, img in enumerate(images):
        # Create a deterministic but unique name per image
        fname = _safe_filename(config.prompt + f"_{idx}", config.seed)
        out_path = config.out_dir / fname
        # Validate path not escaping out_dir
        try:
            out_path.resolve().relative_to(config.out_dir.resolve())
        except Exception:
            # If path traversal detected, fallback to safe filename
            out_path = config.out_dir / f"{Path(fname).name}"
        # Save as PNG
        img.save(out_path, format="PNG")
        saved_paths.append(out_path)
        logger.info("Saved image: %s", out_path)

    return saved_paths


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from text using Hugging Face diffusers")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate the image from")
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face Diffusers model identifier (default: runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("./outputs"), help="Directory to save generated images")
    parser.add_argument("--height", type=int, default=512, help="Output image height (must be divisible by 8)")
    parser.add_argument("--width", type=int, default=512, help="Output image width (must be divisible by 8)")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic outputs")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate per prompt")
    parser.add_argument("--device", type=str, default=None, help="Device to run on: 'cuda' or 'cpu' (auto by default)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (prefer HF_TOKEN env var instead)")
    parser.add_argument("--no_torch_cuda", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--verbosity", type=str, default="info", choices=["debug", "info", "warning", "error"], help="Logging verbosity")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Configure logging level
    logger.setLevel(getattr(logging, args.verbosity.upper(), logging.INFO))

    # Determine device
    preferred = None if args.device is None else args.device
    device = _select_device(preferred)
    if args.no_torch_cuda:
        device = "cpu"

    # Determine torch dtype
    torch_dtype = None
    if device == "cuda":
        # Use mixed precision for GPU to save VRAM and increase throughput
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Auth token preference: CLI arg wins, fallback to HF_TOKEN environment variable
    auth_token = args.hf_token or os.getenv("HF_TOKEN")

    # Build config
    config = GenerationConfig(
        prompt=args.prompt,
        model=args.model,
        out_dir=args.out_dir,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_images_per_prompt=max(1, args.num_images),
        device=device,
        torch_dtype=torch_dtype,
        auth_token=auth_token,
    )

    # Log runtime configuration without sensitive data
    logger.info("Starting generation; model=%s device=%s images=%d", config.model, config.device, config.num_images_per_prompt)

    start_time = time.time()
    try:
        pipeline = load_pipeline(config)
        saved = generate_images(config, pipeline)
    except Exception as exc:
        logger.exception("Generation pipeline failed: %s", exc)
        return 2

    duration = time.time() - start_time
    logger.info("Completed generation in %.2f seconds. Saved %d file(s).", duration, len(saved))

    # Print saved paths to stdout for easy piping
    for p in saved:
        print(str(p))

    return 0


if __name__ == "__main__":
    sys.exit(main())
