#!/usr/bin/env python3
"""
text2image.py

A production-ready CLI utility to generate images from text prompts using Hugging Face Diffusers.

Features:
- Configurable model, prompt, size, steps, guidance scale, seed, and number of images
- Automatic device selection (GPU if available, otherwise CPU)
- Performance optimizations: attention slicing, xformers (if available), fp16 on CUDA
- Robust input validation, error handling, and logging
- Saves images with deterministic filenames (timestamp + seed + index)

Usage example:
  export HUGGINGFACE_TOKEN=your_hf_token  # required if using gated models
  python text2image.py --prompt "A sci-fi cityscape at twilight" --num-images 2 --out-dir ./outputs

"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
from PIL import Image

try:
    import torch
    from diffusers import StableDiffusionPipeline
except Exception as exc:  # pragma: no cover - graceful failure when deps missing
    raise RuntimeError(
        "Missing required packages. Please install dependencies from requirements.txt. "
        "Original error: {}".format(exc)
    )

# Module-level logger
logger = logging.getLogger("text2image")


@dataclass
class GenerationConfig:
    model_id: str
    prompt: str
    out_dir: Path
    num_images: int = 1
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token: Optional[str] = None


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for console output.

    Args:
        level: Logging level (default INFO).
    """
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    # avoid adding multiple handlers in interactive runs
    if not root.handlers:
        root.addHandler(handler)
    else:
        # replace handlers to ensure consistent formatting
        root.handlers = [handler]


def _slugify(text: str, max_len: int = 50) -> str:
    """Create a filesystem-safe short slug from text.

    Args:
        text: Input string.
        max_len: Max length of slug.

    Returns:
        Sanitized slug.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "prompt"


def _generate_filename_slug(prompt: str, seed: Optional[int]) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(prompt, max_len=40)
    seedpart = f"s{seed}" if seed is not None else "s-"
    # include hash of prompt to avoid collisions for similar prompts
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{timestamp}_{slug}_{digest}_{seedpart}"


def validate_config(cfg: GenerationConfig) -> None:
    """Validate the generation config and raise ValueError on invalid inputs.

    Args:
        cfg: GenerationConfig to validate.
    """
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    if cfg.num_images < 1 or cfg.num_images > 16:
        raise ValueError("num_images must be between 1 and 16")
    if not (64 <= cfg.height <= 2048 and 64 <= cfg.width <= 2048):
        raise ValueError("height and width must be between 64 and 2048")
    if not (1 <= cfg.num_inference_steps <= 500):
        raise ValueError("num_inference_steps must be between 1 and 500")
    if not (0.0 <= cfg.guidance_scale <= 30.0):
        raise ValueError("guidance_scale must be between 0 and 30")


def get_generator(seed: Optional[int], device: str) -> Optional[torch.Generator]:
    """Return a torch Generator for deterministic sampling when seed provided.

    Args:
        seed: Optional seed.
        device: Device string for generator allocation.

    Returns:
        torch.Generator or None
    """
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def load_pipeline(model_id: str, device: str, hf_token: Optional[str], torch_dtype: Optional[torch.dtype]) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline with sensible performance settings.

    Args:
        model_id: HF model id (e.g., "runwayml/stable-diffusion-v1-5")
        device: Device string ("cuda" or "cpu").
        hf_token: Optional HuggingFace token for gated models.
        torch_dtype: dtype for model weights (e.g., torch.float16) or None.

    Returns:
        Instantiated StableDiffusionPipeline ready for inference.
    """
    logger.info("Loading model '%s' on device=%s", model_id, device)
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # explicit: models may not include a safety checker; caller should be aware
            use_auth_token=hf_token,
        )
    except TypeError:
        # older versions of diffusers may not accept use_auth_token param name
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

    # Move to device
    pipeline = pipeline.to(device)

    # Enable attention slicing to reduce memory footprint
    try:
        pipeline.enable_attention_slicing()
        logger.debug("Enabled attention slicing")
    except Exception:
        logger.debug("Attention slicing not supported by this pipeline version")

    # Try to enable xformers memory efficient attention, if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.debug("Enabled xformers memory efficient attention")
    except Exception:
        logger.debug("xformers not available or couldn't be enabled")

    return pipeline


def generate_images(cfg: GenerationConfig) -> List[Path]:
    """Generate images from a text prompt based on the configuration.

    Args:
        cfg: GenerationConfig with generation parameters.

    Returns:
        List of saved image file paths.
    """
    validate_config(cfg)

    # Choose dtype: use float16 on CUDA for speed and memory
    torch_dtype = torch.float16 if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else torch.float32

    pipeline = load_pipeline(cfg.model_id, cfg.device, cfg.hf_token, torch_dtype)

    # Generator for deterministic output
    generator = get_generator(cfg.seed, cfg.device) if cfg.seed is not None else None

    results: List[Path] = []

    # Create output directory
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Note: use inference_mode for speed, autocast for fp16 when on CUDA
    try:
        if cfg.device.startswith("cuda"):
            # Use autocast for appropriate dtype when available
            context_manager = torch.autocast(cfg.device, dtype=torch_dtype) if torch_dtype == torch.float16 else torch.no_grad()
        else:
            context_manager = torch.no_grad()
    except Exception:
        context_manager = torch.no_grad()

    prompt = cfg.prompt
    file_slug_base = _generate_filename_slug(prompt, cfg.seed)

    logger.info("Generating %d image(s) for prompt: %s", cfg.num_images, prompt)

    with context_manager:
        for i in range(cfg.num_images):
            logger.debug("Generating image %d/%d", i + 1, cfg.num_images)
            try:
                output = pipeline(
                    prompt,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    generator=generator,
                )
            except Exception as exc:
                logger.exception("Model inference failed: %s", exc)
                raise

            image = output.images[0]
            filename = f"{file_slug_base}_{i+1}.png"
            path = cfg.out_dir / filename

            try:
                image.save(path, format="PNG")
                results.append(path)
                logger.info("Saved image: %s", path)
            except Exception as exc:
                logger.exception("Failed to save image to %s: %s", path, exc)
                raise

    return results


def parse_args(argv: Optional[List[str]] = None) -> GenerationConfig:
    """Parse CLI arguments and construct GenerationConfig.

    Args:
        argv: Optional argv list for testing.

    Returns:
        GenerationConfig populated from CLI.
    """
    parser = argparse.ArgumentParser(description="Generate images from text using Hugging Face Diffusers")

    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id to use")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate an image from")
    parser.add_argument("--out-dir", default="outputs", help="Directory to save generated images")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate (1-16)")
    parser.add_argument("--height", type=int, default=512, help="Image height (pixels)")
    parser.add_argument("--width", type=int, default=512, help="Image width (pixels)")

    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for deterministic output")

    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device to run on (defaults to GPU if available)")

    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated models. Can also be provided via HUGGINGFACE_TOKEN env var.")

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    # Determine device preference
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve HF token (cli overrides env)
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    cfg = GenerationConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        out_dir=Path(args.out_dir),
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device,
        hf_token=hf_token,
    )

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    """Main entrypoint for CLI.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code (0 on success)
    """
    try:
        cfg = parse_args(argv)
        paths = generate_images(cfg)
        logger.info("Generation completed successfully. %d file(s) saved.", len(paths))
        return 0
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover - run as script
    raise SystemExit(main())
