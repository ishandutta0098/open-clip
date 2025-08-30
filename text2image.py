#!/usr/bin/env python3
"""
text2image.py

Command-line utility to generate images from text prompts using Hugging Face Diffusers

Features:
- Auto device detection (CUDA if available, otherwise CPU)
- Optional Hugging Face token usage for private models
- Input validation for prompt and resolution
- Deterministic generation using seed
- Configurable inference parameters (steps, guidance_scale, scheduler)
- Safe defaults and comprehensive error handling & logging

Usage example:
python text2image.py --prompt "A fantasy landscape, vibrant colors" --out outputs/landscape.png --width 768 --height 512 --steps 25 --seed 42

"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Diffusers imports are done lazily at runtime to allow early error messages if not present
try:
    from diffusers import (
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        LMSDiscreteScheduler,
        EulerDiscreteScheduler,
    )
except Exception as e:  # pragma: no cover - runtime environment dependent
    raise RuntimeError(
        "Could not import diffusers. Ensure required packages are installed (see requirements.txt)."
    ) from e

# Module-level logger
logger = logging.getLogger("text2image")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str
    negative_prompt: Optional[str]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    seed: Optional[int]
    model_id: str
    scheduler: str
    device: str
    hf_token: Optional[str]
    out: Path


def parse_args(argv: Optional[list[str]] = None) -> GenerationConfig:
    """Parse command-line arguments into a GenerationConfig.

    Args:
        argv: Optional list of CLI args for testing. If None, sys.argv is used.

    Returns:
        GenerationConfig populated from parsed args.
    """
    parser = argparse.ArgumentParser(description="Text -> Image using Hugging Face Diffusers")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to render")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt (optional)")
    parser.add_argument("--out", type=Path, required=True, help="Output file path (png/jpeg)." )
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model repo id")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device to use")
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=30, help="Number of inference steps (25-50 recommended)")
    parser.add_argument("--guidance", dest="guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8 recommended)")
    parser.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8 recommended)")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic output")
    parser.add_argument("--scheduler", type=str, default="dpmsolver", choices=["dpmsolver", "lms", "euler"], help="Scheduler choice")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token (or set HUGGINGFACE_HUB_TOKEN env var)")
    args = parser.parse_args(argv)

    # Resolve token from environment if not provided explicitly
    hf_token = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Device autodetection
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_path: Path = args.out
    if out_path.is_dir():
        raise ValueError(f"Output path {out_path} is a directory; specify a file path (e.g. outputs/image.png)")

    return GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        model_id=args.model_id,
        scheduler=args.scheduler,
        device=device,
        hf_token=hf_token,
        out=out_path,
    )


def validate_config(cfg: GenerationConfig) -> None:
    """Validate user configuration and raise a clear error for invalid combinations.

    Ensures prompts are not empty, resolution is valid, and step counts are in reasonable ranges.
    """
    if not cfg.prompt.strip():
        raise ValueError("Prompt must not be empty")

    if cfg.height <= 0 or cfg.width <= 0:
        raise ValueError("Width and height must be positive integers")

    if cfg.width % 8 != 0 or cfg.height % 8 != 0:
        logger.warning("Recommended: width/height should be multiples of 8 for most models; continuing anyway.")

    if not (1 <= cfg.num_inference_steps <= 200):
        raise ValueError("num_inference_steps must be between 1 and 200")

    if not (0.0 <= cfg.guidance_scale <= 30.0):
        raise ValueError("guidance_scale must be between 0.0 and 30.0")


def choose_scheduler(name: str, pipeline) -> None:
    """Replace the pipeline scheduler based on name chosen by the user.

    Args:
        name: one of 'dpmsolver', 'lms', 'euler'
        pipeline: the loaded pipeline instance
    """
    name = name.lower()
    if name == "dpmsolver":
        scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif name == "lms":
        scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif name == "euler":
        scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    pipeline.scheduler = scheduler


def load_pipeline(model_id: str, device: str, hf_token: Optional[str], pretrained_dtype: Optional[torch.dtype] = None):
    """Load the Stable Diffusion pipeline with sensible defaults.

    - Uses float16 when on CUDA to reduce VRAM usage
    - Uses CPU-friendly dtype when on CPU

    Returns:
        Initialized StableDiffusionPipeline
    """
    logger.info("Loading model %s on device=%s", model_id, device)

    # Choose dtype
    if pretrained_dtype is None:
        pretrained_dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

    # from_pretrained will download/cache the model; passing use_auth_token if provided
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=pretrained_dtype,
            use_auth_token=hf_token,
        )
    except TypeError:
        # Some versions expect `use_auth_token` to be a keyword or not present
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=pretrained_dtype)

    # Move to desired device
    pipeline = pipeline.to(device)

    # Disable the safety checker (explicitly) — note: you should implement your own safety if required
    # Many recent diffusers versions expose a safety checker; if present, we keep default behavior.
    try:
        # pipeline.safety_checker might be present; if you want to enforce default, keep as-is.
        pass
    except Exception:
        # ignore if not present
        pass

    return pipeline


def _seed_generator(seed: Optional[int]) -> torch.Generator:
    if seed is None:
        # nondeterministic
        g = torch.Generator()
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))
    return g


def generate_image(cfg: GenerationConfig) -> Path:
    """Generate an image according to the provided configuration and save it.

    Returns:
        Path to the saved image file
    """
    validate_config(cfg)

    # Load pipeline
    dtype = torch.float16 if (cfg.device == "cuda" and torch.cuda.is_available()) else torch.float32
    pipeline = load_pipeline(cfg.model_id, cfg.device, cfg.hf_token, pretrained_dtype=dtype)

    # Replace scheduler if requested
    choose_scheduler(cfg.scheduler, pipeline)

    # Prepare generator
    generator = _seed_generator(cfg.seed)

    # Ensure output parent directory exists
    cfg.out.parent.mkdir(parents=True, exist_ok=True)

    # Build inference kwargs
    infer_kwargs = dict(
        prompt=cfg.prompt,
        height=cfg.height,
        width=cfg.width,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
    )
    if cfg.negative_prompt:
        infer_kwargs["negative_prompt"] = cfg.negative_prompt

    logger.info("Starting generation: prompt=%s, size=%dx%d, steps=%d, guidance=%.2f",
                (cfg.prompt if len(cfg.prompt) <= 120 else cfg.prompt[:120] + "..."),
                cfg.width, cfg.height, cfg.num_inference_steps, cfg.guidance_scale)

    start = time.time()
    # Use appropriate autocast for speed/memory if on cuda
    try:
        if cfg.device == "cuda" and torch.cuda.is_available():
            # Note: depending on torch/diffusers version, autocast context may vary
            with torch.autocast(device_type="cuda"):
                image = pipeline(**infer_kwargs).images[0]
        else:
            # CPU generation
            image = pipeline(**infer_kwargs).images[0]
    except Exception as e:
        logger.exception("Failed during model inference")
        raise RuntimeError("Model inference failed: %s" % e) from e

    duration = time.time() - start
    logger.info("Generation completed in %.2fs", duration)

    # Validate image object
    if not isinstance(image, Image.Image):
        # If pipeline returns numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        else:
            raise RuntimeError("Unexpected image type returned from pipeline: %s" % type(image))

    # Save image
    try:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        image.save(cfg.out)
        logger.info("Saved image to %s", cfg.out)
    except Exception as e:
        logger.exception("Failed to save image to %s", cfg.out)
        raise

    return cfg.out


def main(argv: Optional[list[str]] = None) -> int:
    try:
        cfg = parse_args(argv)
        logger.debug("Parsed config: %s", cfg)
        out_path = generate_image(cfg)
        logger.info("Completed successfully. Output: %s", out_path)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
