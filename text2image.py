#!/usr/bin/env python3
"""
text2image.py

A production-ready CLI and library entrypoint for generating images from text prompts
using Hugging Face Diffusers (Stable Diffusion). Designed with security, input
validation, logging, and simple caching for performance.

Usage (CLI):
  python text2image.py --prompt "A cozy cabin in a snowy forest" --out_path ./outputs/cabin.png

Library usage example:
  from text2image import generate_image
  out = generate_image("A cat wearing a suit", out_path="./cat.png")

Requirements: see requirements.txt
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image

# Diffusers imports placed inside functions where possible to avoid import cost for non-generation flows

# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
)
logger.addHandler(handler)

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
VALID_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]")

# Simple in-memory cache for loaded pipelines to avoid repeated loads
_PIPELINE_CACHE = {}


class Text2ImageError(Exception):
    """Generic error for the text2image module."""


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for image generation.

    Attributes:
        prompt: The text prompt to generate from.
        out_path: File path to save generated image.
        model_id: Hugging Face diffusers model id.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed for reproducibility. If None, randomized.
        width: Image width (multiple of 8).
        height: Image height (multiple of 8).
        device: Device identifier (e.g., 'cpu', 'cuda', 'mps') or None to auto-select.
        hf_token: Hugging Face token or None to read from env.
    """

    prompt: str
    out_path: str
    model_id: str = DEFAULT_MODEL
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    width: int = 512
    height: int = 512
    device: Optional[str] = None
    hf_token: Optional[str] = None


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Sanitize arbitrary string for safe filesystem usage.

    Removes characters outside [A-Za-z0-9._-] and truncates to max_length.
    """
    if not name:
        return "output"
    name = VALID_FILENAME_RE.sub("_", name)
    return name[:max_length]


def _select_device(preferred: Optional[str] = None) -> str:
    """Select the best device available.

    Preference order: explicit preferred if available -> CUDA -> MPS (Apple Silicon) -> CPU
    """
    if preferred:
        p = preferred.lower()
        if p == "cuda" and torch.cuda.is_available():
            return "cuda"
        if p == "mps" and getattr(torch, "has_mps", False):
            return "mps"
        if p == "cpu":
            return "cpu"
        logger.warning("Preferred device '%s' not available, falling back to auto-detect.", preferred)

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "has_mps", False):
        return "mps"
    return "cpu"


def _validate_config(cfg: GenerationConfig) -> GenerationConfig:
    """Validate and normalize generation configuration.

    Raises Text2ImageError on invalid values.
    """
    if not cfg.prompt or not isinstance(cfg.prompt, str):
        raise Text2ImageError("Prompt must be a non-empty string")

    if not isinstance(cfg.num_inference_steps, int) or not (1 <= cfg.num_inference_steps <= 500):
        raise Text2ImageError("num_inference_steps must be an int in [1, 500]")

    if not isinstance(cfg.guidance_scale, (int, float)) or not (0.0 <= float(cfg.guidance_scale) <= 30.0):
        raise Text2ImageError("guidance_scale must be a number in [0.0, 30.0]")

    for dim_name, dim in (("width", cfg.width), ("height", cfg.height)):
        if not isinstance(dim, int) or dim <= 0:
            raise Text2ImageError(f"{dim_name} must be a positive integer")
        if dim % 8 != 0:
            raise Text2ImageError(f"{dim_name} must be a multiple of 8 (Stable Diffusion requirement)")
        if dim > 2048:
            raise Text2ImageError(f"{dim_name} is too large; maximum supported is 2048 to avoid OOM")

    device = _select_device(cfg.device)
    return GenerationConfig(
        prompt=cfg.prompt.strip(),
        out_path=cfg.out_path,
        model_id=cfg.model_id,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=float(cfg.guidance_scale),
        seed=cfg.seed,
        width=cfg.width,
        height=cfg.height,
        device=device,
        hf_token=cfg.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
    )


def _get_pipeline_key(model_id: str, device: str, dtype: torch.dtype) -> str:
    return f"{model_id}::{device}::{str(dtype)}"


def load_pipeline(model_id: str, device: str, hf_token: Optional[str], dtype: torch.dtype = torch.float16):
    """Load or reuse a cached Diffusers pipeline.

    Returns a loaded pipeline configured for the given device and dtype. Caches pipelines in-memory
    to avoid repeated downloads and initializations within the same process.
    """
    key = _get_pipeline_key(model_id, device, dtype)
    if key in _PIPELINE_CACHE:
        logger.info("Reusing cached pipeline for %s on %s", model_id, device)
        return _PIPELINE_CACHE[key]

    try:
        logger.info("Loading model %s onto device=%s dtype=%s", model_id, device, dtype)

        # Late import to keep module lightweight until generation is required
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        # Use scheduler override that tends to be stable and faster for many models
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=hf_token,
            revision=None,
            safety_checker=None,  # We intentionally skip safety checker here; callers should be aware
            torch_dtype=dtype,
        )

        # Optionally swap to a high-performance scheduler
        try:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        except Exception:
            # Non-fatal; keep default scheduler if replacement fails
            logger.debug("Could not replace scheduler with DPMSolver; continuing with default")

        pipeline = pipeline.to(device)

        _PIPELINE_CACHE[key] = pipeline
        return pipeline
    except Exception as exc:
        logger.exception("Failed to load pipeline for model %s: %s", model_id, exc)
        raise Text2ImageError(f"Failed to load model {model_id}: {exc}") from exc


def generate_image(
    prompt: str,
    out_path: str,
    model_id: str = DEFAULT_MODEL,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Generate an image from a text prompt.

    Args:
        prompt: Text prompt to generate from.
        out_path: Output path to save the result. Can be a directory (then auto-generated filename)
                  or a file path. The parent directories will be created if necessary.
        model_id: Hugging Face model id (diffusers format).
        num_inference_steps: Denoising steps (higher -> better quality, slower).
        guidance_scale: Classifier-free guidance (higher -> more prompt adherence).
        seed: Optional random seed for reproducibility.
        width: Output width in pixels (multiple of 8).
        height: Output height in pixels (multiple of 8).
        device: 'cuda', 'cpu', or 'mps'. If None, auto-select.
        hf_token: Hugging Face token; if None, read from HUGGINGFACE_HUB_TOKEN env var.

    Returns:
        The absolute path to the saved image.

    Raises:
        Text2ImageError on failures.
    """
    cfg = _validate_config(
        GenerationConfig(
            prompt=prompt,
            out_path=out_path,
            model_id=model_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            device=device,
            hf_token=hf_token,
        )
    )

    out_path_obj = Path(cfg.out_path)
    if out_path_obj.is_dir() or cfg.out_path.endswith(os.sep):
        # Create deterministic filename from prompt + time + seed
        digest = hashlib.sha256((cfg.prompt + (str(cfg.seed) if cfg.seed is not None else "") + str(time.time())).encode()).hexdigest()
        filename = sanitize_filename(cfg.prompt)[:80] + "_" + digest[:8] + ".png"
        out_path_obj = out_path_obj.joinpath(filename)
    else:
        # Ensure parent dir exists
        parent = out_path_obj.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if cfg.device in ("cuda", "mps") else torch.float32

    pipeline = load_pipeline(cfg.model_id, cfg.device, cfg.hf_token, dtype=dtype)

    # Use a generator for reproducibility
    generator = None
    if cfg.seed is not None:
        try:
            gen_device = "cpu" if cfg.device == "cpu" else cfg.device
            generator = torch.Generator(device=gen_device).manual_seed(int(cfg.seed))
        except Exception as exc:
            logger.warning("Could not set generator seed on device '%s': %s", cfg.device, exc)
            generator = torch.Generator().manual_seed(int(cfg.seed))

    # Prepare call args
    call_kwargs = {
        "prompt": cfg.prompt,
        "height": cfg.height,
        "width": cfg.width,
        "num_inference_steps": int(cfg.num_inference_steps),
        "guidance_scale": float(cfg.guidance_scale),
        "generator": generator,
    }

    logger.info("Generating image with model=%s device=%s steps=%s guidance=%.2f seed=%s",
                cfg.model_id, cfg.device, cfg.num_inference_steps, cfg.guidance_scale, str(cfg.seed))

    try:
        # Autocast for float16 speedups on CUDA/MPS
        if cfg.device == "cuda" and dtype == torch.float16:
            with torch.autocast("cuda"):
                result = pipeline(**call_kwargs)
        elif cfg.device == "mps" and dtype == torch.float16:
            # mps autocast path - MPS currently works with float16 in many setups
            with torch.autocast("mps"):
                result = pipeline(**call_kwargs)
        else:
            result = pipeline(**call_kwargs)

        if not hasattr(result, "images") or not result.images:
            raise Text2ImageError("Pipeline returned no images")

        image: Image.Image = result.images[0]

        # Save as PNG to preserve quality and alpha if present
        out_abs = str(out_path_obj.resolve())
        image.save(out_abs, format="PNG")
        logger.info("Image saved to %s", out_abs)
        return out_abs
    except Exception as exc:
        logger.exception("Error during generation: %s", exc)
        raise Text2ImageError(f"Generation failed: {exc}") from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate images from text prompts via Hugging Face Diffusers")
    p.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt to generate from")
    p.add_argument("--out_path", "-o", type=str, default="./outputs/", help="Output file or directory")
    p.add_argument("--model_id", "-m", type=str, default=DEFAULT_MODEL, help="Diffusers model id")
    p.add_argument("--steps", "-s", type=int, default=30, help="Number of denoising steps (1-500)")
    p.add_argument("--scale", type=float, default=7.5, help="Guidance scale (0.0-30.0)")
    p.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")
    p.add_argument("--width", type=int, default=512, help="Image width (multiple of 8)")
    p.add_argument("--height", type=int, default=512, help="Image height (multiple of 8)")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Device to run on (auto if omitted)")
    p.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (or set HUGGINGFACE_HUB_TOKEN env var)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        out = generate_image(
            prompt=args.prompt,
            out_path=args.out_path,
            model_id=args.model_id,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            seed=args.seed,
            width=args.width,
            height=args.height,
            device=args.device,
            hf_token=args.hf_token,
        )
        logger.info("Generation completed. Output: %s", out)
        return 0
    except Text2ImageError as err:
        logger.error("Generation failed: %s", err)
        return 2
    except Exception as uncaught:
        logger.exception("Unexpected error: %s", uncaught)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
