#!/usr/bin/env python3
"""
text2image.py

Create images from text prompts using HuggingFace Diffusers (Stable Diffusion).

Features:
- CLI with typed arguments and environment variable support for HF token
- GPU/CPU automatic device selection
- Performance optimizations (fp16 on GPU, attention slicing, sequential CPU offload)
- Deterministic seed support
- Input validation and safe output path handling
- Progress callback and logging

Usage (examples):
  python text2image.py --prompt "A cozy cottage in a snowy forest" --out ./out.png --num_inference_steps 30 --guidance_scale 7.5
  HF_TOKEN=your_token python text2image.py --model runwayml/stable-diffusion-v1-5 --prompt "..."

Note: You need a valid Hugging Face token for gated models (e.g., stable-diffusion). Set HF_TOKEN env var or pass --hf_token.

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
from typing import Any, Callable, Dict, Optional

try:
    import torch
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline
except Exception as e:  # pragma: no cover - import/runtime environment dependent
    raise ImportError(
        "Required packages are not installed. Please run `pip install -r requirements.txt`.\n"
        f"Import error: {e}"
    )

# Configure module logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)


@dataclass
class GenerationConfig:
    model: str
    prompt: str
    negative_prompt: Optional[str]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    seed: Optional[int]
    output: Path
    hf_token: Optional[str]
    device: str
    dtype: Any
    safe_output_format: str = "png"


def _validate_args(args: argparse.Namespace) -> GenerationConfig:
    """Validate and normalize CLI args into GenerationConfig.

    Raises:
        ValueError: if invalid arguments are provided.
    """
    if not args.prompt or not args.prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    # Ensure height/width are multiples of 8 (Stable Diffusion requirement)
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("Height and width must be multiples of 8.")

    if args.num_inference_steps <= 0 or args.num_inference_steps > 200:
        raise ValueError("num_inference_steps must be between 1 and 200.")

    if not (0.0 <= args.guidance_scale <= 50.0):
        raise ValueError("guidance_scale must be between 0.0 and 50.0.")

    out_path = Path(args.out).expanduser().resolve()
    out_dir = out_path.parent
    if not out_dir.exists():
        logger.debug("Creating output directory: %s", out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # enforce extension
    ext = out_path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        # replace with safe format
        out_path = out_path.with_suffix(".png")

    # Device selection
    device = "cpu"
    dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda"
        # Use fp16 for better VRAM usage/performance
        dtype = torch.float16

    cfg = GenerationConfig(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=(args.negative_prompt or None),
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=(None if args.seed is None else int(args.seed)),
        output=out_path,
        hf_token=(args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        device=device,
        dtype=dtype,
    )
    return cfg


def _seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_save_image(img, path: Path) -> None:
    """Save PIL image safely by writing to temp and atomically moving.

    Ensures partially written files are not left if process crashes.
    """
    from PIL import Image

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp_path)
    tmp_path.replace(path)


def _progress_callback(step: int, timestep: int, latents: Any) -> None:
    # Only log step-level progress to avoid spamming logs with large arrays
    logger.debug("Generation progress - step: %d, timestep: %d", step, timestep)


def load_pipeline(cfg: GenerationConfig) -> DiffusionPipeline:
    """Load a HuggingFace Diffusers pipeline with performance-oriented settings.

    Returns:
        A configured DiffusionPipeline ready for inference.
    """
    # Use a deterministic scheduler known for good quality/speed tradeoffs
    scheduler = DPMSolverMultistepScheduler

    logger.info("Loading model: %s", cfg.model)
    try:
        # Use StableDiffusionPipeline alias where available
        pipeline = DiffusionPipeline.from_pretrained(
            cfg.model,
            torch_dtype=cfg.dtype,
            use_auth_token=cfg.hf_token,
        )
    except TypeError:
        # Older/newer API differences: try alternative keyword
        pipeline = DiffusionPipeline.from_pretrained(
            cfg.model,
            torch_dtype=cfg.dtype,
            use_auth_token=cfg.hf_token,
        )

    # Replace scheduler with DPMSolverMultistep for faster sampling and stable results
    try:
        pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    except Exception:
        logger.debug("Could not replace scheduler, continuing with default.")

    # Performance heuristics
    try:
        if cfg.device == "cuda":
            pipeline.to(device=cfg.device)
            # Enable fp16 attention & memory optimizations
            pipeline.enable_attention_slicing()
            # Reduce memory by enabling sequential CPU offload if available
            if hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.to(device=cfg.device)
    except Exception as e:
        logger.warning("Failed to apply some performance optimizations: %s", e)

    # Turn on safety checker if present (best-effort; may be None)
    if hasattr(pipeline, "safety_checker") and pipeline.safety_checker is None:
        logger.debug("No safety checker available on model; using raw outputs.")

    return pipeline


def generate_image(cfg: GenerationConfig) -> Path:
    """Run the diffusion pipeline to generate an image using provided configuration.

    Returns:
        Path to the saved image.
    """
    if cfg.seed is not None:
        _seed_everything(cfg.seed)

    pipeline = load_pipeline(cfg)

    # Prepare generator for reproducibility
    generator = None
    if cfg.seed is not None:
        device_for_gen = 0 if cfg.device == "cuda" else -1
        generator = torch.Generator(device=(cfg.device if cfg.device == "cuda" else "cpu")).manual_seed(cfg.seed)

    logger.info("Generating image (device=%s, dtype=%s)", cfg.device, cfg.dtype)

    # Safety: limit prompt length moderately
    if len(cfg.prompt) > 2000:
        logger.warning("Prompt is very long (%d chars). Truncating to 2000 chars.", len(cfg.prompt))
        prompt = cfg.prompt[:2000]
    else:
        prompt = cfg.prompt

    try:
        out = pipeline(
            prompt=prompt,
            negative_prompt=cfg.negative_prompt,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=(generator if cfg.seed is not None else None),
            callback=_progress_callback,
            callback_steps=1,
        )
    except RuntimeError as e:
        # Try to give actionable advice for CUDA OOM
        if "out of memory" in str(e).lower():
            logger.error("CUDA out of memory during generation: %s", e)
            logger.info("Consider reducing height/width, batch size, or using --num_inference_steps smaller value.")
        raise

    images = out.images if hasattr(out, "images") else out[0]
    if isinstance(images, list):
        image = images[0]
    else:
        image = images

    # Save to output path atomically
    _safe_save_image(image, cfg.output)
    logger.info("Saved image to %s", cfg.output)
    return cfg.output


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from text using HuggingFace Diffusers")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="HF model id or local path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate from")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to avoid features")
    parser.add_argument("--height", type=int, default=512, help="Image height (multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (multiple of 8)")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale / guidance")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--out", type=str, default="./output.png", help="Output file path (png/jpg/webp)")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        cfg = _validate_args(args)
    except ValueError as e:
        logger.error("Invalid arguments: %s", e)
        return 2

    try:
        start = time.time()
        out_path = generate_image(cfg)
        end = time.time()
        logger.info("Done in %.2f sec. Output: %s", end - start, out_path)
        return 0
    except Exception as e:
        logger.exception("Failed to generate image: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
