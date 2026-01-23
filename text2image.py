#!/usr/bin/env python3
"""
text2image.py

Command-line utility to generate images from text prompts using Hugging Face Diffusers.

Features:
- Supports model loading with optional Hugging Face token (from env var or CLI)
- Device selection (CUDA/CPU) with mixed precision where available
- Configurable guidance scale, steps, seed, batch size
- Safety: input validation, token requirement for gated models, minimal exposure of tokens
- Performance: attention slicing, xformers (optional) enabling
- Robust error handling and logging

Usage example:
  HF_TOKEN=your_token python text2image.py \
    --model "runwayml/stable-diffusion-v1-5" \
    --prompt "A fantasy landscape, sunset, ultra detailed" \
    --outdir ./outputs --num_inference_steps 30 --guidance_scale 7.5

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch

try:
    from diffusers import DiffusionPipeline, StableDiffusionPipeline
except Exception:
    # Generic import fallback for different diffusers versions
    from diffusers import DiffusionPipeline  # type: ignore


# ----------------------------- Configuration ---------------------------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"


# ----------------------------- Data Classes ---------------------------------
@dataclass
class GenerationConfig:
    model: str
    prompt: str
    negative_prompt: Optional[str]
    outdir: Path
    seed: Optional[int]
    num_inference_steps: int
    guidance_scale: float
    width: int
    height: int
    batch_size: int
    device: str
    dtype: str
    hf_token: Optional[str]


# ------------------------------- Utilities ----------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger.

    Args:
        level: logging level (default INFO)
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)


def validate_args(args: argparse.Namespace) -> GenerationConfig:
    """Validate and normalize CLI args into GenerationConfig.

    Raises:
        SystemExit: if invalid arguments are provided
    """
    outdir = Path(args.outdir).expanduser().resolve()
    if not outdir.exists():
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error("Failed to create outdir=%s: %s", outdir, e)
            raise SystemExit(1)

    if not args.prompt or not args.prompt.strip():
        logging.error("Prompt must be a non-empty string")
        raise SystemExit(2)

    if args.width % 8 != 0 or args.height % 8 != 0:
        logging.warning("Width and height should be divisible by 8 for many models; proceeding anyway")

    if args.batch_size < 1:
        logging.error("batch_size must be >= 1")
        raise SystemExit(2)

    # Determine dtype
    dtype = args.dtype.lower()
    if dtype not in ("fp16", "fp32"):
        logging.error("Invalid dtype: %s. Choose fp16 or fp32", args.dtype)
        raise SystemExit(2)

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return GenerationConfig(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        outdir=outdir,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        hf_token=hf_token,
    )


# ------------------------------- Core Logic ---------------------------------

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map dtype string to torch dtype."""
    return torch.float16 if dtype_str == "fp16" else torch.float32


def load_pipeline(model_id: str, hf_token: Optional[str], device: str, dtype: str) -> DiffusionPipeline:
    """Load a diffusers pipeline safely with configured dtype and device.

    This function attempts to enable memory optimizations and safe defaults.

    Args:
        model_id: Hugging Face model repo id
        hf_token: optional token for gated models
        device: 'cuda' or 'cpu'
        dtype: 'fp16' or 'fp32'

    Returns:
        Instantiated and device-moved DiffusionPipeline

    Raises:
        RuntimeError: on load failures
    """
    torch_dtype = get_torch_dtype(dtype)

    logging.info("Loading model %s with dtype=%s on device=%s", model_id, torch_dtype, device)

    # Use safe defaults for from_pretrained; many models require use_auth_token
    kwargs = {"torch_dtype": torch_dtype}
    if hf_token:
        kwargs["use_auth_token"] = True  # pass-through; HF CLI will pick token from env

    try:
        # Attempt to load a StableDiffusionPipeline first (common)
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    except Exception:
        # Fallback to generic DiffusionPipeline
        pipeline = DiffusionPipeline.from_pretrained(model_id, **kwargs)

    # Move to device
    pipeline = pipeline.to(device)

    # Performance: enable attention slicing to reduce peak memory
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        logging.debug("enable_attention_slicing not available for this pipeline")

    # Try enable xformers memory efficient attention if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logging.info("xFormers memory efficient attention enabled")
    except Exception:
        logging.debug("xFormers not available or failed to enable")

    return pipeline


def generate_images(
    pipeline: DiffusionPipeline,
    cfg: GenerationConfig,
) -> List[Path]:
    """Generate images from text prompts using a loaded pipeline.

    Args:
        pipeline: loaded Diffusers pipeline
        cfg: generation configuration

    Returns:
        List of image file paths that were written
    """
    rng = None
    if cfg.seed is not None:
        generator = torch.manual_seed(cfg.seed)
        # Create a torch Generator for GPU if needed
        device = torch.device(cfg.device)
        gen = torch.Generator(device)
        gen.manual_seed(cfg.seed)
    else:
        gen = None

    saved_paths: List[Path] = []

    total = cfg.batch_size
    prompts = [cfg.prompt] * total
    negative_prompts = [cfg.negative_prompt] * total if cfg.negative_prompt else None

    logging.info(
        "Generating %d image(s) with steps=%d guidance=%.2f",
        total,
        cfg.num_inference_steps,
        cfg.guidance_scale,
    )

    # Build pipeline kwargs in a version-agnostic way
    pipeline_kwargs = dict(
        prompt=prompts,
        height=cfg.height,
        width=cfg.width,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=gen,
    )
    if negative_prompts is not None:
        pipeline_kwargs["negative_prompt"] = negative_prompts

    # Use autocast for fp16 if on CUDA
    autocast_ctx = torch.cuda.amp.autocast if (cfg.device.startswith("cuda") and cfg.dtype == "fp16") else torch.no_grad

    start = time.time()
    try:
        with autocast_ctx():
            result = pipeline(**pipeline_kwargs)
    except Exception as e:
        logging.exception("Pipeline generation failed: %s", e)
        raise

    images = None
    if isinstance(result, dict) and "images" in result:
        images = result["images"]
    else:
        # Some pipeline versions return a returned object with .images
        images = getattr(result, "images", None)

    if images is None:
        logging.error("Could not retrieve images from pipeline result")
        raise RuntimeError("Pipeline returned no images")

    # Save images
    for i, img in enumerate(images):
        if not isinstance(img, Image.Image):
            # Convert numpy arrays to PIL
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
            else:
                logging.warning("Unexpected image type: %s", type(img))

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        fname = f"img_{timestamp}_{i:03d}.png"
        outpath = cfg.outdir / fname
        try:
            img.save(outpath, format="PNG")
            saved_paths.append(outpath)
            logging.info("Saved image: %s", outpath)
        except Exception as e:
            logging.exception("Failed to save image to %s: %s", outpath, e)

    elapsed = time.time() - start
    logging.info("Generation completed in %.2fs", elapsed)

    # Optionally write manifest/metadata
    manifest = cfg.outdir / f"metadata_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    metadata = {
        "model": cfg.model,
        "prompt": cfg.prompt,
        "negative_prompt": cfg.negative_prompt,
        "num_images": len(saved_paths),
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "width": cfg.width,
        "height": cfg.height,
        "seed": cfg.seed,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "files": [str(p.name) for p in saved_paths],
    }
    try:
        manifest.write_text(json.dumps(metadata, indent=2))
        logging.info("Wrote metadata manifest: %s", manifest)
    except Exception:
        logging.debug("Failed to write metadata manifest")

    return saved_paths


# --------------------------------- CLI -------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to image generation using Hugging Face Diffusers")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF repo id for the model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the image generation")
    parser.add_argument("--negative-prompt", dest="negative_prompt", type=str, default=None, help="Negative prompt to steer away from")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1, help="How many images to generate in one run")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Compute dtype")
    parser.add_argument("--hf-token", dest="hf_token", type=str, default=None, help="Hugging Face token (optional, or set HF_TOKEN env var)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    cfg = validate_args(args)

    # Basic warning about CPU-mode
    if cfg.device == "cpu":
        logging.warning("Using CPU for generation. This will be slow and may not fit some models in memory.")

    try:
        pipeline = load_pipeline(cfg.model, cfg.hf_token, cfg.device, cfg.dtype)
    except Exception as e:
        logging.exception("Failed to load pipeline: %s", e)
        return 3

    try:
        paths = generate_images(pipeline, cfg)
        logging.info("Generated %d images", len(paths))
        return 0
    except Exception as e:
        logging.exception("Image generation failed: %s", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
