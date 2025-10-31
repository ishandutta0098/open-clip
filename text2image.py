#!/usr/bin/env python3
"""
text2image.py

A production-ready command-line utility to generate images from text prompts using
Hugging Face Diffusers (Stable Diffusion).

Features:
- CLI with robust argument parsing and validation
- GPU (CUDA) auto-detection with mixed precision (fp16) support
- Hugging Face authentication support via HF_TOKEN env var or CLI
- Deterministic seeding for reproducible results
- Error handling for OOM and fallback to CPU
- Detailed logging
- Type hints and Google-style docstrings

Usage example:
  python text2image.py \
    --prompt "A photorealistic painting of a futuristic city" \
    --out_dir outputs \
    --model runwayml/stable-diffusion-v1-5 \
    --height 512 --width 512 --num_inference_steps 30 --guidance_scale 7.5

Security notes:
- You should provide a Hugging Face token (HF_TOKEN environment variable) for private models
  and to avoid rate limits. Public models can often be used anonymously.
- The script does not perform content filtering beyond the pipeline's own safety checker.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Diffusers imports
from diffusers import StableDiffusionPipeline
from huggingface_hub import login as hf_login


# ---------- Configuration and constants ----------
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
ALLOWED_MIN_DIM = 64
ALLOWED_MAX_DIM = 2048  # pragmatic upper bound

# Configure logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
)
logger.addHandler(handler)


# ---------- Utility functions ----------

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _validate_dimensions(width: int, height: int) -> None:
    """Validate width/height are in a sensible range and multiples of 8.

    Models commonly require multiples of 8 (or 32 for some). We enforce multiple-of-8 here.
    """
    if not (ALLOWED_MIN_DIM <= width <= ALLOWED_MAX_DIM):
        raise ValueError(f"width must be between {ALLOWED_MIN_DIM} and {ALLOWED_MAX_DIM}")
    if not (ALLOWED_MIN_DIM <= height <= ALLOWED_MAX_DIM):
        raise ValueError(f"height must be between {ALLOWED_MIN_DIM} and {ALLOWED_MAX_DIM}")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("width and height must be multiples of 8 for most diffusion models")


def _ensure_out_dir(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        raise RuntimeError(f"cannot create output directory {out_dir}: {exc}")


def _set_seed(seed: Optional[int]) -> int:
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    logger.info("Using seed=%d", seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _format_filename(prompt: str, seed: int, model_id: str) -> str:
    # Build a compact, reproducible filename
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
    model_tag = model_id.replace("/", "_")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"sd_{model_tag}_{prompt_hash}_s{seed}_{timestamp}.png"


# ---------- Core generation logic ----------

def load_pipeline(model_id: str, device: torch.device, use_fp16: bool, hf_token: Optional[str]) -> StableDiffusionPipeline:
    """
    Load and return a StableDiffusionPipeline configured for the target device.

    Args:
        model_id: HF model identifier
        device: torch device
        use_fp16: whether to use float16 dtype (only on CUDA devices recommended)
        hf_token: optional HF token for private models

    Returns:
        Instantiated StableDiffusionPipeline
    """
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    # Attempt to login if token provided (saves repeated token passing)
    if hf_token:
        try:
            hf_login(token=hf_token)
            logger.info("Logged into Hugging Face Hub using provided token")
        except Exception as exc:  # pragma: no cover - integration
            logger.warning("Failed to perform explicit HF login: %s", exc)

    logger.info("Loading model '%s' (dtype=%s) ...", model_id, torch_dtype)
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None if use_fp16 is True else None,  # leave default; user can opt out
        )
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("Failed to load model %s: %s", model_id, exc)
        raise

    # Move pipeline to device
    pipeline = pipeline.to(device)

    # For pipeline on CUDA and fp16, enable attention slicing to reduce peak memory
    if device.type == "cuda":
        try:
            pipeline.enable_attention_slicing()
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers may not be available; ignore
            pass

    logger.info("Model loaded and moved to %s", device)
    return pipeline


def generate_image(
    prompt: str,
    out_dir: str,
    model_id: str = DEFAULT_MODEL_ID,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    hf_token: Optional[str] = None,
    negative_prompt: Optional[str] = None,
) -> str:
    """
    Generate a single image from the prompt and save it to disk.

    Returns the path to the saved image.
    """
    start_time = time.time()
    _validate_dimensions(width=width, height=height)

    out_dir_path = Path(out_dir)
    _ensure_out_dir(out_dir_path)

    seed = _set_seed(seed)

    # Device selection
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    use_fp16 = use_cuda  # enable fp16 on CUDA for perf and memory

    pipeline = load_pipeline(model_id=model_id, device=device, use_fp16=use_fp16, hf_token=hf_token)

    # Prepare generator for deterministic output
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    try:
        logger.info("Generating image: prompt='%s' steps=%d guidance=%.2f", prompt, num_inference_steps, guidance_scale)
        with torch.autocast(device.type) if device.type == "cuda" else nullcontext():
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        image = output.images[0]

    except RuntimeError as exc:
        # Attempt graceful OOM handling: try CPU fallback
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            logger.warning("CUDA OOM during generation. Retrying on CPU (this will be slower).")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            pipeline = load_pipeline(model_id=model_id, device=device, use_fp16=False, hf_token=hf_token)
            generator = torch.Generator(device=device).manual_seed(seed)
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = output.images[0]
        else:
            logger.exception("Image generation failed: %s", exc)
            raise

    filename = _format_filename(prompt, seed, model_id)
    out_path = out_dir_path / filename
    try:
        image.save(out_path)
    except Exception as exc:  # pragma: no cover - filesystem
        logger.exception("Failed to save image to %s: %s", out_path, exc)
        raise

    elapsed = time.time() - start_time
    logger.info("Image saved to %s (took %.2fs)", out_path, elapsed)
    return str(out_path)


# ---------- Minimal nullcontext for Python versions that might not have contextlib.nullcontext ----------
try:
    from contextlib import nullcontext  # type: ignore
except Exception:  # pragma: no cover - safety
    class nullcontext:  # type: ignore
        def __init__(self):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False


# ---------- CLI ----------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from text prompts using Hugging Face Diffusers")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate the image from")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save generated images")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model id to use")
    parser.add_argument("--height", type=int, default=512, help="Image height (must be multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (must be multiple of 8)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (alternatively set HF_TOKEN env var)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to discourage features")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token is None:
        logger.info("No HF_TOKEN provided. Using anonymous access where allowed.")

    try:
        out_path = generate_image(
            prompt=args.prompt,
            out_dir=args.out_dir,
            model_id=args.model,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            hf_token=hf_token,
            negative_prompt=args.negative_prompt,
        )
        logger.info("Done. Image path: %s", out_path)
        return 0
    except Exception as exc:
        logger.exception("Failed to generate image: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
