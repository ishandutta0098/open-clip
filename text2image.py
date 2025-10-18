#!/usr/bin/env python3
"""
text2image.py

A production-ready CLI and library entrypoint for generating images from text prompts
using Hugging Face Diffusers (Stable Diffusion).

Features:
- CLI-driven with sensible defaults and comprehensive validation
- Support for GPU (CUDA) with mixed precision when available
- Model and cache configuration, with HF token read from environment
- Robust error handling and logging
- Batch generation support and deterministic seeds
- Performance optimizations (slicing, attention slicing)

Usage (example):
  export HUGGINGFACE_HUB_TOKEN="<your_token>"
  python text2image.py --prompt "A scenic landscape, sunrise" --num_images 2 --out_dir ./outputs

"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch

# Diffusers and transformers imports
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except Exception as e:
    raise RuntimeError(
        "Missing diffusers/transformers packages. Install dependencies from requirements.txt"
    ) from e

# ----------------------------------------------------------------------------
# Configuration and constants
# ----------------------------------------------------------------------------

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
HF_TOKEN_ENV_VARS = ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------

logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
)
logger.addHandler(handler)

# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Configuration for image generation.

    Attributes:
        prompt: Text prompt for the model.
        model_id: HF Diffusers model identifier.
        out_dir: Directory to save generated images.
        num_images: Number of images to generate.
        guidance_scale: Classifier-free guidance scale.
        num_inference_steps: Number of denoising steps.
        seed: Optional seed for reproducibility.
        height: Output height (must be multiple of 8).
        width: Output width (must be multiple of 8).
        device: Torch device identifier.
        torch_dtype: Torch dtype to use (auto-selected if None).
        cache_dir: Optional local cache directory for huggingface models.
    """

    prompt: str
    model_id: str = DEFAULT_MODEL_ID
    out_dir: str = "./outputs"
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    height: int = 512
    width: int = 512
    device: str = "cpu"
    torch_dtype: Optional[torch.dtype] = None
    cache_dir: Optional[str] = None


# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def _get_hf_token() -> Optional[str]:
    """Return Hugging Face token from environment or None."""
    for env_name in HF_TOKEN_ENV_VARS:
        val = os.environ.get(env_name)
        if val:
            logger.debug("Found HF token in %s", env_name)
            return val
    return None


def _validate_config(cfg: GenerationConfig) -> None:
    """Validate GenerationConfig values and raise ValueError on invalid input."""
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    if not (1 <= cfg.num_images <= 50):
        raise ValueError("num_images must be between 1 and 50.")
    if not (1 <= cfg.num_inference_steps <= 500):
        raise ValueError("num_inference_steps must be between 1 and 500.")
    if not (0.0 <= cfg.guidance_scale <= 30.0):
        raise ValueError("guidance_scale must be between 0 and 30.")
    if cfg.height % 8 != 0 or cfg.width % 8 != 0:
        raise ValueError("width and height must be divisible by 8 for Stable Diffusion.")
    if cfg.seed is not None and (cfg.seed < 0 or cfg.seed > 2 ** 31 - 1):
        raise ValueError("seed must be a 32-bit non-negative integer.")


def _ensure_out_dir(path: str) -> pathlib.Path:
    """Create output directory if it doesn't exist and return the Path."""
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _make_output_filename(prefix: str, seed: Optional[int], idx: int) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    seedpart = f"_s{seed}" if seed is not None else ""
    return f"{prefix}_{ts}{seedpart}_{idx:03d}.png"


# ----------------------------------------------------------------------------
# Model loading and generation
# ----------------------------------------------------------------------------

def load_pipeline(
    model_id: str,
    device: str = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> StableDiffusionPipeline:
    """Load and return a StableDiffusionPipeline with recommended optimizations.

    Args:
        model_id: Model repo id.
        device: 'cuda' or 'cpu'.
        torch_dtype: Torch dtype to load model with (e.g., torch.float16).
        hf_token: Hugging Face token for private models.
        cache_dir: Optional cache directory path.

    Returns:
        Configured StableDiffusionPipeline on the requested device.
    """
    logger.info("Loading model %s on device=%s dtype=%s", model_id, device, torch_dtype)

    # Select scheduler: DPMSolverMultistepScheduler is a performant choice
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)  # type: ignore[arg-type]

    # from_pretrained will handle model and VAE loading
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            safety_checker=None,  # disable automatic safety checker; downstream systems should handle content moderation
            torch_dtype=torch_dtype,
            revision=None,
            use_auth_token=hf_token,
            cache_dir=cache_dir,
        )
    except TypeError:
        # older/newer diffusers variants have different signatures
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch_dtype,
            use_auth_token=hf_token,
            cache_dir=cache_dir,
        )

    # Device placement
    pipe = pipe.to(device)

    # Performance optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        logger.debug("xformers not available or failed to enable (optional optimization).")
    try:
        pipe.enable_attention_slicing()
    except Exception:
        logger.debug("enable_attention_slicing failed (optional optimization).")

    logger.info("Model loaded and optimized.")
    return pipe


def _prepare_generator(seed: Optional[int], device: str) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(seed)
    return gen


def generate_images(
    cfg: GenerationConfig,
) -> List[Tuple[Image.Image, Dict]]:
    """Generate images per the config and return list of (PIL.Image, metadata) tuples.

    The metadata includes seed, prompt, model_id and parameters used.
    """
    _validate_config(cfg)

    hf_token = _get_hf_token()
    if hf_token is None:
        logger.warning(
            "No Hugging Face token found in environment. Public models may still be accessible, but private models will fail."
        )

    # Choose dtype automatically: use float16 on CUDA for performance if not explicitly set
    torch_dtype = cfg.torch_dtype
    if torch_dtype is None:
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    pipe = load_pipeline(
        model_id=cfg.model_id,
        device=cfg.device,
        torch_dtype=torch_dtype,
        hf_token=hf_token,
        cache_dir=cfg.cache_dir,
    )

    # Set up generator
    generator = _prepare_generator(cfg.seed, cfg.device)

    logger.info(
        "Generating %d image(s) with prompt=%r, steps=%d, guidance=%.2f",
        cfg.num_images,
        cfg.prompt,
        cfg.num_inference_steps,
        cfg.guidance_scale,
    )

    results: List[Tuple[Image.Image, Dict]] = []

    for i in range(cfg.num_images):
        seed_for_image = cfg.seed if cfg.seed is None else (cfg.seed + i)
        image_gen = _prepare_generator(seed_for_image, cfg.device)

        try:
            # Autocast for fp16 on CUDA
            if cfg.device.startswith("cuda") and torch_dtype == torch.float16:
                with torch.autocast(device_type="cuda"):
                    out = pipe(
                        prompt=cfg.prompt,
                        height=cfg.height,
                        width=cfg.width,
                        num_inference_steps=cfg.num_inference_steps,
                        guidance_scale=cfg.guidance_scale,
                        generator=image_gen,
                    )
            else:
                out = pipe(
                    prompt=cfg.prompt,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    generator=image_gen,
                )
        except Exception as e:
            logger.exception("Generation failed on iteration %d: %s", i, str(e))
            raise

        if hasattr(out, "images"):
            img = out.images[0]
        elif isinstance(out, dict) and "images" in out:
            img = out["images"][0]
        else:
            raise RuntimeError("Unexpected pipeline output format")

        meta = {
            "prompt": cfg.prompt,
            "model_id": cfg.model_id,
            "seed": seed_for_image,
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "height": cfg.height,
            "width": cfg.width,
        }
        results.append((img, meta))

    return results


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate images from text using Stable Diffusion")

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate images from.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face Diffusers model id.")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save generated images.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate (1-50).")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=None, help="Optional integer seed for deterministic outputs.")
    parser.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8).")
    parser.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8).")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Torch device to run on (cpu or cuda).")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional HF cache directory for downloaded models.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    dtype = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dtype = torch.float16

    cfg = GenerationConfig(
        prompt=args.prompt,
        model_id=args.model_id,
        out_dir=args.out_dir,
        num_images=args.num_images,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        height=args.height,
        width=args.width,
        device=args.device,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
    )

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)

    try:
        out_dir_path = _ensure_out_dir(cfg.out_dir)
    except Exception as e:
        logger.exception("Failed to prepare output directory: %s", e)
        return 2

    try:
        images = generate_images(cfg)
    except Exception as e:
        logger.exception("Image generation failed: %s", e)
        return 3

    saved_files: List[str] = []
    for i, (img, meta) in enumerate(images):
        prefix_hash = hashlib.sha1(meta["prompt"].encode("utf-8")).hexdigest()[:8]
        fname = _make_output_filename(prefix=f"sd_{prefix_hash}", seed=meta.get("seed"), idx=i)
        out_path = out_dir_path / fname
        try:
            img.save(out_path, format="PNG")
            saved_files.append(str(out_path))
            logger.info("Saved image to %s", out_path)
        except Exception:
            logger.exception("Failed to save image to %s", out_path)

    logger.info("Completed generation. %d files saved.", len(saved_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
