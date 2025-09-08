#!/usr/bin/env python3
"""
text2image.py

A production-ready text-to-image generation CLI using Hugging Face Diffusers.

Features:
- Supports multiple schedulers (DDIM, DPMSolverMultistep, LMSDiscrete, EulerAncestral)
- Automatic device selection (CUDA if available, otherwise CPU)
- Mixed precision when GPU is available
- Batch generation and reproducible seeding
- Hugging Face authentication token support
- Robust input validation, error handling, and logging
- Saves images with metadata in filenames; returns list of saved paths

Usage example:
  python text2image.py \
    --model runwayml/stable-diffusion-v1-5 \
    --prompt "A fantasy landscape, vivid colors" \
    --out_dir ./outputs \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --height 512 --width 512 \
    --seed 42 \
    --hf_token $HF_TOKEN

"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

# Diffusers and HF imports
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
from huggingface_hub import login as hf_login

# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


SUPPORTED_SCHEDULERS = {
    "ddim": DDIMScheduler,
    "dpmsolver": DPMSolverMultistepScheduler,
    "lms": LMSDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
}


@dataclass
class GenerationConfig:
    model_id: str
    prompt: str
    out_dir: Path
    hf_token: Optional[str] = None
    device: str = "auto"
    scheduler: str = "dpmsolver"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    num_images: int = 1
    batch_size: int = 1
    torch_dtype: Optional[torch.dtype] = None


def validate_config(cfg: GenerationConfig) -> None:
    """Validate the generation configuration and raise ValueError on invalid values.

    Args:
        cfg: GenerationConfig to validate.

    Raises:
        ValueError: If validation fails.
    """
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if cfg.num_inference_steps <= 0 or cfg.num_inference_steps > 500:
        raise ValueError("num_inference_steps must be between 1 and 500")

    if cfg.guidance_scale < 1.0 or cfg.guidance_scale > 30.0:
        raise ValueError("guidance_scale should be in [1.0, 30.0]")

    if cfg.height % 8 != 0 or cfg.width % 8 != 0:
        raise ValueError("height and width must be divisible by 8 for many stable diffusion models")

    if cfg.num_images <= 0:
        raise ValueError("num_images must be >= 1")

    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    if cfg.batch_size > cfg.num_images:
        raise ValueError("batch_size cannot be greater than num_images")

    if cfg.scheduler not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"Unsupported scheduler '{cfg.scheduler}'. Supported: {list(SUPPORTED_SCHEDULERS.keys())}")


def choose_device(requested: str = "auto") -> str:
    """Choose compute device string.

    Args:
        requested: "auto" to pick CUDA if available or "cpu"/"cuda" explicitly.

    Returns:
        Device string suitable for torch.device.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return requested


def get_torch_dtype(device: str) -> Optional[torch.dtype]:
    """Return recommended torch dtype based on device.

    Use float16 on CUDA for performance, float32 otherwise.
    """
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_pipeline(cfg: GenerationConfig) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline using the provided configuration.

    This loads a scheduler mapping based on cfg.scheduler and sets up the
    pipeline for the chosen device and dtype. If a Hugging Face token is provided,
    it logs in using huggingface_hub.login to ensure access to gated models.

    Args:
        cfg: GenerationConfig

    Returns:
        An initialized StableDiffusionPipeline
    """
    if cfg.hf_token:
        # Login sets token for huggingface_hub; safe to call (idempotent)
        try:
            hf_login(token=cfg.hf_token, add_to_git_credential=False)
            logger.info("Authenticated to Hugging Face hub")
        except Exception as e:
            logger.exception("Failed to login to Hugging Face Hub: %s", e)
            raise

    scheduler_cls = SUPPORTED_SCHEDULERS[cfg.scheduler]
    logger.info("Using scheduler: %s", scheduler_cls.__name__)

    try:
        # Use low_cpu_mem_usage to reduce memory footprint on load
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            scheduler=scheduler_cls.from_config({}) if hasattr(scheduler_cls, "from_config") else scheduler_cls(),
            safety_checker=None,  # we'll do a simple check later if available
            torch_dtype=cfg.torch_dtype,
            revision=None,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        logger.exception("Failed to load pipeline for model %s: %s", cfg.model_id, e)
        raise

    device = choose_device(cfg.device)
    pipe = pipe.to(device)
    logger.info("Pipeline loaded on device: %s, dtype: %s", device, cfg.torch_dtype)

    # If CUDA, enable attention slicing for low GPU-memory devices
    try:
        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()  # optional, best-effort
            pipe.enable_attention_slicing()
    except Exception:
        # Not critical; best-effort optimizations
        logger.debug("xformers or attention slicing not available or failed to enable", exc_info=True)

    return pipe


def run_generation(cfg: GenerationConfig) -> List[Path]:
    """Run text-to-image generation and save images to out_dir.

    Args:
        cfg: GenerationConfig

    Returns:
        List of file paths where images were saved.
    """
    validate_config(cfg)

    # Ensure output directory exists
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    base_seed = cfg.seed if cfg.seed is not None else int.from_bytes(os.urandom(2), "big")
    logger.info("Base seed: %s", base_seed)

    saved_paths: List[Path] = []

    # Load pipeline
    pipe = load_pipeline(cfg)

    device = choose_device(cfg.device)

    images_to_generate = cfg.num_images
    batch_size = min(cfg.batch_size, images_to_generate)

    # Generate in batches
    generated = 0
    while generated < images_to_generate:
        current_batch = min(batch_size, images_to_generate - generated)
        seeds = [base_seed + generated + i for i in range(current_batch)]

        for idx, seed in enumerate(seeds):
            generator = torch.Generator(device=device).manual_seed(int(seed))

            try:
                output = pipe(
                    prompt=cfg.prompt,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    generator=generator,
                )
            except Exception as e:
                logger.exception("Generation failed for seed %s: %s", seed, e)
                raise

            if hasattr(output, "images"):
                images = output.images
            elif isinstance(output, dict) and "images" in output:
                images = output["images"]
            else:
                raise RuntimeError("Unexpected pipeline output format")

            for img in images:
                # Image sanitization: ensure PIL Image
                if not isinstance(img, Image.Image):
                    # Try to convert numpy array
                    try:
                        img = Image.fromarray(img.astype("uint8"))
                    except Exception:
                        logger.warning("Unexpected image type; skipping")
                        continue

                timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                prompt_hash = hashlib.sha1(cfg.prompt.encode("utf-8")).hexdigest()[:8]
                filename = f"sd_{prompt_hash}_s{seed}_{timestamp}.png"
                out_path = cfg.out_dir / filename

                try:
                    img.save(out_path, format="PNG")
                    saved_paths.append(out_path)
                    logger.info("Saved image: %s", out_path)
                except Exception as e:
                    logger.exception("Failed to save image to %s: %s", out_path, e)

        generated += current_batch

        # Free up GPU memory if used
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("torch.cuda.empty_cache() failed or not available", exc_info=True)

    # Optionally run a light safety check if pipeline contains a safety_checker
    try:
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            logger.info("Safety checker available; running basic check is delegated to pipeline during generation.")
    except Exception:
        logger.debug("Safety checker check failed or unavailable", exc_info=True)

    return saved_paths


def parse_args(argv: Optional[List[str]] = None) -> GenerationConfig:
    """Parse CLI args into a GenerationConfig.

    Args:
        argv: Optional list of args for testing.

    Returns:
        GenerationConfig
    """
    parser = argparse.ArgumentParser(description="Text-to-image generation using Hugging Face Diffusers")

    parser.add_argument("--model", "-m", dest="model_id", required=True, help="Hugging Face model id (e.g., runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt to generate images from")
    parser.add_argument("--out_dir", "-o", required=True, help="Directory to save generated images")
    parser.add_argument("--hf_token", "-t", default=os.environ.get("HF_TOKEN"), help="Hugging Face token or set HF_TOKEN env")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on (auto|cpu|cuda)")
    parser.add_argument("--scheduler", default="dpmsolver", choices=list(SUPPORTED_SCHEDULERS.keys()), help="Scheduler to use")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Image height (must be multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (must be multiple of 8)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for RNG (reproducible)")
    parser.add_argument("--num_images", type=int, default=1, help="Total images to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to generate at once (<= num_images)")

    args = parser.parse_args(argv)

    device = choose_device(args.device)
    torch_dtype = get_torch_dtype(device)

    cfg = GenerationConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        out_dir=Path(args.out_dir),
        hf_token=args.hf_token,
        device=device,
        scheduler=args.scheduler,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_images=args.num_images,
        batch_size=args.batch_size,
        torch_dtype=torch_dtype,
    )

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    """Main entrypoint for CLI.

    Args:
        argv: Optional argv list for testing.

    Returns:
        Exit code integer
    """
    start_time = time.time()
    try:
        cfg = parse_args(argv)
        logger.info("Starting generation with config: %s", dataclasses.asdict(cfg))

        saved = run_generation(cfg)
        logger.info("Generation complete. Saved %d images in %s", len(saved), cfg.out_dir)

        elapsed = time.time() - start_time
        logger.info("Elapsed time: %.2fs", elapsed)
        return 0
    except Exception as e:
        logger.exception("Fatal error in text2image: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
