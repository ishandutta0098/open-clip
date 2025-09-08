#!/usr/bin/env python3
"""
text2image.py

A production-ready CLI utility to generate images from text prompts using Hugging Face Diffusers.

Features:
- Automatic device selection (CUDA/CPU)
- Mixed precision when available
- Attention slicing and optional memory optimization
- Input validation and security guardrails
- Configurable generation parameters (seed, guidance, steps, size, batch)
- Saves images with deterministic names and metadata

Usage example:
  export HF_TOKEN=your_hf_token_here
  python3 text2image.py --prompt "A cozy cabin in a snowy forest, photorealistic" --model "runwayml/stable-diffusion-v1-5" --outdir ./outputs --num_images 2 --seed 42

"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

try:
    # Diffusers and HuggingFace libs
    from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
    from huggingface_hub import login as hf_login
except Exception as exc:  # pragma: no cover - run-time dependency management
    raise ImportError(
        "Missing dependencies. Please install requirements from requirements.txt. "
        "See README or use pip install -r requirements.txt"
    ) from exc


# ----- Logging Configuration -----
LOG = logging.getLogger("text2image")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
)
LOG.addHandler(handler)


# ----- Constants & Limits -----
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
VALID_WIDTH = (64, 2048)  # min, max
VALID_HEIGHT = (64, 2048)
MAX_IMAGES_PER_RUN = 8
MIN_STEPS = 1
MAX_STEPS = 200
MAX_GUIDANCE = 30.0
MIN_GUIDANCE = 0.0


@dataclass
class GenConfig:
    model_id: str
    hf_token: Optional[str]
    prompt: str
    negative_prompt: Optional[str]
    outdir: Path
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    num_images: int
    seed: Optional[int]
    scheduler: Optional[str]
    device: str
    dtype: torch.dtype
    enable_xformers: bool


# ----- Utility functions -----

def _ensure_multiple_of_8(value: int) -> int:
    # Many diffusion models expect width/height to be multiples of 8
    return (value + 7) // 8 * 8


def _validate_and_construct_config(args: argparse.Namespace) -> GenConfig:
    if not args.prompt or not args.prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    if args.num_images < 1 or args.num_images > MAX_IMAGES_PER_RUN:
        raise ValueError(f"num_images must be between 1 and {MAX_IMAGES_PER_RUN}")

    if args.num_inference_steps < MIN_STEPS or args.num_inference_steps > MAX_STEPS:
        raise ValueError(f"num_inference_steps must be between {MIN_STEPS} and {MAX_STEPS}")

    if not (MIN_GUIDANCE <= args.guidance_scale <= MAX_GUIDANCE):
        raise ValueError(f"guidance_scale must be between {MIN_GUIDANCE} and {MAX_GUIDANCE}")

    width = max(VALID_WIDTH[0], min(VALID_WIDTH[1], args.width))
    height = max(VALID_HEIGHT[0], min(VALID_HEIGHT[1], args.height))
    width = _ensure_multiple_of_8(width)
    height = _ensure_multiple_of_8(height)

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine device & dtype
    if torch.cuda.is_available() and args.device in ("cuda", "auto"):
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    cfg = GenConfig(
        model_id=args.model,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        prompt=args.prompt.strip(),
        negative_prompt=(args.negative_prompt.strip() if args.negative_prompt else None),
        outdir=outdir,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images=args.num_images,
        seed=args.seed,
        scheduler=(args.scheduler.lower() if args.scheduler else None),
        device=device,
        dtype=dtype,
        enable_xformers=not args.disable_xformers,
    )

    LOG.debug("Constructed GenConfig: %s", cfg)
    return cfg


def _get_scheduler_by_name(name: Optional[str]):
    if not name:
        return None
    name = name.lower()
    if name == "ddim":
        return DDIMScheduler
    if name in ("euler", "euler_discrete"):
        return EulerDiscreteScheduler
    if name in ("dpm_solver", "dpmsolver", "dpmsolver_multistep"):
        return DPMSolverMultistepScheduler
    raise ValueError(f"Unsupported scheduler: {name}")


def _hash_prompt(prompt: str, seed: Optional[int]) -> str:
    m = hashlib.sha256()
    m.update(prompt.encode("utf-8"))
    m.update(str(seed or 0).encode("utf-8"))
    return m.hexdigest()[:16]


def _save_images(images: List[PIL.Image.Image], prompt: str, outdir: Path, meta: Dict) -> List[Path]:
    saved_paths: List[Path] = []
    base_hash = _hash_prompt(prompt, meta.get("seed"))
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    for idx, img in enumerate(images):
        filename = f"img_{timestamp}_{base_hash}_{idx+1}.png"
        path = outdir / filename
        img.save(path)
        saved_paths.append(path)

    # also dump metadata
    meta_path = outdir / f"meta_{timestamp}_{base_hash}.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    return saved_paths


# ----- Main generation logic -----

def generate_images(cfg: GenConfig) -> List[Path]:
    """
    Generate images from text according to GenConfig.

    Returns list of saved image Path objects.
    """
    # Optional HF login for private models
    if cfg.hf_token:
        try:
            hf_login(token=cfg.hf_token, add_to_git_credential=False)
            LOG.debug("Hugging Face token set via hf_login")
        except Exception as exc:
            LOG.warning("Unable to login to Hugging Face hub: %s", exc)

    # Pick dtype and device
    device = torch.device(cfg.device)

    # Pick scheduler class if provided
    scheduler_cls = _get_scheduler_by_name(cfg.scheduler)

    LOG.info("Loading pipeline (this may take a while): model=%s, dtype=%s", cfg.model_id, cfg.dtype)

    try:
        pipe_kwargs = {}
        if scheduler_cls is not None:
            pipe_kwargs["scheduler"] = scheduler_cls.from_pretrained(cfg.model_id, subfolder="scheduler")

        pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            safety_checker=None,  # explicit: do not use the legacy safety checker here
            **({} if not pipe_kwargs else pipe_kwargs),
        )
    except Exception as exc:
        LOG.exception("Failed to load pipeline for model %s: %s", cfg.model_id, exc)
        raise

    try:
        pipeline = pipeline.to(device)
    except Exception:
        LOG.warning("Couldn't move pipeline to device %s, continuing on CPU", device)
        pipeline = pipeline.to("cpu")
        device = torch.device("cpu")

    # Performance optimizations
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        LOG.debug("enable_attention_slicing not available for this pipeline")

    if cfg.enable_xformers and device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            LOG.info("Enabled xFormers memory efficient attention")
        except Exception:
            LOG.debug("xFormers not available or failed to enable; continuing")

    # Use generator to make generation deterministic given the seed
    generator = None
    if cfg.seed is not None:
        gen_device = "cpu" if device.type == "cpu" else device
        generator = torch.Generator(device=gen_device).manual_seed(int(cfg.seed))

    # Prepare call kwargs
    call_kwargs = dict(
        prompt=cfg.prompt,
        height=cfg.height,
        width=cfg.width,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        num_images_per_prompt=cfg.num_images,
    )
    if cfg.negative_prompt:
        call_kwargs["negative_prompt"] = cfg.negative_prompt
    if generator is not None:
        call_kwargs["generator"] = generator

    LOG.info(
        "Generating images: prompt_len=%d, size=%dx%d, images=%d, steps=%d, guidance=%.2f",
        len(cfg.prompt),
        cfg.width,
        cfg.height,
        cfg.num_images,
        cfg.num_inference_steps,
        cfg.guidance_scale,
    )

    try:
        # Diffusers pipelines return an object with .images list
        result = pipeline(**call_kwargs)
    except Exception as exc:
        LOG.exception("Generation failed: %s", exc)
        raise

    images = result.images if hasattr(result, "images") else []
    meta = {
        "prompt": cfg.prompt,
        "negative_prompt": cfg.negative_prompt,
        "model_id": cfg.model_id,
        "seed": cfg.seed,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "width": cfg.width,
        "height": cfg.height,
        "num_images": cfg.num_images,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    saved = _save_images(images, cfg.prompt, cfg.outdir, meta)
    LOG.info("Saved %d images to %s", len(saved), cfg.outdir)
    return saved


# ----- CLI -----

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images from text using Hugging Face Diffusers"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate images for")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt to discourage features")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model id (stable diffusion compatible)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (optional). You can set HF_TOKEN env var too.")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Directory to save generated images")
    parser.add_argument("--width", type=int, default=512, help="Image width (multiple of 8)")
    parser.add_argument("--height", type=int, default=512, help="Image height (multiple of 8)")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate per prompt (max 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic output")
    parser.add_argument("--scheduler", type=str, default=None, help="Optional scheduler: ddim, euler, dpmsolver")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cuda, cpu")
    parser.add_argument("--disable-xformers", action="store_true", help="Disable xFormers even if available")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    try:
        cfg = _validate_and_construct_config(args)
    except Exception as exc:
        LOG.error("Invalid arguments: %s", exc)
        return 2

    try:
        saved = generate_images(cfg)
        LOG.info("Generation succeeded. Files:\n%s", "\n".join(str(p) for p in saved))
        return 0
    except Exception as exc:
        LOG.exception("Generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
