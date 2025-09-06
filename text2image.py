#!/usr/bin/env python3
"""
text2image.py

A production-ready CLI utility to generate images from text prompts using HuggingFace Diffusers.

Features:
- Robust argument parsing and input validation
- Device auto-selection (CUDA/Metal/CPU)
- Deterministic generation via seed
- Performance optimizations (attention slicing, xformers if available)
- Secure handling of Hugging Face token via env var or CLI
- Clear logging and error handling

Usage example:
  python text2image.py --prompt "A cozy cabin in a snowy forest" --out output.png --steps 30 --seed 42

Requirements:
  See requirements.txt

"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

# Diffusers imports are inside functions to allow early validation and better error messages


# Module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(_handler)


@dataclass
class GenerationConfig:
    prompt: str
    out_path: str
    model_id: str = "runwayml/stable-diffusion-v1-5"
    hf_token: Optional[str] = None
    device: Optional[str] = None
    seed: Optional[int] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    dtype: Optional[str] = None  # 'fp16' or 'fp32'


def _validate_and_normalize_cfg(cfg: GenerationConfig) -> GenerationConfig:
    """Validate user input and normalize values.

    Raises ValueError on invalid inputs.
    """
    if not cfg.prompt or not isinstance(cfg.prompt, str):
        raise ValueError("prompt must be a non-empty string")

    if cfg.num_inference_steps <= 0 or cfg.num_inference_steps > 200:
        raise ValueError("num_inference_steps must be between 1 and 200")

    if cfg.guidance_scale < 1.0 or cfg.guidance_scale > 30.0:
        raise ValueError("guidance_scale should be in [1.0, 30.0]")

    # Stable Diffusion requires multiples of 8 for H/W in many implementations
    if cfg.height % 8 != 0 or cfg.width % 8 != 0:
        raise ValueError("height and width must be multiples of 8")

    # Output path sanity checks: avoid overwriting directories
    out_dir = os.path.dirname(cfg.out_path) or "."
    if os.path.isdir(cfg.out_path):
        raise ValueError(f"out_path {cfg.out_path} is a directory; please provide a file path")

    os.makedirs(out_dir, exist_ok=True)

    # Normalize device string
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.device not in {"cuda", "cpu"} and not cfg.device.startswith("mps"):
        logger.warning("Unknown device specified; falling back to CPU")
        cfg.device = "cpu"

    # dtype
    if cfg.dtype not in {None, "fp16", "fp32"}:
        raise ValueError("dtype must be one of: fp16, fp32, or None")

    return cfg


def _get_torch_dtype(dtype_flag: Optional[str]) -> torch.dtype:
    if dtype_flag == "fp16":
        return torch.float16
    return torch.float32


def _setup_pipeline(model_id: str, hf_token: Optional[str], device: str, dtype_flag: Optional[str]):
    """Load and prepare the Diffusers pipeline with sensible defaults and optimizations.

    Returns the loaded pipeline object and the torch dtype used.
    """
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    except Exception as exc:
        raise RuntimeError(
            "Failed to import diffusers. Ensure diffusers is installed (see requirements.txt)."
        ) from exc

    torch_dtype = _get_torch_dtype(dtype_flag)

    logger.info("Loading model %s ...", model_id)
    # Use scheduler that often performs well; allow user to change if needed
    try:
        # Permit use_auth_token for private models
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_auth_token=hf_token,
            safety_checker=None,  # leave safety to user/host; to enable, remove or set appropriately
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_id}'. Ensure the model id is correct and the HF token (if required) is valid."
        ) from exc

    # Use a faster scheduler if available
    try:
        scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler
    except Exception:
        logger.debug("Unable to swap scheduler; using default scheduler shipped with the model")

    # Move to device
    try:
        pipe = pipe.to(device)
    except Exception as exc:
        raise RuntimeError(f"Failed to move pipeline to device '{device}': {exc}") from exc

    # Performance: enable attention slicing
    try:
        pipe.enable_attention_slicing()
    except Exception:
        logger.debug("enable_attention_slicing not available for this pipeline")

    # Try enabling xformers memory efficient attention if available
    try:
        if torch.cuda.is_available():
            pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        logger.debug("xformers not available or failed to enable; continuing without it")

    return pipe, torch_dtype


def generate_image(cfg: GenerationConfig) -> str:
    """Generate an image from a text prompt and save it to out_path.

    Returns the path to the saved image.
    """
    cfg = _validate_and_normalize_cfg(cfg)

    # Load pipeline
    pipe, torch_dtype = _setup_pipeline(cfg.model_id, cfg.hf_token, cfg.device, cfg.dtype)

    # Prepare generator for reproducibility
    generator = None
    if cfg.seed is not None:
        if not isinstance(cfg.seed, int) or cfg.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        generator = torch.manual_seed(cfg.seed)

    # Run generation
    logger.info(
        "Generating image: steps=%d, guidance_scale=%.2f, size=%dx%d, device=%s",
        cfg.num_inference_steps,
        cfg.guidance_scale,
        cfg.width,
        cfg.height,
        cfg.device,
    )

    try:
        # Use autocast on CUDA when using fp16 for faster generation
        autocast_ctx = torch.autocast(cfg.device, dtype=torch.float16) if (cfg.device.startswith("cuda") and torch_dtype == torch.float16) else torch.no_grad()
        with autocast_ctx:
            output = pipe(
                prompt=cfg.prompt,
                height=cfg.height,
                width=cfg.width,
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=cfg.num_inference_steps,
                generator=None if cfg.seed is None else torch.Generator(device=cfg.device).manual_seed(cfg.seed),
            )
    except Exception as exc:
        raise RuntimeError(f"Image generation failed: {exc}") from exc

    if not hasattr(output, "images") or not output.images:
        raise RuntimeError("Pipeline returned no images")

    image = output.images[0]
    if not isinstance(image, Image.Image):
        # convert to PIL if it's a numpy array or tensor
        try:
            image = Image.fromarray(image)
        except Exception:
            raise RuntimeError("Generated output is not a valid image")

    # Save image
    try:
        image.save(cfg.out_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save image to {cfg.out_path}: {exc}") from exc

    logger.info("Saved image to %s", cfg.out_path)
    return cfg.out_path


def _parse_args(argv) -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate images from text prompts using HuggingFace Diffusers")

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to guide image generation")
    parser.add_argument("--out", type=str, required=True, help="Output image file path (e.g. ./out.png)")

    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id to use")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token. If not provided uses HF_TOKEN env var")

    parser.add_argument("--device", type=str, default=None, help="Device to run on: cuda, cpu, or mps (auto-detected if not provided)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic output")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (1-200)")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Output image height (must be multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Output image width (must be multiple of 8)")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default=None, help="Compute dtype to use (fp16 on capable GPUs improves speed and memory)")

    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")

    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    return GenerationConfig(
        prompt=args.prompt,
        out_path=args.out,
        model_id=args.model_id,
        hf_token=hf_token,
        device=args.device,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        dtype=args.dtype,
    )


def main(argv=None) -> int:
    """CLI entry point. Returns exit code (0 on success)."""
    try:
        cfg = _parse_args(argv)
        generate_image(cfg)
        return 0
    except Exception as exc:
        logger.exception("Error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
