#!/usr/bin/env python3
"""
text2image.py

Command-line utility to generate images from text prompts using Hugging Face Diffusers.

Features:
- Configurable model, scheduler, guidance scale, steps, seed, and image size
- Automatic device selection (CUDA/CPU)
- Safety checks for common parameter mistakes
- Typed functions, detailed logging, and helpful CLI

Usage example:
  python text2image.py --prompt "A serene landscape at sunset" --out_dir ./outputs --num_images 2 --height 512 --width 768 --model runwayml/stable-diffusion-v1-5

Note: A Hugging Face token may be required for some models. Provide it via --auth_token or set HF_AUTH_TOKEN env var.

"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

# Constants
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
MAX_DIMENSION = 2048  # safety upper bound for height/width
MIN_DIMENSION = 64

# Configure module-level logger
logger = logging.getLogger("text2image")


@dataclass
class Text2ImageConfig:
    prompt: str
    out_dir: Path
    model: str = DEFAULT_MODEL
    auth_token: Optional[str] = None
    num_images: int = 1
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    device: str = "auto"
    torch_dtype: Optional[str] = None


def configure_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def validate_config(cfg: Text2ImageConfig) -> None:
    """Validate configuration values and raise ValueError for invalid combos.

    - Enforces size multiples required by many diffusion models (multiple of 8)
    - Limits dimensions to reasonable bounds
    - Validates steps and guidance scale ranges
    """
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    if cfg.num_images < 1 or cfg.num_images > 16:
        raise ValueError("num_images must be between 1 and 16 (to avoid excessive memory use).")

    for name, val in (("height", cfg.height), ("width", cfg.width)):
        if val < MIN_DIMENSION or val > MAX_DIMENSION:
            raise ValueError(f"{name} must be between {MIN_DIMENSION} and {MAX_DIMENSION}.")
        if val % 8 != 0:
            raise ValueError(f"{name} must be a multiple of 8 (most diffusion models require this).")

    if not (1 <= cfg.num_inference_steps <= 150):
        raise ValueError("num_inference_steps must be in the range [1, 150].")

    if not (0.0 <= cfg.guidance_scale <= 30.0):
        raise ValueError("guidance_scale must be in the range [0.0, 30.0].")


def pick_device(requested: str = "auto") -> str:
    """Select the execution device.

    Args:
        requested: 'auto', 'cpu', or 'cuda'.

    Returns:
        Device string to use with torch.
    """
    requested = requested.lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(model_id: str, device: str, auth_token: Optional[str], torch_dtype: Optional[str]) -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline with appropriate device and dtype.

    Args:
        model_id: Hugging Face model id
        device: 'cpu' or 'cuda'
        auth_token: HF token or None
        torch_dtype: Optional string, 'float16' recommended for CUDA

    Returns:
        Instantiated StableDiffusionPipeline
    """
    # Determine dtype
    dtype = None
    if device == "cuda":
        # Use float16 for GPU to reduce memory, fallback to float32 on explicit request
        if torch_dtype == "float16":
            dtype = torch.float16
        else:
            # If torch supports bfloat16 and user wants it, else float16 by default for speed/memory
            dtype = torch.float16
    else:
        dtype = torch.float32

    logger.info("Loading model '%s' on device=%s dtype=%s", model_id, device, dtype)

    # Allow auth_token via env var 'HF_AUTH_TOKEN' as a convenience
    hf_token = auth_token or os.environ.get("HF_AUTH_TOKEN")

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=hf_token,
            torch_dtype=dtype,
            safety_checker=None,  # Keep safety checker disabled here; models may differ
        )
    except TypeError:
        # Some versions of diffusers expect no use_auth_token kwarg; try without
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # Move to device
    pipe = pipe.to(device)

    # Enable attention slicing if available to reduce memory usage on constrained GPUs
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
            logger.debug("Enabled attention slicing to reduce peak memory usage.")
        except Exception:
            logger.debug("Could not enable attention slicing.")

    return pipe


def generate_images(cfg: Text2ImageConfig) -> List[Path]:
    """Generate images from a text prompt using the configured pipeline.

    Args:
        cfg: Text2ImageConfig instance with validated values

    Returns:
        List of paths to generated images
    """
    device = pick_device(cfg.device)
    pipe = load_pipeline(cfg.model, device, cfg.auth_token, cfg.torch_dtype)

    generator = None
    if cfg.seed is not None:
        logger.debug("Using deterministic seed: %s", cfg.seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.seed)

    images_out: List[Path] = []
    timestamp = int(time.time())
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Use automatic mixed precision on CUDA
    use_autocast = device == "cuda"

    logger.info(
        "Generating %d image(s): model=%s height=%d width=%d steps=%d guidance=%s device=%s",
        cfg.num_images,
        cfg.model,
        cfg.height,
        cfg.width,
        cfg.num_inference_steps,
        cfg.guidance_scale,
        device,
    )

    for i in range(cfg.num_images):
        out_name = f"img_{timestamp}_{i+1}.png"
        out_path = cfg.out_dir / out_name
        prompt = cfg.prompt

        try:
            if use_autocast:
                with torch.autocast(device_type="cuda"):
                    image = pipe(
                        prompt,
                        height=cfg.height,
                        width=cfg.width,
                        guidance_scale=cfg.guidance_scale,
                        num_inference_steps=cfg.num_inference_steps,
                        generator=generator,
                    ).images[0]
            else:
                image = pipe(
                    prompt,
                    height=cfg.height,
                    width=cfg.width,
                    guidance_scale=cfg.guidance_scale,
                    num_inference_steps=cfg.num_inference_steps,
                    generator=generator,
                ).images[0]

            # Validate image type and save
            if not isinstance(image, Image.Image):
                raise RuntimeError("Pipeline did not return a PIL.Image instance.")

            image.save(out_path, format="PNG")
            images_out.append(out_path)
            logger.info("Saved image to %s", out_path)
        except Exception as e:
            logger.exception("Failed to generate or save image %d: %s", i + 1, str(e))
            # Continue generating other images where possible

    return images_out


def parse_args(argv: Optional[List[str]] = None) -> Text2ImageConfig:
    """Parse CLI arguments and return a Text2ImageConfig instance.

    Args:
        argv: Optional list of args (for tests). If None, argparse takes from sys.argv.
    """
    parser = argparse.ArgumentParser(description="Text->Image generation using Diffusers (Stable Diffusion)")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt to generate the image from")
    parser.add_argument("--out_dir", "-o", default="./outputs", help="Directory to write generated images")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--auth_token", default=None, help="Hugging Face auth token (or set HF_AUTH_TOKEN env var)")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate (1-16)")
    parser.add_argument("--height", type=int, default=512, help="Image height (multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (multiple of 8)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps (1-150)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for deterministic output")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device")
    parser.add_argument("--torch_dtype", default=None, choices=[None, "float16", "float32"], help="Torch dtype for model weights (float16 recommended on CUDA)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    ns = parser.parse_args(argv)

    cfg = Text2ImageConfig(
        prompt=ns.prompt,
        out_dir=Path(ns.out_dir),
        model=ns.model,
        auth_token=ns.auth_token,
        num_images=ns.num_images,
        height=ns.height,
        width=ns.width,
        guidance_scale=ns.guidance_scale,
        num_inference_steps=ns.num_inference_steps,
        seed=ns.seed,
        device=ns.device,
        torch_dtype=ns.torch_dtype,
    )

    configure_logging(ns.verbose)

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 == success)
    """
    try:
        cfg = parse_args(argv)
        validate_config(cfg)
        generated = generate_images(cfg)
        if not generated:
            logger.error("No images were generated. See logs for details.")
            return 2
        logger.info("Generation complete. %d image(s) created.", len(generated))
        return 0
    except Exception as e:
        logger.exception("Error: %s", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
