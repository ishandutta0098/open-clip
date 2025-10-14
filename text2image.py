# -*- coding: utf-8 -*-
"""
text2image.py

A production-ready CLI utility to generate images from text prompts using HuggingFace Diffusers

Features:
- Configurable model, device, precision (fp16/float32)
- Stable and reproducible generation using seeds
- Input validation (sizes, steps, scales)
- Performance options (attention slicing, optional xformers if available)
- Detailed logging and error handling
- Safe defaults and guidance for authenticated/private models

Usage examples:
  python text2image.py --prompt "A scenic mountain sunrise" --outdir ./outputs --num_images 3 --width 512 --height 512 --steps 30

Requirements:
- A Hugging Face token is recommended for some models. Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN environment variable.

"""
from __future__ import annotations

import argparse
import os
import sys
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


LOGGER = logging.getLogger("text2image")


def setup_logging(verbosity: int = 1) -> None:
    """Configure logging.

    Args:
        verbosity: 0 = WARNING, 1 = INFO, >=2 = DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)


def validate_image_size(value: int) -> int:
    """Validate width/height for Stable Diffusion: multiples of 8 and positive.

    Args:
        value: dimension in pixels

    Returns:
        value if valid

    Raises:
        argparse.ArgumentTypeError: if invalid
    """
    try:
        iv = int(value)
    except Exception:
        raise argparse.ArgumentTypeError("Must be an integer")
    if iv <= 0:
        raise argparse.ArgumentTypeError("Size must be positive")
    if iv % 8 != 0:
        raise argparse.ArgumentTypeError("Size must be a multiple of 8 for Stable Diffusion models")
    return iv


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate images from text prompts using Hugging Face Diffusers")

    parser.add_argument("--prompt", type=str, required=True, help="Main text prompt to generate the image")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt (things to avoid)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate (integer >=1)")
    parser.add_argument("--width", type=validate_image_size, default=512, help="Image width (multiple of 8)")
    parser.add_argument("--height", type=validate_image_size, default=512, help="Image height (multiple of 8)")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps (e.g., 20-50)")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale (classifier-free guidance)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id to load")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda). Default attempts cuda if available")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 (recommended on recent NVIDIA GPUs)")
    parser.add_argument("--revision", type=str, default=None, help="Model revision to use (optional)")
    parser.add_argument("--use_dpm_solver", action="store_true", help="Switch to DPM-Solver scheduler for faster sampling (optional)")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Alias for --steps; provided for compatibility")
    parser.add_argument("--verbose", action="count", default=0, help="Increase logging verbosity (repeat for more)")

    return parser


def choose_device(cli_device: Optional[str], prefer_fp16: bool) -> torch.device:
    """Choose device based on CLI and environment.

    Args:
        cli_device: device string passed by user
        prefer_fp16: whether the pipeline will be configured to use fp16

    Returns:
        torch.device object
    """
    if cli_device:
        device = cli_device.lower()
        if device not in ("cpu", "cuda"):
            raise ValueError("--device must be 'cpu' or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    # auto-select
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipeline(model_id: str, device: torch.device, use_fp16: bool = False, revision: Optional[str] = None, use_dpm: bool = False) -> StableDiffusionPipeline:
    """Load and configure a Stable Diffusion pipeline.

    Args:
        model_id: Hugging Face model id
        device: torch device to move the pipeline to
        use_fp16: whether to prefer torch.float16 dtype
        revision: optional model revision
        use_dpm: whether to switch scheduler to DPMSolverMultistepScheduler

    Returns:
        configured StableDiffusionPipeline
    """
    # Choose torch dtype
    torch_dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32

    # Log information
    LOGGER.info("Loading model '%s' to device=%s dtype=%s", model_id, device, torch_dtype)

    # Use from_pretrained with safety: do not force revision unless provided
    try:
        model_kwargs = {"torch_dtype": torch_dtype}
        if revision:
            model_kwargs["revision"] = revision

        pipe = StableDiffusionPipeline.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        LOGGER.exception("Failed to load model '%s'. Ensure the model id is correct and you have access. If the model requires auth, set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN.", model_id)
        raise

    # Move to device
    pipe.to(device)

    # Performance tweaks
    try:
        pipe.enable_attention_slicing()
        LOGGER.debug("Enabled attention slicing for reduced memory usage")
    except Exception:
        LOGGER.debug("Could not enable attention slicing")

    # Try enabling xformers if available for memory-efficient attention
    try:
        pipe.enable_xformers_memory_efficient_attention()
        LOGGER.debug("Enabled xformers memory efficient attention")
    except Exception:
        LOGGER.debug("xformers not available or could not be enabled")

    # Optionally swap to DPM-Solver sampler (faster/quality trade-offs)
    if use_dpm:
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            LOGGER.info("Switched scheduler to DPM-Solver")
        except Exception:
            LOGGER.warning("Failed to set DPM-Solver scheduler; continuing with default scheduler")

    return pipe


def generate_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    num_images: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
) -> List[Image.Image]:
    """Generate images using the provided pipeline and configuration.

    Args:
        pipe: initialized StableDiffusionPipeline
        prompt: main prompt
        negative_prompt: optional negative prompt
        num_images: how many images to produce
        width: output width
        height: output height
        guidance_scale: classifier-free guidance scale
        num_inference_steps: sampling steps
        seed: optional int seed for reproducibility

    Returns:
        List of PIL.Image objects
    """
    # Input validation
    if num_images < 1:
        raise ValueError("num_images must be >= 1")
    if num_inference_steps < 1 or num_inference_steps > 200:
        LOGGER.warning("num_inference_steps value %s looks unusual. Typical range: 10-100.", num_inference_steps)

    generator = None
    if seed is not None:
        # Use a generator on the same device as the pipeline
        device = pipe.device
        try:
            generator = torch.Generator(device=device.type).manual_seed(seed)
        except Exception:
            # Fallback to CPU generator if device-specific generator fails
            generator = torch.Generator().manual_seed(seed)
        LOGGER.info("Generation seed set to %d", seed)

    images: List[Image.Image] = []

    # Create a single call that supports batching internally via num_images param
    # The pipeline typically supports num_images_per_prompt
    # Use inference with autocast when fp16
    use_autocast = (pipe.unet.dtype == torch.float16) and (pipe.device.type == "cuda")

    try:
        if use_autocast:
            LOGGER.debug("Using autocast for fp16 inference")
            with torch.autocast(device_type=pipe.device.type):
                output = pipe(
                    prompt=[prompt] * num_images,
                    negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
        else:
            output = pipe(
                prompt=[prompt] * num_images,
                negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
    except Exception:
        LOGGER.exception("Image generation failed")
        raise

    # The pipeline returns a SafeOutput object with images attribute
    raw_images = getattr(output, "images", None) or output

    for img in raw_images:
        if not isinstance(img, Image.Image):
            # Convert to PIL if numpy array or torch tensor
            try:
                img = Image.fromarray(img.astype("uint8"))
            except Exception:
                LOGGER.debug("Could not convert image to PIL; skipping")
                continue
        images.append(img)

    return images


def save_images(images: List[Image.Image], outdir: Path, base_name: str, start_index: int = 0) -> List[Path]:
    """Save list of PIL images to disk.

    Args:
        images: list of PIL.Image
        outdir: output directory
        base_name: base filename (no extension)
        start_index: starting index for numbering

    Returns:
        list of saved file paths
    """
    outdir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for i, img in enumerate(images, start=start_index):
        filename = f"{base_name}-{i:04d}.png"
        out_path = outdir / filename
        # Ensure RGB for PNG
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path, format="PNG")
        LOGGER.info("Saved image: %s", out_path)
        saved_paths.append(out_path)
    return saved_paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_arg_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    # Map aliases
    if args.num_inference_steps is not None:
        args.steps = args.num_inference_steps

    # Basic validation
    if args.num_images < 1:
        LOGGER.error("--num_images must be >= 1")
        return 2
    if args.steps < 1:
        LOGGER.error("--steps must be >= 1")
        return 2

    # Determine device
    try:
        device = choose_device(args.device, args.fp16)
    except ValueError as e:
        LOGGER.error(str(e))
        return 2

    # Load pipeline
    try:
        pipe = load_pipeline(model_id=args.model, device=device, use_fp16=args.fp16, revision=args.revision, use_dpm=args.use_dpm_solver)
    except Exception as e:
        LOGGER.error("Unable to load pipeline: %s", e)
        return 3

    # Generate images
    try:
        images = generate_images(
            pipe=pipe,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            width=args.width,
            height=args.height,
            guidance_scale=args.scale,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
    except Exception as e:
        LOGGER.error("Generation failed: %s", e)
        return 4

    # Save results
    try:
        base_name = "sd"
        saved = save_images(images, Path(args.outdir), base_name)
        LOGGER.info("Saved %d images to %s", len(saved), args.outdir)
    except Exception as e:
        LOGGER.exception("Failed to save images: %s", e)
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
