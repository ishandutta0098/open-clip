#!/usr/bin/env python3
"""
text2image.py

A production-ready command-line utility to generate images from text prompts using
Hugging Face Diffusers (Stable Diffusion). The script contains input validation,
logging, device selection (GPU/CPU), reproducibility via seeding, and optional
safety checker disabling.

Usage example:
  python text2image.py \
    --prompt "A photorealistic painting of a red fox in a snowy forest" \
    --model "runwayml/stable-diffusion-v1-5" \
    --output ./out.png \
    --height 512 --width 512 --seed 42 --num_inference_steps 30 --guidance_scale 7.5

Environment variables:
  HF_TOKEN: (optional) Hugging Face access token if the model requires it.

Requirements: See requirements.txt for pinned versions.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline

# Configure module-level logger
LOGGER_NAME = "text2image"
logger = logging.getLogger(LOGGER_NAME)


def configure_logging(verbosity: int = 1) -> None:
    """Configure logging for the script.

    Args:
        verbosity: integer controlling log level (0=WARNING, 1=INFO, 2=DEBUG).
    """
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)
    root.addHandler(handler)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of arguments (for testing). If None uses sys.argv.

    Returns:
        argparse.Namespace with parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using Hugging Face Diffusers."
    )

    parser.add_argument(
        "--prompt",
        required=True,
        type=str,
        help="Text prompt to generate the image from.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model repo id (default: runwayml/stable-diffusion-v1-5).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.png"),
        help="Output image path (PNG).",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height in pixels. Must be divisible by 8.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width in pixels. Must be divisible by 8.",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps (higher = more quality, slower).",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale. 1.0 = no guidance.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not set uses non-deterministic seed.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run the pipeline on. 'auto' prefers CUDA if available.",
    )

    parser.add_argument(
        "--disable_safety_checker",
        action="store_true",
        help="Disable the built-in safety checker. Use with caution.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 (half precision) when running on CUDA for memory savings.",
    )

    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Logging verbosity: 0=WARNING, 1=INFO, 2=DEBUG.",
    )

    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations and values.

    Raises:
        ValueError: if validation fails.
    """
    if args.height <= 0 or args.width <= 0:
        raise ValueError("Height and width must be positive integers.")

    # Stable Diffusion requirement: dimensions divisible by 8 (or 32 depending on model). 8 is safe.
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("Height and width must be divisible by 8 for Stable Diffusion models.")

    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")

    if args.guidance_scale < 1.0:
        logger.warning("guidance_scale < 1.0 is uncommon; results may be unexpected.")


def pick_device(preferred: str) -> str:
    """Choose an execution device based on preference and availability.

    Args:
        preferred: one of 'auto', 'cpu', 'cuda'.

    Returns:
        device string that can be passed to .to(device) (e.g. 'cpu' or 'cuda').
    """
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return "cuda"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(
    model_id: str,
    device: str,
    use_fp16: bool,
    disable_safety: bool,
    hf_token: Optional[str] = None,
) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline from the Hugging Face Hub.

    The function attempts to optimize device/precision. If fp16 is requested but a
    CUDA device is not present, it falls back to fp32.

    Args:
        model_id: HF repo id for the model.
        device: 'cpu' or 'cuda'.
        use_fp16: whether to attempt fp16 precision.
        disable_safety: whether to disable the safety checker.
        hf_token: optional Hugging Face token.

    Returns:
        An initialized StableDiffusionPipeline moved to the requested device.

    Raises:
        RuntimeError: on loading failures.
    """
    logger.info("Loading pipeline for model '%s' on device %s", model_id, device)

    # Choose dtype
    dtype = None
    if device == "cuda" and use_fp16 and torch.cuda.is_available():
        dtype = torch.float16
        logger.info("Attempting to use fp16 (torch.float16) for reduced memory usage")
    else:
        dtype = torch.float32

    try:
        # from_pretrained will cache the weights. If the model requires authentication,
        # HF_TOKEN environment variable can be provided.
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_auth_token=hf_token,
        )
    except Exception as exc:
        logger.exception("Failed to load model '%s': %s", model_id, exc)
        raise RuntimeError(f"Failed to load model '{model_id}': {exc}") from exc

    # Optionally disable safety checker - note: only do this with caution
    if disable_safety:
        try:
            # Some pipelines expose a safety_checker attribute, others may not.
            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None  # type: ignore[attr-defined]
                logger.warning("Safety checker disabled. Generated content will not be filtered.")
        except Exception:
            logger.debug("Safety checker could not be disabled or was not present.")

    # Move to device
    try:
        pipe = pipe.to(device)
    except Exception as exc:
        logger.exception("Failed to move pipeline to device %s: %s", device, exc)
        raise RuntimeError(f"Failed to move pipeline to device {device}: {exc}") from exc

    # Try to enable memory efficient attention if available
    try:
        pipe.enable_attention_slicing()
        # xformers may provide further speedups if installed and available
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.debug("Enabled xFormers memory efficient attention")
            except Exception:
                logger.debug("xFormers not available or failed to enable (optional)")
    except Exception:
        # Not all pipeline implementations have these methods
        logger.debug("Could not enable attention slicing or xformers (optional)")

    logger.info("Model loaded and ready")
    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int] = None,
) -> Image.Image:
    """Generate a single image from a text prompt.

    Args:
        pipe: preloaded StableDiffusionPipeline.
        prompt: text prompt.
        height: output height.
        width: output width.
        num_inference_steps: number of denoising steps.
        guidance_scale: classifier-free guidance scale.
        seed: optional random seed for deterministic results.

    Returns:
        PIL.Image produced by the model.

    Raises:
        RuntimeError: if generation fails.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device.type if hasattr(pipe, "device") else "cpu")
        generator = torch.Generator(device=str(pipe.device)) if hasattr(pipe, "device") else torch.Generator()
        try:
            generator = torch.Generator(device=str(pipe.device))
            generator.manual_seed(seed)
        except Exception:
            # Fallback for older torch versions
            generator = torch.Generator()
            generator.manual_seed(seed)

    logger.info(
        "Generating image (h=%d, w=%d, steps=%d, guidance=%.2f, seed=%s)",
        height,
        width,
        num_inference_steps,
        guidance_scale,
        str(seed),
    )

    try:
        # Generate
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # output.images is typically a list of PIL Images
        images = getattr(output, "images", None)
        if not images:
            # Some pipeline versions return a dict-like object
            if isinstance(output, dict) and "images" in output:
                images = output["images"]

        if not images:
            raise RuntimeError("Generation finished but no images were returned by the pipeline.")

        return images[0]

    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        raise RuntimeError(f"Image generation failed: {exc}") from exc


def ensure_parent_dir(path: Path) -> None:
    """Ensure that the parent directory exists and is writable.

    Args:
        path: output path to ensure.

    Raises:
        PermissionError: if directory cannot be created or is not writable.
    """
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"Output directory '{parent}' is not writable.")


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entrypoint.

    Returns:
        exit code (0 on success).
    """
    args = parse_args(argv)
    configure_logging(args.verbosity)

    try:
        validate_args(args)
    except Exception as exc:
        logger.error("Invalid arguments: %s", exc)
        return 2

    device = pick_device(args.device)
    hf_token = os.environ.get("HF_TOKEN")

    try:
        pipe = load_pipeline(
            model_id=args.model,
            device=device,
            use_fp16=args.fp16,
            disable_safety=args.disable_safety_checker,
            hf_token=hf_token,
        )
    except Exception as exc:
        logger.error("Failed to initialize model pipeline: %s", exc)
        return 3

    try:
        ensure_parent_dir(args.output)
    except Exception as exc:
        logger.error("Output path error: %s", exc)
        return 4

    start = time.time()
    try:
        image = generate_image(
            pipe=pipe,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    except Exception as exc:
        logger.error("Generation error: %s", exc)
        return 5

    try:
        # Ensure output path has a .png extension
        out_path = args.output
        if out_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            out_path = out_path.with_suffix(".png")

        image.save(out_path)
        elapsed = time.time() - start
        logger.info("Saved image to %s (took %.2f seconds)", out_path, elapsed)
    except Exception as exc:
        logger.exception("Failed to save image: %s", exc)
        return 6

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
