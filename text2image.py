#!/usr/bin/env python3
"""
text2image.py

A production-ready utility to generate images from text prompts using Hugging Face Diffusers.

Features:
- Automatic device selection (GPU with mixed precision or CPU fallback)
- Input validation (prompt, dimensions, seed)
- Deterministic seeding for reproducible outputs
- Safety/NSFW detection handling when available in the pipeline
- Configurable inference parameters (steps, guidance scale, batch size)
- Image file naming and output directory management
- Detailed logging, error handling and type hints

Usage examples:
  python text2image.py --prompt "A fantasy landscape, sunrise" --outdir ./outputs --num_inference_steps 30

See generate_image() docstring for programmatic usage.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from PIL import Image
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "Missing required libraries. Ensure 'torch', 'diffusers' and 'Pillow' are installed."
    ) from exc


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("text2image")
_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def sanitize_filename(s: str, max_len: int = 200) -> str:
    """Return a filesystem-safe filename derived from the input string.

    Non-alphanumeric characters are replaced with underscores. The result is truncated
    to max_len characters.

    Args:
        s: Input string to sanitize.
        max_len: Maximum length of the returned filename.

    Returns:
        A sanitized filename string.
    """
    if not s:
        return "untitled"
    sanitized = FILENAME_SAFE_RE.sub("_", s).strip("._-")
    if not sanitized:
        sanitized = "untitled"
    return sanitized[:max_len]


def ensure_outdir(path: str) -> Path:
    """Ensure the output directory exists and is writable.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.

    Raises:
        PermissionError: If the directory cannot be created or written to.
    """
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    if not os.access(str(p), os.W_OK):
        raise PermissionError(f"No write permission for output directory: {p}")
    return p


def validate_dimensions(width: int, height: int) -> None:
    """Validate width and height for common diffusion models.

    Most stable diffusion-based models require both width and height to be divisible by 8.

    Raises:
        ValueError: If invalid dimensions are provided.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("Width and height must be divisible by 8 for most diffusion models")
    if width > 2048 or height > 2048:
        logger.warning("Large output sizes may cause OOM on many GPUs")


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


def _select_device() -> Tuple[torch.device, Optional[torch.dtype]]:
    """Select the best available device and precision.

    Returns:
        (device, dtype) where dtype may be None for CPU float32.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Use float16 on CUDA to reduce memory usage and speed up inference
        dtype = torch.float16
        logger.info("CUDA available. Using GPU with float16 precision.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        logger.info("CUDA not available. Falling back to CPU with float32 precision.")
    return device, dtype


def _load_pipeline(model_id: str, device: torch.device, dtype: Optional[torch.dtype]):
    """Load the diffusers pipeline with sensible defaults and optimizations.

    Args:
        model_id: HF model identifier or local path.
        device: torch.device to move the pipeline to.
        dtype: Torch dtype for model weights (torch.float16 recommended for GPU).

    Returns:
        An initialized DiffusionPipeline.
    """
    # Choose scheduler that often improves sample quality and speed
    # DPMSolverMultistepScheduler is a good default; diffusers will switch if incompatible
    kwargs: Dict = {"torch_dtype": dtype} if dtype is not None else {}

    logger.info("Loading pipeline model: %s", model_id)
    pipeline = DiffusionPipeline.from_pretrained(model_id, **kwargs)

    # Move to device
    pipeline = pipeline.to(device)

    # Enable xformers memory efficient attention if available for faster inference and lower memory
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xFormers memory efficient attention")
    except Exception:
        logger.debug("xFormers not available or failed to enable")

    # If scheduler attribute exists and DPMSolverMultistepScheduler is compatible, set it
    try:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        logger.info("Using DPMSolverMultistepScheduler for improved sampling")
    except Exception:
        logger.debug("Could not switch scheduler to DPM solver; using default scheduler")

    return pipeline


def generate_image(
    prompt: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    out_dir: str = "outputs",
    seed: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    negative_prompt: Optional[str] = None,
    num_images_per_prompt: int = 1,
) -> List[Path]:
    """Generate one or more images from a text prompt.

    Args:
        prompt: Text prompt describing the desired image. Must be non-empty.
        model_id: Hugging Face model id or local path to a diffusion model.
        out_dir: Directory to store generated images.
        seed: Optional random seed for reproducible outputs. If None, non-deterministic.
        num_inference_steps: Number of denoising steps. More steps = more detail, slower.
        guidance_scale: Classifier-free guidance scale. 7.5 is typical.
        width: Output width in pixels (must be divisible by 8).
        height: Output height in pixels (must be divisible by 8).
        negative_prompt: Optional negative prompt to guide what not to generate.
        num_images_per_prompt: Number of images to generate per prompt. Keep small to avoid OOM.

    Returns:
        List of Paths to saved image files.

    Raises:
        ValueError: For invalid inputs.
        RuntimeError: For pipeline or inference-related issues.
    """
    # Validate inputs
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    if num_images_per_prompt <= 0 or num_images_per_prompt > 8:
        # Protect resources: don't allow huge batches
        raise ValueError("num_images_per_prompt must be between 1 and 8 (inclusive)")
    validate_dimensions(width, height)

    out_path = ensure_outdir(out_dir)

    device, dtype = _select_device()

    pipeline = _load_pipeline(model_id=model_id, device=device, dtype=dtype)

    # Reproducible generator if seed provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        logger.info("Using provided seed: %d", seed)

    # Prepare kwargs for pipeline
    call_kwargs: Dict = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "num_images_per_prompt": int(num_images_per_prompt),
        "generator": generator,
    }

    if negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt

    logger.info(
        "Generating image(s) with model=%s, steps=%s, guidance=%.2f, size=%dx%d, batch=%d",
        model_id,
        num_inference_steps,
        guidance_scale,
        width,
        height,
        num_images_per_prompt,
    )

    try:
        # Use the pipeline to generate images. Return dicts differ across diffusers versions; allow both.
        output = pipeline(**call_kwargs)

        # Some pipeline versions return a tuple (images, nsfw_flags) when return_dict=False
        images = None
        nsfw_flags = None
        if hasattr(output, "images"):
            # StableDiffusionPipelineOutput
            images = output.images
            nsfw_flags = getattr(output, "nsfw_content_detected", None)
        elif isinstance(output, tuple):
            # (images, nsfw)
            images = output[0]
            if len(output) > 1:
                nsfw_flags = output[1]
        else:
            raise RuntimeError("Unexpected pipeline output format")

        if images is None:
            raise RuntimeError("Pipeline did not return images")

        saved_paths: List[Path] = []
        base_name = sanitize_filename(prompt)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        for idx, img in enumerate(images):
            if not isinstance(img, Image.Image):
                # Some pipelines return torch tensors -> convert
                img = Image.fromarray(img)

            # If nsfw flag exists and is True for this image, skip saving and warn
            if nsfw_flags:
                try:
                    flag = nsfw_flags[idx]
                except Exception:
                    flag = None
                if flag:
                    logger.warning("Image %d flagged as NSFW by model safety_checker; skipping save.", idx)
                    continue

            filename = f"{base_name}_{timestamp}_{idx}.png"
            out_file = out_path / filename
            img.save(out_file)
            saved_paths.append(out_file)
            logger.info("Saved image to %s", out_file)

        if not saved_paths:
            raise RuntimeError("No images saved; possibly all outputs were filtered by safety checker")

        return saved_paths

    except Exception as exc:
        logger.exception("Failed to generate images: %s", exc)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    This function is separated for easier unit testing.
    """
    parser = argparse.ArgumentParser(description="Text-to-Image generator using Hugging Face Diffusers")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate an image for")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="HF model id or local path")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory for generated images")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Output width (divisible by 8)")
    parser.add_argument("--height", type=int, default=512, help="Output height (divisible by 8)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to avoid elements")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt (1-8)")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        saved = generate_image(
            prompt=args.prompt,
            model_id=args.model,
            out_dir=args.outdir,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=args.num_images_per_prompt,
        )
        logger.info("Generation complete. %d image(s) saved.", len(saved))
        return 0
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
