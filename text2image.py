#!/usr/bin/env python3
"""
text2image.py

Command-line tool to generate images from text prompts using Hugging Face Diffusers.

Features:
- Uses Stable Diffusion via diffusers
- GPU/CPU automatic device selection
- Mixed precision when CUDA is available (fp16)
- Batch generation support and deterministic seeding
- Input validation, logging, and safe resource cleanup
- Supports passing HF_TOKEN via environment or CLI

Usage:
    python text2image.py --prompt "A fantasy castle on a hill" --outdir outputs --num_images 4

Environment variables:
    HF_TOKEN  - Hugging Face access token (optional if public model)

Note: This script assumes compatibility between installed versions of torch, diffusers,
      accelerate, and transformers. See requirements.txt for recommended versions.

Security note: The script will not upload artifacts anywhere. Be aware of license and
             safety rules for the model you choose. Pay attention to the model's intended
             use and content filters.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
import math
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except Exception as exc:  # pragma: no cover - runtime dependency errors
    raise RuntimeError(
        "Missing required dependencies. Make sure you installed packages from requirements.txt"
    ) from exc

from PIL import Image


# ----------------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("text2image")


# ----------------------------------------------------------------------------
# Utility and Validation
# ----------------------------------------------------------------------------

def _validate_dimensions(width: int, height: int) -> None:
    """Validate width/height constraints for common diffusion models.

    Many stable diffusion models require width and height to be divisible by 8.
    This function raises ValueError if invalid.

    Args:
        width: Output image width.
        height: Output image height.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("width and height must be divisible by 8 for most models")


def _create_output_dir(path: Path) -> Path:
    """Create output directory (including parents) and return absolute path.

    Args:
        path: Target output directory path.
    Returns:
        Absolute Path object.
    """
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------------------------------------------------------
# Core generation logic
# ----------------------------------------------------------------------------

def generate_images(
    prompt: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    hf_token: Optional[str] = None,
    outdir: str = "outputs",
    num_images: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    batch_size: int = 1,
    dtype: Optional[str] = None,
) -> List[Path]:
    """Generate images for a given prompt using HuggingFace diffusers Stable Diffusion pipeline.

    Args:
        prompt: Text prompt to condition the model.
        model_id: Hugging Face model repo id to load.
        hf_token: Optional HF token (or rely on local cache / public access).
        outdir: Directory where generated images will be saved.
        num_images: Total number of images to generate.
        num_inference_steps: Number of denoising steps (higher -> better quality, slower).
        guidance_scale: Classifier-free guidance scale.
        height: Output height (must be divisible by 8).
        width: Output width (must be divisible by 8).
        seed: Optionally set a deterministic seed.
        device: Force device (e.g. "cuda", "cpu"). If None, will auto-select.
        batch_size: How many images to generate per pipeline call.
        dtype: Optional string indicating dtype to use ("fp16" or "fp32"). Defaults to fp16 on CUDA.

    Returns:
        List of Paths to saved images.

    Raises:
        RuntimeError: If the backend fails to load or generation fails.
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt must be a non-empty string")

    _validate_dimensions(width, height)

    if num_images <= 0:
        raise ValueError("num_images must be >= 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    device_str = device
n    # auto device selection
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device_str)

    # determine torch dtype
    if dtype is None:
        dtype = "fp16" if (device_str == "cuda") else "fp32"
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

    outdir_path = _create_output_dir(Path(outdir))

    # Load pipeline
    try:
        logger.info("Loading model %s (torch_dtype=%s)", model_id, torch_dtype)
        # Use an explicit scheduler for potentially improved quality/speed tradeoffs
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            revision=None,
            use_auth_token=hf_token,
        )

        # if the model does not have a scheduler or we want a deterministic one, set it
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            # Not fatal - continue with default scheduler
            logger.debug("Could not replace scheduler, continuing with default")

        # Move to device
        pipe = pipe.to(device_str)

        # Enable attention slicing and VAE tiling for memory efficiency
        try:
            pipe.enable_attention_slicing()
        except Exception:
            logger.debug("enable_attention_slicing not supported on this pipeline version")

    except Exception as exc:
        logger.exception("Failed to load pipeline for model %s", model_id)
        raise RuntimeError("Failed to load model pipeline") from exc

    total_saved: List[Path] = []

    # compute number of batches
    batches = math.ceil(num_images / batch_size)

    try:
        for batch_idx in range(batches):
            current_batch_size = min(batch_size, num_images - batch_idx * batch_size)

            # deterministic generator per batch
            generator = None
            if seed is not None:
                # use CPU generator seeded but on same device as pipeline for reproducibility
                gen_device = device_str if device_str == "cuda" else "cpu"
                generator = torch.Generator(device=gen_device)
                # derive a batch seed so each batch differs while reproducible
                batch_seed = int(seed + batch_idx)
                generator.manual_seed(batch_seed)

            logger.info(
                "Generating batch %d/%d (size=%d): prompt='%s'",
                batch_idx + 1,
                batches,
                current_batch_size,
                (prompt if len(prompt) <= 80 else prompt[:77] + "..."),
            )

            # Context manager for mixed precision on CUDA
            images = None
            if device_str == "cuda" and torch_dtype == torch.float16:
                with torch.autocast(device_type="cuda"):
                    result = pipe(
                        prompt=[prompt] * current_batch_size,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
            else:
                result = pipe(
                    prompt=[prompt] * current_batch_size,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            images = result.images

            # Save images
            for i, img in enumerate(images):
                unique_id = uuid.uuid4().hex[:8]
                timestamp = int(time.time())
                filename = f"sd_{timestamp}_{batch_idx}_{i}_{unique_id}.png"
                save_path = outdir_path / filename
                # convert to RGB and save
                if not isinstance(img, Image.Image):
                    # Some pipelines return numpy arrays
                    img = Image.fromarray(img)
                img = img.convert("RGB")
                img.save(save_path, format="PNG")
                total_saved.append(save_path)
                logger.info("Saved image: %s", save_path)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user - cleaning up and exiting")
    except Exception as exc:
        logger.exception("Generation failed")
        raise
    finally:
        # best-effort resource cleanup
        try:
            if device_str == "cuda":
                # release VRAM
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to empty cuda cache")

    return total_saved


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images from text using HuggingFace diffusers (Stable Diffusion)."
    )
    parser.add_argument("--prompt", required=True, help="Text prompt to generate images from.")
    parser.add_argument(
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model id (default: runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--outdir", default="outputs", help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=1, help="Total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to generate per batch call")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Image height (must be divisible by 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (must be divisible by 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic outputs")
    parser.add_argument("--device", type=str, default=None, help="Force device: 'cuda' or 'cpu' (auto-detect if omitted)")
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default=None, help="Precision to use. Defaults to fp16 on CUDA")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        saved = generate_images(
            prompt=args.prompt,
            model_id=args.model,
            hf_token=args.hf_token,
            outdir=args.outdir,
            num_images=args.num_images,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        logger.info("Completed generation. %d images saved.", len(saved))
        return 0
    except Exception as exc:
        logger.exception("Error occurred: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
