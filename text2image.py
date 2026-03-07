#!/usr/bin/env python3
"""
Text-to-Image script using Hugging Face Diffusers

This script provides a production-ready CLI to generate images from text prompts
using a Stable Diffusion pipeline from Hugging Face diffusers. It includes
sensible defaults, GPU/CPU handling, logging, input validation, reproducible
seeds, and memory/performance optimizations.

Usage example:
  export HF_TOKEN="<your_hf_token>"
  python text2image.py --prompt "A fantasy castle on a lake at sunrise" --outdir outputs/

For private models you must provide a Hugging Face token (env HF_TOKEN or --hf-token).

Security note: Be careful when using unrestricted prompts or sharing generated
images that might contain private or copyrighted content. The pipeline may also
flag NSFW content; check outputs before use.
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
from typing import List, Optional, Tuple

try:
    import torch
except Exception as exc:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "PyTorch must be installed to run this script. See requirements.txt"
    ) from exc

try:
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import check_min_version
except Exception as exc:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "diffusers must be installed to run this script. See requirements.txt"
    ) from exc

from PIL import Image
import numpy as np

# Configure module-level logger
logger = logging.getLogger("text2image")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class GenerationConfig:
    """Configuration for image generation.

    Attributes:
        model_id: HF model id (e.g., 'runwayml/stable-diffusion-v1-5')
        prompt: prompt text
        negative_prompt: negative prompt text
        height: image height
        width: image width
        num_inference_steps: number of denoising steps
        guidance_scale: classifier-free guidance scale
        num_images_per_prompt: number of images to generate per prompt
        seed: RNG seed for reproducibility (None means random)
        device: torch device string
        hf_token: Hugging Face token for private models or rate-limited access
        output_dir: directory to save images
        use_fp16: whether to use mixed precision on CUDA
    """

    model_id: str
    prompt: str
    negative_prompt: Optional[str]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    num_images_per_prompt: int
    seed: Optional[int]
    device: str
    hf_token: Optional[str]
    output_dir: str
    use_fp16: bool


def validate_config(cfg: GenerationConfig) -> None:
    """Validate user-provided configuration and raise ValueError for invalid inputs."""

    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    if cfg.height <= 0 or cfg.width <= 0:
        raise ValueError("Image height and width must be positive integers")

    if cfg.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")

    if cfg.guidance_scale < 1.0 or cfg.guidance_scale > 30.0:
        logger.warning(
            "guidance_scale is outside common ranges (1.0-30.0). Unexpected results may occur."
        )

    if cfg.num_images_per_prompt <= 0 or cfg.num_images_per_prompt > 8:
        raise ValueError("num_images_per_prompt must be between 1 and 8 (inclusive)")

    outdir = Path(cfg.output_dir)
    if not outdir.exists():
        logger.info("Output directory does not exist, creating: %s", str(outdir))
        outdir.mkdir(parents=True, exist_ok=True)


def seed_all(seed: Optional[int]) -> int:
    """Set seeds for Python, NumPy and Torch to make generation reproducible.

    Returns the effective seed used.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    logger.info("Using seed: %d", seed)

    try:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        logger.exception("Failed to set some RNG seeds")

    return seed


def get_device(prefer_cuda: bool = True) -> str:
    """Return a device string: 'cuda' if available and preferred else 'cpu'."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_output_filename(prompt: str, idx: int, seed: int) -> str:
    """Create a deterministic but human-readable filename for the output image."""
    # compact hash of prompt to avoid long filenames
    compact = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"sd_{compact}_s{seed}_i{idx}_{timestamp}.png"


def load_pipeline(
    model_id: str, device: str, hf_token: Optional[str], use_fp16: bool
) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline with memory optimizations.

    - Attempts to use float16 on CUDA.
    - Enables attention slicing and VAE tiling to reduce peak memory.
    - Does not enable full device_map auto-placement here to keep deployment
      predictable; for multi-GPU or very large models use accelerate/optimum.
    """

    # Choose dtype
    if device.startswith("cuda") and use_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logger.info("Loading pipeline: %s (device=%s, dtype=%s)", model_id, device, torch_dtype)

    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision=None,
            torch_dtype=torch_dtype,
            use_auth_token=hf_token,
        )
    except TypeError:
        # Older versions of diffusers used a different argument name
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

    # Move to device
    pipeline = pipeline.to(device)

    # Memory-savers
    try:
        pipeline.enable_attention_slicing()
        logger.debug("Enabled attention slicing")
    except Exception:
        logger.debug("Attention slicing not available for this pipeline")

    try:
        pipeline.enable_vae_slicing()
        logger.debug("Enabled VAE slicing")
    except Exception:
        logger.debug("VAE slicing not available for this pipeline")

    # xformers is optional; enable if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.debug("Enabled xformers memory efficient attention")
    except Exception:
        logger.debug("xFormers not available or failed to enable")

    # Note: the safety checker may exist; we leave it enabled by default for safety

    return pipeline


def generate_images(cfg: GenerationConfig) -> List[Path]:
    """Generate images for a given configuration and return saved file paths.

    This function handles pipeline invocation and IO. It attempts to catch OOM
    errors and provides informative messages for remediation.
    """
    validate_config(cfg)
    seed = seed_all(cfg.seed)

    saved_paths: List[Path] = []

    pipeline = load_pipeline(cfg.model_id, cfg.device, cfg.hf_token, cfg.use_fp16)

    # For reproducibility, set generator
    generator = torch.Generator(device=cfg.device)
    generator = generator.manual_seed(seed)

    # Prepare call kwargs
    run_kwargs = dict(
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt or None,
        height=cfg.height,
        width=cfg.width,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        num_images_per_prompt=cfg.num_images_per_prompt,
        generator=generator,
    )

    logger.info("Running pipeline with args: num_images=%d, steps=%d, guidance=%.2f",
                cfg.num_images_per_prompt, cfg.num_inference_steps, cfg.guidance_scale)

    try:
        outputs = pipeline(**run_kwargs)
    except RuntimeError as exc:
        # Common cause: OOM on GPU
        logger.exception("Runtime error while running pipeline: %s", str(exc))
        if 'out of memory' in str(exc).lower() and cfg.device.startswith('cuda'):
            raise RuntimeError(
                'CUDA out of memory. Try a smaller image size, fewer inference steps, or run on CPU.'
            ) from exc
        raise

    images: List[Image.Image] = []

    # The output object may provide 'images' attribute
    if hasattr(outputs, "images"):
        images = outputs.images
    elif isinstance(outputs, list):
        # Some pipelines may return list directly
        images = outputs
    else:
        raise RuntimeError("Unexpected pipeline output format")

    outdir = Path(cfg.output_dir)
    for idx, img in enumerate(images):
        filename = _make_output_filename(cfg.prompt, idx, seed)
        path = outdir / filename
        try:
            # Save with deterministic parameters
            img.save(path, format="PNG")
            saved_paths.append(path)
            logger.info("Saved image: %s", str(path))
        except Exception:
            logger.exception("Failed to save image to %s", str(path))

    return saved_paths


def parse_args(argv: Optional[List[str]] = None) -> GenerationConfig:
    """Parse command-line arguments and return a GenerationConfig."""
    parser = argparse.ArgumentParser(
        description="Generate images from text using Hugging Face diffusers"
    )

    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face model id for a Stable Diffusion checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate")
    parser.add_argument("--negative-prompt", type=str, default=None,
                        help="Negative prompt to discourage content")
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num-images", type=int, default=1,
                        help="Number of images to generate per prompt (1-8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Directory to write generated images")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token; can be set via HF_TOKEN env var")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable fp16 even if CUDA is available")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cpu' or 'cuda'); default auto-detected")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")

    args = parser.parse_args(argv)

    if args.quiet:
        logger.setLevel(logging.WARNING)

    device = args.device or get_device(prefer_cuda=True)

    cfg = GenerationConfig(
        model_id=args.model_id,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        seed=args.seed,
        device=device,
        hf_token=args.hf_token,
        output_dir=args.outdir,
        use_fp16=args.fp16,
    )

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    """Main entrypoint for CLI. Returns 0 on success, non-zero otherwise."""
    try:
        cfg = parse_args(argv)
        start = time.time()
        saved = generate_images(cfg)
        elapsed = time.time() - start
        logger.info("Generation completed in %.2f seconds", elapsed)
        logger.info("Saved %d images to %s", len(saved), cfg.output_dir)
        return 0
    except Exception as exc:
        logger.exception("Failed to generate images: %s", str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
