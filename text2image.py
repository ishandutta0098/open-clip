#!/usr/bin/env python3
"""
text2image.py

Production-ready CLI and library to generate images from text using HuggingFace Diffusers.

Features:
- CLI + callable API
- Device auto-detection (CUDA/CPU)
- Seeded deterministic generation
- Input validation and robust error handling
- Optional use of half precision on CUDA
- Optional Hugging Face token from env or CLI
- Logging and simple progress reporting

Usage (CLI):
    python text2image.py --prompt "A fantasy castle at sunset" --out_dir ./outputs --num_inference_steps 30 --guidance_scale 7.5

API example:
    from text2image import generate_image
    img_path = generate_image("A cat wearing a suit", out_dir="./outs")

Requirements: see requirements.txt

"""
from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Import inside try/except so we can provide actionable error messages
try:
    from diffusers import StableDiffusionPipeline
except Exception as exc:  # pragma: no cover - runtime import failure
    raise ImportError(
        "diffusers is required. Install with: pip install diffusers==0.19.0 "
        "transformers==4.30.2 accelerate==0.19.0 torch==2.0.1 safetensors==0.3.2 pillow==9.5.0"
    ) from exc


# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
)
logger.addHandler(handler)


@dataclass
class GenerationConfig:
    prompt: str
    out_dir: Path = Path("./outputs")
    filename: Optional[str] = None
    model_id: str = "runwayml/stable-diffusion-v1-5"
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    device: Optional[str] = None  # 'cuda' | 'cpu'. If None, auto-detect
    use_fp16: bool = True
    hf_token: Optional[str] = None


def _validate_config(cfg: GenerationConfig) -> None:
    """Validate the generation configuration and raise ValueError for invalid values."""
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("`prompt` is required and must be a non-empty string")

    if cfg.width <= 0 or cfg.height <= 0:
        raise ValueError("`width` and `height` must be positive integers")

    if not (0.0 <= cfg.guidance_scale <= 50.0):
        raise ValueError("`guidance_scale` must be between 0.0 and 50.0")

    if not (1 <= cfg.num_inference_steps <= 500):
        raise ValueError("`num_inference_steps` must be between 1 and 500")

    if cfg.seed is not None and (cfg.seed < 0 or cfg.seed > 2 ** 31 - 1):
        raise ValueError("`seed` must be between 0 and 2**31-1")


def _get_device(preferred: Optional[str] = None) -> str:
    """Return 'cuda' if available and requested, otherwise 'cpu'."""
    if preferred:
        preferred_low = preferred.lower()
        if preferred_low == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("Requested CUDA but CUDA is not available; falling back to CPU")
            return "cpu"
        if preferred_low == "cpu":
            return "cpu"
        logger.warning("Unrecognized device '%s', auto-detecting instead", preferred)

    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_optimize_pipeline(pipe: StableDiffusionPipeline, device: str, use_fp16: bool) -> None:
    """Apply optional optimizations to the pipeline (xformers, fp16) when supported."""
    if device == "cuda":
        try:
            # Use fp16 if requested and supported
            if use_fp16:
                pipe.to(torch.float16)
            # Enable xformers memory efficient attention if available
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe.enable_xformers_memory_efficient_attention()
                logger.debug("Enabled xformers memory efficient attention")
        except Exception as exc:  # pragma: no cover - runtime optimization issues
            logger.warning("Could not enable some optimizations: %s", exc)


def generate_image(cfg: GenerationConfig) -> Path:
    """
    Generate an image from text using HuggingFace Diffusers Stable Diffusion pipeline.

    Args:
        cfg: GenerationConfig containing settings.

    Returns:
        Path to the generated image file.

    Raises:
        ValueError: for invalid configuration
        RuntimeError: when generation fails
    """
    _validate_config(cfg)

    device = _get_device(cfg.device)
    logger.info("Using device: %s", device)

    out_dir = _ensure_out_dir(cfg.out_dir)

    # Resolve filename and ensure it does not overwrite by default — append index if exists
    filename = cfg.filename or "image.png"
    out_path = out_dir / filename
    if out_path.exists():
        base = out_path.stem
        suffix = out_path.suffix or ".png"
        # find a non-colliding name
        i = 1
        while True:
            candidate = out_dir / f"{base}_{i}{suffix}"
            if not candidate.exists():
                out_path = candidate
                break
            i += 1

    # Convert to correct dtype
    torch_dtype = torch.float16 if (device == "cuda" and cfg.use_fp16) else torch.float32

    # Load pipeline
    logger.info("Loading model '%s' ... (this may take a while on first run)", cfg.model_id)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=torch_dtype,
            use_auth_token=cfg.hf_token,
        )
    except Exception as exc:  # pragma: no cover - runtime model loading
        raise RuntimeError(
            f"Failed to load model '{cfg.model_id}'. Ensure network access and valid model id/token: {exc}"
        ) from exc

    # Move to device
    try:
        pipe = pipe.to(device)
    except Exception:
        # Some diffusers versions expect pipe.to(device) with dtype already set
        pipe.to(device)

    _maybe_optimize_pipeline(pipe, device, cfg.use_fp16)

    # Prepare generator if seed provided
    generator = None
    if cfg.seed is not None:
        try:
            gen_device = device if device == "cpu" else "cuda"
            generator = torch.Generator(device=gen_device).manual_seed(int(cfg.seed))
        except Exception as exc:  # pragma: no cover - seed errors
            logger.warning("Could not create deterministic generator: %s", exc)
            generator = None

    logger.info(
        "Generating image with steps=%s guidance=%s size=%sx%s",
        cfg.num_inference_steps,
        cfg.guidance_scale,
        cfg.width,
        cfg.height,
    )

    try:
        # Some pipelines support height & width parameters
        output = pipe(
            prompt=cfg.prompt,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
        )
    except TypeError:
        # Fallback for older diffusers that don't accept height/width in call
        logger.debug("Falling back to pipeline without height/width in call")
        output = pipe(
            [cfg.prompt],
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
        )

    if not hasattr(output, "images"):
        # Some diffusers return a tuple
        images = output[0] if isinstance(output, tuple) and len(output) > 0 else None
    else:
        images = output.images

    if not images:
        raise RuntimeError("Model did not return any images")

    # Take first image
    image = images[0]
    if not isinstance(image, Image.Image):
        # Convert tensor to PIL if necessary
        try:
            image = Image.fromarray(image)
        except Exception as exc:  # pragma: no cover - conversion failure
            raise RuntimeError(f"Could not convert model output to image: {exc}") from exc

    # Save result
    try:
        image.save(out_path, format="PNG")
        logger.info("Saved image to %s", out_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save image to {out_path}: {exc}") from exc

    return out_path


def _parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate images from text using Diffusers (Stable Diffusion)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the image generator")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save generated images")
    parser.add_argument("--filename", type=str, default=None, help="Optional filename for the generated image (defaults to image.png with collision avoidance)")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face Diffusers model id")
    parser.add_argument("--width", type=int, default=512, help="Output image width (px)")
    parser.add_argument("--height", type=int, default=512, help="Output image height (px)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Device to run on (auto-detected by default)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16 even if CUDA is available")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models (or set HF_TOKEN env var)")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    cfg = GenerationConfig(
        prompt=args.prompt,
        out_dir=Path(args.out_dir),
        filename=args.filename,
        model_id=args.model_id,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        device=args.device,
        use_fp16=not args.no_fp16,
        hf_token=hf_token,
    )

    return cfg


def main() -> None:
    cfg = _parse_args()

    try:
        out_path = generate_image(cfg)
        logger.info("Image generation complete: %s", out_path)
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
