#!/usr/bin/env python3
"""
text2image.py

Command-line tool to generate images from text prompts using Hugging Face Diffusers.

Features:
- Flexible model selection (model id on HF Hub)
- Device autodetection (GPU/CPU) with fp16 on CUDA when available
- Input validation and parameter bounds checking
- Robust error handling and logging
- Optional Hugging Face auth token via env var or CLI for gated models
- Output image saved with timestamped filename or custom path

Usage:
    python text2image.py --prompt "a fantasy landscape" --model "runwayml/stable-diffusion-v1-5" --out output.png

Security and operational notes:
- Use a model you have the rights to use. If a model is gated, provide HF token via HUGGINGFACE_TOKEN env var or --hf_token
- Be mindful of model card and safety checker (some pipelines may skip safety checks depending on model and diffusers version)
- Running on CPU will be significantly slower; prefer CUDA-enabled device with sufficient VRAM.

"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from PIL import Image

try:
    import torch
    from diffusers import StableDiffusionPipeline
    from transformers import logging as transformers_logging
except Exception as e:  # pragma: no cover - runtime import errors should be surfaced
    raise RuntimeError(
        "Missing required libraries. Ensure dependencies are installed (see requirements.txt). "
        "Import failed: %s" % (e,)
    )

# Silence overly verbose libraries while keeping our logging level informative
transformers_logging.set_verbosity_error()

# Configure module-level logger
logger = logging.getLogger("text2image")


@dataclass
class GenerationConfig:
    prompt: str
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output: Optional[str] = None
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    hf_token: Optional[str] = None
    device: str = "auto"
    torch_dtype: Optional[torch.dtype] = None


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging.

    Args:
        level: Logging level.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)


def detect_device(preferred: str = "auto") -> Tuple[str, Optional[torch.dtype]]:
    """Detect best device and torch dtype to use.

    Returns:
        Tuple of device string and torch dtype (if applicable)
    """
    if preferred and preferred.lower() in ("cpu", "cuda", "auto"):
        pref = preferred.lower()
    else:
        pref = "auto"

    if pref == "cpu":
        logger.debug("Force using CPU")
        return "cpu", torch.float32

    if torch.cuda.is_available():
        # Use fp16 on CUDA to reduce memory usage and improve throughput
        logger.debug("CUDA is available; selecting cuda with torch.float16")
        return "cuda", torch.float16

    logger.debug("CUDA not available; selecting cpu")
    return "cpu", torch.float32


def validate_config(cfg: GenerationConfig) -> None:
    """Validate generation configuration and provide helpful errors.

    Args:
        cfg: GenerationConfig instance

    Raises:
        ValueError: if validation fails
    """
    if not cfg.prompt or not isinstance(cfg.prompt, str):
        raise ValueError("Prompt must be a non-empty string")

    if not (64 <= cfg.height <= 2048):
        raise ValueError("Height must be between 64 and 2048")

    if not (64 <= cfg.width <= 2048):
        raise ValueError("Width must be between 64 and 2048")

    if not (0.0 <= cfg.guidance_scale <= 50.0):
        raise ValueError("guidance_scale must be between 0.0 and 50.0")

    if not (1 <= cfg.num_inference_steps <= 200):
        raise ValueError("num_inference_steps must be between 1 and 200")

    if cfg.seed is not None and not (0 <= cfg.seed <= 2 ** 32 - 1):
        raise ValueError("seed must be a 32-bit unsigned integer")


def load_pipeline(model_id: str, device: str, dtype: Optional[torch.dtype], hf_token: Optional[str]) -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline from Hugging Face.

    Args:
        model_id: model repo id on HF hub
        device: 'cuda' or 'cpu'
        dtype: torch dtype to use on device (torch.float16 for cuda recommended)
        hf_token: optional token for accessing gated models

    Returns:
        An initialized StableDiffusionPipeline

    Raises:
        RuntimeError: If pipeline can't be created or moved to device
    """
    logger.info("Loading pipeline for model '%s' on device '%s'", model_id, device)

    try:
        # Pass torch_dtype to pipeline for memory optimization (if supported)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_auth_token=hf_token,
        )
    except TypeError:
        # Older/newer versions of diffusers may not accept use_auth_token kwarg; try without
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # Disable NSFW filter or leave as-is? Keep whatever the model provides. In new diffusers,
    # safety_checker may be None for some community models. We do not remove checks here.

    try:
        if device == "cuda":
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cpu")
    except Exception as e:
        raise RuntimeError("Failed to move pipeline to device: %s" % (e,))

    # Optimize for inference if available (enable attention slicing to reduce peak memory)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        # Some pipeline classes might not have this method depending on diffusers version
        logger.debug("enable_attention_slicing not available for this pipeline version")

    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    cfg: GenerationConfig,
) -> Image.Image:
    """Generate an image from prompt using the provided pipeline.

    Args:
        pipe: Initialized StableDiffusionPipeline
        cfg: GenerationConfig

    Returns:
        PIL Image
    """
    logger.info("Generating image with prompt: %s", cfg.prompt)

    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(cfg.seed))

    # Use autocast for performance if using fp16 on CUDA
    use_autocast = getattr(torch, "cuda", None) is not None and pipe.device.type == "cuda" and cfg.torch_dtype == torch.float16

    try:
        if use_autocast:
            with torch.autocast(device_type="cuda"):
                output = pipe(
                    prompt=cfg.prompt,
                    height=cfg.height,
                    width=cfg.width,
                    guidance_scale=cfg.guidance_scale,
                    num_inference_steps=cfg.num_inference_steps,
                    generator=generator,
                )
        else:
            output = pipe(
                prompt=cfg.prompt,
                height=cfg.height,
                width=cfg.width,
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=cfg.num_inference_steps,
                generator=generator,
            )
    except Exception as e:
        logger.exception("Image generation failed: %s", e)
        raise

    # output.images is typically a list of PIL images (or numpy arrays)
    images = getattr(output, "images", None)
    if not images:
        raise RuntimeError("Pipeline did not return images")

    logger.info("Image generation complete")
    return images[0]


def save_image(img: Image.Image, out_path: Optional[str]) -> str:
    """Save PIL Image to disk with safe filename handling.

    Args:
        img: PIL Image
        out_path: Optional output filename. If None, a timestamped filename is created.

    Returns:
        The path where the image was saved.
    """
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        filename = out_path
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"sd_out_{timestamp}.png"

    # Ensure we don't overwrite an existing file accidentally
    base, ext = os.path.splitext(filename)
    if not ext:
        filename = filename + ".png"
    else:
        # Only allow common image exts
        if ext.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            logger.warning("Unrecognized extension '%s', using .png instead", ext)
            filename = base + ".png"

    # Avoid race condition on create: if exists, append counter
    final = filename
    counter = 1
    while os.path.exists(final):
        final = f"{base}_{counter}{os.path.splitext(final)[1]}"
        counter += 1

    img.save(final)
    logger.info("Saved image to %s", final)
    return final


def parse_args(argv: Optional[list] = None) -> GenerationConfig:
    """Parse CLI args into GenerationConfig.

    Args:
        argv: Optional list of args (for testing). Defaults to sys.argv.

    Returns:
        GenerationConfig
    """
    parser = argparse.ArgumentParser(description="Generate images from text using Hugging Face Diffusers")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate an image for")
    parser.add_argument("--model", dest="model_id", default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id (default: runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--out", dest="output", default=None, help="Output filename (png/jpg). If omitted, a timestamped file will be created")
    parser.add_argument("--height", type=int, default=512, help="Output image height in pixels (default: 512)")
    parser.add_argument("--width", type=int, default=512, help="Output image width in pixels (default: 512)")
    parser.add_argument("--guidance", dest="guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face auth token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    # Prefer CLI token, fall back to env var
    token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN")

    device, dtype = detect_device(args.device)

    cfg = GenerationConfig(
        prompt=args.prompt,
        model_id=args.model_id,
        output=args.output,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        hf_token=token,
        device=device,
        torch_dtype=dtype,
    )

    if args.debug:
        configure_logging(logging.DEBUG)

    return cfg


def main(argv: Optional[list] = None) -> int:
    """Main entrypoint for the CLI.

    Returns:
        exit code (0=success)
    """
    configure_logging()

    try:
        cfg = parse_args(argv)
        logger.debug("Parsed configuration: %s", cfg)
        validate_config(cfg)

        # Load pipeline
        pipe = load_pipeline(cfg.model_id, cfg.device, cfg.torch_dtype, cfg.hf_token)

        # Generate
        img = generate_image(pipe, cfg)

        # Save
        saved_path = save_image(img, cfg.output)
        logger.info("Done. Image saved to: %s", saved_path)
        return 0

    except Exception as e:
        logger.error("Error: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
