#!/usr/bin/env python3
"""
text2image.py

Command-line utility to generate images from text prompts using Hugging Face Diffusers.

Features:
- CLI with sensible defaults and input validation
- Device detection (CUDA/CPU) with memory-optimized loading
- Optional Hugging Face authentication token support
- Image saving with reproducible seeds
- Comprehensive logging and error handling

Usage example:
  export HF_TOKEN="<your_hf_token>"
  python text2image.py --prompt "A serene landscape with mountains at sunrise" --out ./output.png

Note: Stable Diffusion models typically require agreeing to model terms on Hugging Face and providing an access token
if the model requires it. Check the model card for details.

"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline
except Exception as e:  # pragma: no cover - defensive import handling
    raise ImportError(
        "diffusers is required to run this script. Install with `pip install diffusers`")


# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(_handler)


@dataclass
class GenerationConfig:
    prompt: str
    out_path: Path
    model_id: str = "runwayml/stable-diffusion-v1-5"
    auth_token: Optional[str] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    device: Optional[str] = None


def _validate_config(cfg: GenerationConfig) -> None:
    """Validate the generation config and raise ValueError for invalid values.

    Args:
        cfg: GenerationConfig to validate.

    Raises:
        ValueError: if any parameter is out of allowed range.
    """
    if not cfg.prompt or not isinstance(cfg.prompt, str):
        raise ValueError("Prompt must be a non-empty string.")

    if cfg.num_inference_steps <= 0 or cfg.num_inference_steps > 500:
        raise ValueError("num_inference_steps must be between 1 and 500.")

    if not (0.0 <= cfg.guidance_scale <= 50.0):
        raise ValueError("guidance_scale must be between 0.0 and 50.0.")

    if cfg.height % 8 != 0 or cfg.width % 8 != 0:
        raise ValueError("height and width must be multiples of 8.")

    if cfg.height <= 0 or cfg.width <= 0 or cfg.height > 2048 or cfg.width > 2048:
        raise ValueError("height and width must be >0 and <=2048.")

    if cfg.seed is not None and (cfg.seed < 0 or cfg.seed > 2**31 - 1):
        raise ValueError("seed must be between 0 and 2**31-1.")


def _get_device(preferred: Optional[str] = None) -> str:
    """Determine the best device to run generation on.

    Preference order: user-specified -> CUDA if available -> CPU
    """
    if preferred:
        # basic validation
        if preferred not in ("cpu", "cuda"):
            logger.warning("Unknown device '%s', falling back to automatic selection.", preferred)
        else:
            if preferred == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
            else:
                return preferred

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(model_id: str, device: str, auth_token: Optional[str] = None) -> StableDiffusionPipeline:
    """Load and return a StableDiffusionPipeline with memory optimizations applied.

    Args:
        model_id: Hugging Face model repo id.
        device: 'cuda' or 'cpu'.
        auth_token: optional HF token for private models.

    Returns:
        An initialized StableDiffusionPipeline.
    """
    logger.info("Loading model '%s' on device '%s'...", model_id, device)

    # Choose dtype for loading
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Use local cache and allow running with token
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            revision=None,
            safety_checker=None,  # explicit: we will not enforce automatic safety checker here; user choice
            use_auth_token=auth_token,
        )
    except TypeError:
        # older/newer diffusers versions have different param names; fall back
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, revision=None)

    # Move to device
    pipe = pipe.to(device)

    # Memory optimizations if CUDA
    if device == "cuda":
        try:
            # enable attention slicing to reduce memory
            pipe.enable_attention_slicing()
        except Exception:
            logger.debug("enable_attention_slicing not available on this diffusers version.")

        try:
            # enable memory efficient attention (xformers) if built
            pipe.enable_xformers_memory_efficient_attention()
            logger.debug("xFormers memory efficient attention enabled.")
        except Exception:
            logger.debug("xFormers not available; continuing without it.")

    return pipe


def generate_image(cfg: GenerationConfig) -> Path:
    """Generate an image from the given prompt and save it to out_path.

    Args:
        cfg: GenerationConfig containing generation parameters.

    Returns:
        Path to the generated image file.

    Raises:
        RuntimeError: if generation fails.
    """
    _validate_config(cfg)

    device = cfg.device or _get_device(None)
    pipe = load_pipeline(cfg.model_id, device, cfg.auth_token)

    # Set deterministic seed for reproducibility
    seed = cfg.seed if cfg.seed is not None else int(time.time())
    generator = torch.Generator(device=device).manual_seed(seed)
    logger.info("Using seed=%d", seed)

    logger.info("Generating image with prompt: %s", cfg.prompt)

    try:
        result = pipe(
            prompt=cfg.prompt,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
        )
    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        raise RuntimeError("Image generation failed") from exc

    # result.images is a list of PIL Images
    if not result or not hasattr(result, "images") or not result.images:
        raise RuntimeError("No images returned by the pipeline")

    image = result.images[0]
    out_path = cfg.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image
    try:
        image.save(out_path)
        logger.info("Saved image to %s", out_path)
    except Exception as exc:
        logger.exception("Failed to save image: %s", exc)
        raise

    return out_path


def parse_args(argv: Optional[list[str]] = None) -> GenerationConfig:
    """Parse CLI args into a GenerationConfig.

    Args:
        argv: Optional list of args for testing.

    Returns:
        GenerationConfig
    """
    parser = argparse.ArgumentParser(description="Generate images from text prompts using Diffusers")

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate the image from")
    parser.add_argument("--out", type=str, required=False, default="./output.png", help="Output image path")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (or set HF_TOKEN env var)")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (1-500)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8)")
    parser.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, choices=("cpu", "cuda"), default=None, help="Device to run on")

    args = parser.parse_args(argv)

    # HF token resolution: CLI -> env
    token = args.token or os.environ.get("HF_TOKEN")

    cfg = GenerationConfig(
        prompt=args.prompt,
        out_path=Path(args.out),
        model_id=args.model,
        auth_token=token,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        seed=args.seed,
        device=args.device,
    )

    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    """Main entrypoint for the CLI.

    Returns:
        Exit code (0 on success, non-zero on error)
    """
    try:
        cfg = parse_args(argv)
    except Exception as exc:
        logger.error("Argument parsing error: %s", exc)
        return 2

    try:
        out_path = generate_image(cfg)
        logger.info("Generation completed successfully: %s", out_path)
        return 0
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
