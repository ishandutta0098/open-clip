#!/usr/bin/env python3
"""
text2image.py

Command-line utility to generate images from text prompts using Hugging Face Diffusers.

Features:
- Loads a Stable Diffusion pipeline from Hugging Face Hub
- Supports GPU (automatic if available) with mixed precision
- Deterministic generation via seed
- Input validation, error handling and logging
- Configurable generation parameters (steps, guidance_scale, size, negative prompt)
- Secure token handling via environment variable HUGGINGFACE_HUB_TOKEN

Usage examples:
  python text2image.py --prompt "A fantasy landscape, vivid colors" --output ./out.png
  python text2image.py --prompt_file prompts.txt --batch_size 2 --num_images_per_prompt 3

Notes:
- Model may require a Hugging Face token for access (set HUGGINGFACE_HUB_TOKEN env var)
- For best performance on GPU, ensure compatible torch + CUDA are installed

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging as diffusers_logging

# Configure logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(handler)

diffusers_logging.set_verbosity_error()


@dataclass
class GenerationConfig:
    model_id: str
    hf_token: Optional[str]
    device: str
    torch_dtype: torch.dtype
    prompt: str
    negative_prompt: Optional[str]
    guidance_scale: float
    num_inference_steps: int
    height: int
    width: int
    seed: Optional[int]
    num_images_per_prompt: int
    batch_size: int
    output_dir: Path


def _validate_prompt(prompt: str) -> None:
    """Validate prompt string.

    Raises ValueError for invalid prompts.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    s = prompt.strip()
    if not s:
        raise ValueError("prompt must not be empty")
    if len(s) > 2000:
        # protect downstream from extremely long prompts
        raise ValueError("prompt is too long (limit 2000 chars)")


def _sanitize_model_id(model_id: str) -> str:
    """Basic sanitization for model id to avoid accidental shell injections or path mistakes.

    Note: This is intentionally simple — model ids are user-provided and validated by HF.
    """
    return model_id.strip()


def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    """Choose device and dtype based on availability.

    Returns device string and torch dtype to use with from_pretrained.
    """
    if torch.cuda.is_available():
        logger.info("CUDA available. Using GPU with float16 for performance.")
        return "cuda", torch.float16
    else:
        # CPU fallback
        logger.info("CUDA not available. Using CPU with float32 (slower). Consider installing GPU-enabled PyTorch.")
        return "cpu", torch.float32


def load_pipeline(model_id: str, hf_token: Optional[str], device: str, torch_dtype: torch.dtype) -> StableDiffusionPipeline:
    """Load and return a StableDiffusionPipeline.

    Args:
        model_id: HF model identifier
        hf_token: optional HF token for gated models
        device: device string ("cuda" or "cpu")
        torch_dtype: torch dtype to use when loading the model

    Returns:
        Initialized StableDiffusionPipeline moved to specified device.
    """
    logger.info("Loading model: %s", model_id)

    # Basic protection against accidental model id mistakes
    model_id = _sanitize_model_id(model_id)

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_auth_token=hf_token,
        )
    except Exception as exc:  # pragma: no cover - model download errors
        logger.exception("Failed to load model from Hugging Face. Check model id and HUGGINGFACE_HUB_TOKEN if required.")
        raise

    # Move the pipeline to chosen device
    try:
        pipe.to(device)
    except Exception:
        logger.exception("Failed to move pipeline to device: %s", device)
        raise

    # Optionally enable xformers or memory efficient attention if available
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            # Best-effort: call if available
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory efficient attention (if supported).")
    except Exception:
        # Non-fatal: just log and continue
        logger.debug("xFormers not available or failed to enable.")

    return pipe


def _generate_single(
    pipe: StableDiffusionPipeline,
    config: GenerationConfig,
    gen: Optional[torch.Generator] = None,
) -> Image.Image:
    """Run the pipeline once for the given prompt and return a PIL Image.

    This function isolates the pipeline call to make error handling easier.
    """
    extra_kwargs = {}
    if gen is not None:
        extra_kwargs["generator"] = gen

    # Keep sample deterministic if generator provided
    try:
        result = pipe(
            config.prompt,
            negative_prompt=config.negative_prompt,
            height=config.height,
            width=config.width,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            num_images_per_prompt=config.num_images_per_prompt,
            **extra_kwargs,
        )
    except Exception:
        logger.exception("Pipeline failed during generation")
        raise

    images = result.images
    if not images or len(images) == 0:
        raise RuntimeError("No images returned by the pipeline")

    # Return first for now; caller can handle multiple
    return images[0]


def generate_images(config: GenerationConfig) -> List[Path]:
    """Generate images according to config and return list of saved file paths.

    This function supports batch generation. It writes PNG files into output_dir.
    """
    logger.info("Generating images to directory: %s", config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(config.model_id, config.hf_token, config.device, config.torch_dtype)

    # Prepare deterministic generator if requested
    generator = None
    if config.seed is not None:
        # When using CPU, generator device should be "cpu"; when using GPU, set to "cuda" device.
        generator = torch.Generator(device=config.device)
        generator.manual_seed(config.seed)
        logger.info("Using deterministic seed: %d", config.seed)

    saved_paths: List[Path] = []

    # We'll generate in batches: for total num images requested per prompt, stepping by batch_size
    total_per_prompt = config.num_images_per_prompt
    produced = 0
    while produced < total_per_prompt:
        current_batch = min(config.batch_size, total_per_prompt - produced)

        # For generator determinism across batches, create a new generator per call with offset seed
        batch_generator = None
        if generator is not None:
            # Clone and advance the generator deterministically
            # Create new generator using seed + produced offset to vary while maintaining determinism
            batch_seed = config.seed + produced
            batch_generator = torch.Generator(device=config.device).manual_seed(batch_seed)

        for i in range(current_batch):
            img = _generate_single(pipe, config, gen=batch_generator)

            # Compose filename
            safe_prompt = (
                "_".join(config.prompt.strip().split())[:80].replace("/", "-").replace("\\", "-")
            )
            timestamp = int(time.time())
            fname = f"sd_{timestamp}_{uuid.uuid4().hex[:8]}_{produced + i}.png"
            out_path = config.output_dir / fname

            # Convert to RGB and save as PNG
            try:
                img = img.convert("RGB")
                img.save(out_path, format="PNG")
                saved_paths.append(out_path)
                logger.info("Saved image: %s", out_path)
            except Exception:
                logger.exception("Failed to save image to disk: %s", out_path)
                raise

        produced += current_batch

    # Clear VRAM if using GPU
    if config.device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to empty CUDA cache (non-fatal)")

    return saved_paths


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to image generation using Hugging Face Diffusers")

    parser.add_argument("--prompt", type=str, help="Prompt text to generate an image for")
    parser.add_argument("--prompt_file", type=str, help="Path to a file with prompts (one per line). If set, --prompt is ignored.")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to write generated images")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of denoising steps (more = higher quality and slower)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Image height (pix)")
    parser.add_argument("--width", type=int, default=512, help="Image width (pix)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for deterministic results")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to avoid elements")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="How many images to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per pipeline call")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (optional). Prefer HUGGINGFACE_HUB_TOKEN env var.)")

    return parser.parse_args(argv)


def _load_prompts_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh.readlines() if line.strip()]
    if not lines:
        raise ValueError("No prompts found in the prompt file")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Choose token: prefer supplied arg, then env var
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_HOME_TOKEN")
    if hf_token is None:
        logger.info("No Hugging Face token provided via --hf_token or HUGGINGFACE_HUB_TOKEN. Some models may require authentication.")

    # Device selection
    device, torch_dtype = get_device_and_dtype()

    # Determine prompts
    prompts: List[str]
    if args.prompt_file:
        prompts = _load_prompts_from_file(args.prompt_file)
    else:
        if not args.prompt:
            logger.error("Either --prompt or --prompt_file must be provided")
            return 2
        prompts = [args.prompt]

    output_dir = Path(args.output_dir)

    # Validate basic numeric params
    if args.num_inference_steps <= 0 or args.num_inference_steps > 250:
        raise ValueError("num_inference_steps must be in 1..250")
    if args.guidance_scale < 1.0 or args.guidance_scale > 30.0:
        raise ValueError("guidance_scale seems out of normal range")
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height and width must be divisible by 8 for Stable Diffusion")
    if args.num_images_per_prompt < 1 or args.num_images_per_prompt > 16:
        raise ValueError("num_images_per_prompt must be between 1 and 16")
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError("batch_size must be between 1 and 8")

    config = GenerationConfig(
        model_id=args.model_id,
        hf_token=hf_token,
        device=device,
        torch_dtype=torch_dtype,
        prompt="",
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        output_dir=output_dir,
    )

    all_saved: List[Path] = []

    for p in prompts:
        try:
            _validate_prompt(p)
        except ValueError as exc:
            logger.error("Invalid prompt: %s -> %s", p, exc)
            continue

        config = GenerationConfig(
            **{**config.__dict__, "prompt": p},
        )

        try:
            saved = generate_images(config)
            all_saved.extend(saved)
        except Exception as exc:
            logger.error("Generation failed for prompt '%s': %s", p, exc)

    if not all_saved:
        logger.error("No images produced")
        return 1

    logger.info("Done. Generated %d images. First image at: %s", len(all_saved), all_saved[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
