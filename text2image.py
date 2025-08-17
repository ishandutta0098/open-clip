#!/usr/bin/env python3
"""Text-to-image generation utility using HuggingFace diffusers.

This script provides a robust, production-ready CLI to generate images from
text prompts using a Stable Diffusion model from HuggingFace Diffusers. It
supports CUDA-enabled GPUs, low-VRAM configurations, safe defaults, and
reproducible results via seeding.

Design goals:
- Simple CLI with sane defaults and reasonable configurability
- Safe by default (safety checker enabled) with an opt-out flag (not recommended)
- Progressive fallback to CPU when CUDA is unavailable
- Proper error handling, logging, and input validation
- Lightweight caching by reusing the loaded pipeline within a single process

Usage example:
  python text2image.py --model-id runwayml/stable-diffusion-v1-5 \
    --prompt "a futuristic city skyline at sunset" --output out.png --steps 50 --width 768 --height 512
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


logger = logging.getLogger(__name__)  # module-level logger


def _configure_logging() -> None:
    """Configure the module-level logging.

    Uses a simple, readable format suitable for production environments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def _load_pipeline(
    model_id: str,
    device: str,
    *,
    torch_dtype: torch.dtype = torch.float16,
    safety_checker: bool = True,
    low_vram: bool = False,
) -> StableDiffusionPipeline:
    """Load and configure the Stable Diffusion pipeline.

    Args:
      model_id: HuggingFace model identifier to preload.
      device: Target device ('cpu' or 'cuda').
      torch_dtype: Torch data type for tensor computations.
      safety_checker: Whether to enable the built-in safety checker.
      low_vram: Enable optimizations for low VRAM GPUs when supported.

    Returns:
      A ready-to-use StableDiffusionPipeline instance bound to the target device.
    """
    logger.info("Loading model '%s' on device '%s'", model_id, device)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

    if not safety_checker:
        # Disable safety checker if explicitly requested
        pipe.safety_checker = None
        pipe.feature_extractor = None  # type: ignore
        logger.warning(
            "Safety checker disabled. Generated content may violate usage policies."
        )

    if device == "cuda":
        try:
            # Memory optimization for CUDA GPUs
            pipe.enable_attention_slicing()
            if low_vram:
                pipe.enable_sequential_cpu_offload()
        except Exception as exc:  # pragma: no cover - optional feature
            logger.debug("Optional memory optimizations unavailable: %s", exc)
    pipe = pipe.to(device)
    logger.info("Model loaded and moved to %s", device)
    return pipe


def _generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """Generate an image from the given prompt using the provided pipeline.

    Args:
      pipe: Preloaded StableDiffusionPipeline instance.
      prompt: Text prompt describing the desired image.
      height, width: Output image dimensions (in pixels).
      num_inference_steps: Number of denoising steps to perform.
      guidance_scale: CFG scale; higher values push the image closer to the prompt.
      negative_prompt: Optional negative prompt to steer generation away from undesired features.
      seed: Optional seed for reproducibility.

    Returns:
      A PIL.Image.Image object containing the generated image.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        logger.info("Using provided seed: %d", seed)

    logger.info(
        "Generating image: prompt='%s', steps=%d, width=%d, height=%d, guidance=%.2f",
        prompt,
        num_inference_steps,
        width,
        height,
        guidance_scale,
    )

    # Call the diffusion pipeline to generate the image
    result = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
    )

    return result.images[0]


def _save_image(image: Image.Image, output_path: str) -> None:
    """Persist the generated image to disk, ensuring the target directory exists."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    logger.info("Saved generated image to '%s'", output_path)


def _validate_args(args: argparse.Namespace) -> None:
    """Basic validation of CLI arguments to provide early failure feedback."""
    if args.output is None or not isinstance(args.output, str) or not args.output:
        raise ValueError('Output path must be a non-empty string')
    if args.steps <= 0:
        raise ValueError('steps must be a positive integer')
    if args.width <= 0 or args.height <= 0:
        raise ValueError('width and height must be positive integers')


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(
        prog='text2image',
        description='Generate an image from a text prompt using HuggingFace diffusers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model-id', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='HuggingFace model id to use for generation')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for image generation')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Path to save the generated image')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--width', type=int, default=512,
                        help='Output image width in pixels')
    parser.add_argument('--height', type=int, default=512,
                        help='Output image height in pixels')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                        help='CFG scale for prompt conditioning')
    parser.add_argument('--negative-prompt', type=str, default=None,
                        help='Optional negative prompt to steer generation away from undesired aspects')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional seed for reproducibility')
    parser.add_argument('--disable-safety-check', action='store_true',
                        help='Disable the safety checker (not recommended)')
    parser.add_argument('--low-vram', action='store_true',
                        help='Enable low VRAM optimizations (may reduce quality or speed)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                        help='Override device: cpu or cuda. If not provided, auto-detects.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate inputs without performing generation')

    args = parser.parse_args()

    # Basic CLI validation
    _validate_args(args)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dry_run:
        logger.info(
            'Dry-run: model=%s, device=%s, prompt=%s, output=%s',
            args.model_id, device, args.prompt, args.output,
        )
        logger.info('steps=%d, width=%d, height=%d, guidance=%.2f', args.steps, args.width, args.height, args.guidance_scale)
        return

    try:
        torch_dtype = torch.float16 if device == 'cuda' else torch.float32
        pipeline = _load_pipeline(
            model_id=args.model_id,
            device=device,
            torch_dtype=torch_dtype,
            safety_checker=not args.disable_safety_check,
            low_vram=args.low_vram,
        )

        generated = _generate_image(
            pipe=pipeline,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
        )

        _save_image(generated, args.output)
        logger.info('Text-to-image generation completed successfully.')
    except Exception as exc:  # pragma: no cover - broad error handling
        logger.exception('Failed to generate image: %s', exc)
        raise


if __name__ == '__main__':  # pragma: no cover
    main()
