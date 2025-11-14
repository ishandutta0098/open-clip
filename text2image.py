#!/usr/bin/env python3
"""
text2image.py

A production-ready CLI script to generate images from text prompts using Hugging Face Diffusers.

Features:
- Uses Stable Diffusion pipeline from diffusers.
- Automatic device selection (CUDA if available, else CPU) with mixed precision (autocast) when possible.
- Deterministic generation via seed.
- Input validation and path sanitization.
- Helpful logging and error handling.

Usage examples:
  python text2image.py --prompt "A fantasy castle on a hill at sunrise" --out output.png --steps 30 --guidance 7.5 --width 768 --height 512

Environment:
- For private models, set HF_TOKEN environment variable with a valid Hugging Face token or use `huggingface-cli login`.

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

try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from PIL import Image
except Exception as exc:  # pragma: no cover - runtime dependency errors
    raise RuntimeError(
        "Required packages are missing. Please install dependencies from requirements.txt"
    ) from exc


# Module-level logger
logger = logging.getLogger("text2image")


@dataclass
class GenerationConfig:
    model: str
    prompt: str
    out_path: Path
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: Optional[torch.dtype] = None


def configure_logging(verbose: bool = False) -> None:
    """Configure module logging.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def validate_prompt(prompt: str) -> None:
    """Validate prompt input to avoid abuse or accidental empty prompts.

    Args:
        prompt: The user-supplied text prompt.

    Raises:
        ValueError: If prompt is invalid.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    if len(prompt) > 2000:
        raise ValueError("Prompt is too long; please limit to 2000 characters.")


def sanitize_output_path(out_path: str) -> Path:
    """Sanitize and normalize the output path.

    Prevents absolute paths outside current working directory to avoid accidental writes
    to sensitive locations.

    Args:
        out_path: The user-provided path as string.

    Returns:
        Path: A resolved, safe path to write the output to.

    Raises:
        ValueError: If the path resolves outside the current working directory tree.
    """
    path = Path(out_path)
    if path.is_dir():
        raise ValueError("Output path must be a file, not a directory.")
    # Create parent directories if needed
    parent = path.parent or Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        resolved.relative_to(cwd)
    except Exception:
        # If not relative to cwd, disallow to prevent path traversal.
        raise ValueError(
            f"Output path {resolved} is outside the current working directory {cwd}."
        )
    return resolved


def get_torch_dtype(device: str) -> Optional[torch.dtype]:
    """Choose an appropriate torch dtype for the device.

    Use float16 for CUDA if available for memory and speed benefits. Keep float32 for CPU.
    """
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def load_pipeline(model_id: str, device: str, torch_dtype: Optional[torch.dtype]) -> StableDiffusionPipeline:
    """Load and prepare the Stable Diffusion pipeline.

    - Downloads the model if not present (requires HF_TOKEN for restricted models).
    - Uses DPMSolverMultistepScheduler for improved sampling performance.
    - Enables attention slicing for lower VRAM usage.

    Args:
        model_id: Hugging Face model identifier.
        device: Device string such as "cuda" or "cpu".
        torch_dtype: Torch data type to use (float16 on CUDA recommended).

    Returns:
        Initialized StableDiffusionPipeline.

    Raises:
        RuntimeError: If model loading fails.
    """
    logger.info("Loading model '%s' on device=%s dtype=%s", model_id, device, torch_dtype)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disable built-in safety checker; users should enforce policies externally
        )
    except Exception as exc:
        logger.exception("Failed to load model %s", model_id)
        raise RuntimeError(f"Failed to load model {model_id}: {exc}") from exc

    # Replace scheduler for better performance if available
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception:
        logger.debug("Could not replace scheduler; continuing with default scheduler.")

    # Performance optimizations
    pipe.enable_attention_slicing()

    # Move to device
    try:
        pipe.to(device)
    except Exception as exc:
        logger.exception("Failed to move pipeline to device %s", device)
        raise RuntimeError(f"Failed to move pipeline to device {device}: {exc}") from exc

    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    cfg: GenerationConfig,
) -> Image.Image:
    """Generate an image given a pipeline and configuration.

    Args:
        pipe: Initialized StableDiffusionPipeline.
        cfg: GenerationConfig with parameters for generation.

    Returns:
        PIL.Image.Image: Generated image.
    """
    logger.info(
        "Generating image: prompt='%s' steps=%d guidance=%s size=%dx%d seed=%s",
        cfg.prompt,
        cfg.num_inference_steps,
        cfg.guidance_scale,
        cfg.width,
        cfg.height,
        cfg.seed,
    )

    generator = None
    if cfg.seed is not None:
        device = cfg.device if cfg.device != "cpu" else "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(cfg.seed)
        except Exception:
            # Fall back to CPU generator if device-specific generator fails
            generator = torch.Generator().manual_seed(cfg.seed)

    # Use autocast for mixed precision on CUDA devices to gain speed and reduce memory
    context_manager = torch.autocast(cfg.device) if cfg.device.startswith("cuda") else torch.cpu.amp.autocast(enabled=False)

    # Call pipeline
    with context_manager:
        try:
            result = pipe(
                prompt=cfg.prompt,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                generator=generator,
            )
        except RuntimeError as exc:
            logger.exception("Runtime error during generation (OOM or similar)")
            raise

    image = result.images[0]
    return image


def build_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    """Convert parsed CLI args into GenerationConfig with validation."""
    validate_prompt(args.prompt)
    out_path = sanitize_output_path(args.out)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = get_torch_dtype(device)

    return GenerationConfig(
        model=args.model,
        prompt=args.prompt,
        out_path=out_path,
        num_inference_steps=max(1, int(args.steps)),
        guidance_scale=float(args.guidance),
        width=int(args.width),
        height=int(args.height),
        seed=(int(args.seed) if args.seed is not None else None),
        device=device,
        torch_dtype=torch_dtype,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from text using Diffusers Stable Diffusion pipeline")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate the image from")
    parser.add_argument("--out", required=True, help="Output image path (e.g., ./outputs/my.png)")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id to use")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (higher = better quality, slower)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8 preferred)")
    parser.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8 preferred)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Device to run on: 'cuda' or 'cpu' (auto-detected if omitted)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)
    if args.verbose:
        configure_logging(verbose=True)

    try:
        cfg = build_config_from_args(args)
    except Exception as exc:
        logger.error("Invalid inputs: %s", exc)
        return 2

    # Warn about CPU usage
    if cfg.device == "cpu":
        logger.warning("Running on CPU. This will be slow and may fail due to memory constraints.")

    # Load pipeline
    try:
        pipe = load_pipeline(cfg.model, cfg.device, cfg.torch_dtype)
    except Exception as exc:
        logger.error("Could not initialize pipeline: %s", exc)
        return 3

    # Generate
    start = time.time()
    try:
        image = generate_image(pipe, cfg)
    except RuntimeError as exc:
        logger.error("Generation failed: %s", exc)
        return 4
    except Exception as exc:
        logger.exception("Unexpected error during generation")
        return 5
    duration = time.time() - start

    # Save
    try:
        # If no extension provided, default to PNG
        suffix = cfg.out_path.suffix or ".png"
        out_file = cfg.out_path.with_suffix(suffix)
        image.save(out_file)
    except Exception as exc:
        logger.exception("Failed to save image to %s", cfg.out_path)
        return 6

    logger.info("Image saved to %s (took %.2f sec)", out_file, duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
