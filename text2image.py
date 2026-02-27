#!/usr/bin/env python3
"""
text2image.py

A production-ready, configurable text-to-image generator using Hugging Face Diffusers.

Features:
- CLI-friendly with sensible defaults
- GPU/CPU support and optional fp16 for memory savings
- Scheduler selection and sampling configuration
- Deterministic seeding for reproducibility
- Input validation, comprehensive logging, and error handling
- Optional performance boosts for xFormers and attention slicing

Usage examples:
  python text2image.py --prompt "A cute corgi wearing a hat" --model "runwayml/stable-diffusion-v1-5" --output ./out.png

See README.md for installation and environment details.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

try:
    from diffusers import (  # type: ignore
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    )
except Exception as e:  # pragma: no cover - library import error handled at runtime
    raise RuntimeError(
        "diffusers library is required. Install with 'pip install diffusers'. See requirements.txt for pinned versions."
    ) from e


# ---------------------------- Configuration DataClass ----------------------------

@dataclass
class GenerationConfig:
    model: str
    prompt: str
    negative_prompt: Optional[str]
    output: Path
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: Optional[int]
    device: str
    use_fp16: bool
    scheduler: str
    enable_xformers: bool
    enable_attention_slicing: bool
    hf_token: Optional[str]
    disable_safety_checker: bool


# ---------------------------- Utilities & Validation ----------------------------

def setup_logging(log_path: Optional[Path] = None) -> None:
    """Configure global logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_path:
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)


def validate_config(cfg: GenerationConfig) -> None:
    """Validate input configuration and raise ValueError for invalid values."""
    logger = logging.getLogger(__name__)

    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    if cfg.width % 8 != 0 or cfg.height % 8 != 0:
        raise ValueError("Image width and height must be multiples of 8 for most stable diffusion models.")

    if not (64 <= cfg.width <= 2048 and 64 <= cfg.height <= 2048):
        logger.warning("Width/height outside 64-2048 range may cause memory issues or unsupported model behavior.")

    if not (1 <= cfg.steps <= 150):
        raise ValueError("Steps must be between 1 and 150.")

    if not (0.0 <= cfg.guidance_scale <= 30.0):
        raise ValueError("guidance_scale must be between 0.0 and 30.0.")

    if cfg.seed is not None and cfg.seed < 0:
        raise ValueError("Seed must be non-negative or omitted for random behavior.")


def get_scheduler_cls(name: str):
    """Return scheduler class matching a short name."""
    name = name.lower()
    mapping: Dict[str, object] = {
        "dpmsolver": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "lms": LMSDiscreteScheduler,
        "pndm": PNDMScheduler,
    }
    return mapping.get(name, DPMSolverMultistepScheduler)


# ---------------------------- Core Generation Logic ----------------------------

def build_pipeline(cfg: GenerationConfig, torch_dtype: torch.dtype):
    """Create and configure the diffusion pipeline.

    Args:
        cfg: GenerationConfig
        torch_dtype: dtype to load model weights (float16 for fp16 mode on CUDA, otherwise float32)

    Returns:
        StableDiffusionPipeline
    """
    logger = logging.getLogger(__name__)

    # Prepare auth token if provided
    hf_kwargs: Dict[str, object] = {}
    token = cfg.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        hf_kwargs["use_auth_token"] = token

    scheduler_cls = get_scheduler_cls(cfg.scheduler)

    try:
        logger.info("Loading model %s with scheduler %s and dtype %s", cfg.model, scheduler_cls.__name__, torch_dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model,
            scheduler=scheduler_cls.from_pretrained(cfg.model) if hasattr(scheduler_cls, "from_pretrained") else None,
            torch_dtype=torch_dtype,
            safety_checker=None if cfg.disable_safety_checker else None,
            **hf_kwargs,
        )
    except Exception as exc:
        logger.exception("Failed to load model from Hugging Face: %s", exc)
        raise

    # Move to device
    try:
        pipe = pipe.to(cfg.device)
    except Exception:
        # Some pipelines require explicit device placement per component
        for name, comp in pipe.components.items():
            try:
                pipe.components[name] = comp.to(cfg.device)
            except Exception:
                pass

    # Performance optimizations
    try:
        if cfg.enable_xformers:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory efficient attention.")
    except Exception:
        logger.warning("xFormers not available or failed to enable.")

    try:
        if cfg.enable_attention_slicing:
            pipe.enable_attention_slicing()
            logger.info("Enabled attention slicing.")
    except Exception:
        logger.warning("Attention slicing not available or failed to enable.")

    return pipe


def generate_image(cfg: GenerationConfig) -> Path:
    """Generate an image for the given configuration and return path to saved file.

    This function provides careful resource management and deterministic seeding when requested.
    """
    logger = logging.getLogger(__name__)

    # dtype logic
    if cfg.device.startswith("cuda") and cfg.use_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Build pipeline
    pipe = build_pipeline(cfg, torch_dtype)

    # Create generator for reproducibility
    gen = None
    if cfg.seed is not None:
        try:
            gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        except Exception:
            # CPU generator fallback
            gen = torch.Generator().manual_seed(cfg.seed)

    # Run generation inside inference mode
    try:
        logger.info("Starting image generation: prompt='%s' steps=%s guidance=%.2f" % (cfg.prompt, cfg.steps, cfg.guidance_scale))
        with torch.inference_mode():
            image = pipe(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                height=cfg.height,
                width=cfg.width,
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=cfg.steps,
                generator=gen,
            ).images[0]

    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise

    # Ensure output folder exists
    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    # Save with atomic write pattern
    out_path = cfg.output
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        image.save(tmp_path)
        tmp_path.replace(out_path)
        logger.info("Saved generated image to %s", out_path)
    except Exception as exc:
        logger.exception("Failed to save image: %s", exc)
        raise

    return out_path


# ---------------------------- CLI Interface ----------------------------

def parse_args(argv: Optional[list] = None) -> GenerationConfig:
    """Parse CLI args into GenerationConfig dataclass."""
    parser = argparse.ArgumentParser(description="Text-to-Image generator using Hugging Face Diffusers")

    parser.add_argument("--prompt", required=True, help="Text prompt describing the desired image.")
    parser.add_argument("--negative_prompt", default=None, help="Negative prompt to dissuade features.")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id or path.")
    parser.add_argument("--output", default="./outputs/result.png", help="Output image path.")
    parser.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8).")
    parser.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8).")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic results.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device, e.g. 'cuda', 'cpu', or 'mps'.")
    parser.add_argument("--fp16", action="store_true", help="Load model in fp16 where supported (reduces memory usage on CUDA).")
    parser.add_argument("--scheduler", default="dpmsolver", help="Scheduler: dpmsolver|euler|lms|pndm")
    parser.add_argument("--enable_xformers", action="store_true", help="Enable xFormers memory efficient attention if installed.")
    parser.add_argument("--attention_slicing", action="store_true", help="Enable attention slicing to reduce memory footprint.")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token when accessing gated models. Can also use HF_TOKEN env var.")
    parser.add_argument("--disable_safety_checker", action="store_true", help="Disable safety checker. Use with caution in production.")

    parsed = parser.parse_args(argv)

    cfg = GenerationConfig(
        model=parsed.model,
        prompt=parsed.prompt,
        negative_prompt=parsed.negative_prompt,
        output=Path(parsed.output),
        width=parsed.width,
        height=parsed.height,
        steps=parsed.steps,
        guidance_scale=parsed.guidance_scale,
        seed=parsed.seed,
        device=parsed.device,
        use_fp16=parsed.fp16,
        scheduler=parsed.scheduler,
        enable_xformers=parsed.enable_xformers,
        enable_attention_slicing=parsed.attention_slicing,
        hf_token=parsed.hf_token,
        disable_safety_checker=parsed.disable_safety_checker,
    )

    return cfg


def main(argv: Optional[list] = None) -> int:
    """Main entry point. Returns 0 on success and non-zero on error."""
    cfg = parse_args(argv)
    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "text2image.log"
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    try:
        validate_config(cfg)
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2

    try:
        out = generate_image(cfg)
        logger.info("Image generation complete: %s", out)
        return 0
    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
