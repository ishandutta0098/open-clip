#!/usr/bin/env python3
"""
text2image.py

A production-ready CLI utility to generate images from text prompts using
Hugging Face Diffusers (Stable Diffusion).

Features:
- GPU/CPU aware model loading with automatic dtype selection
- Batch generation and seeding
- Input validation and safety recommendations
- Configurable inference parameters (steps, guidance, size)
- Proper logging, error handling, and clear CLI
- Optional Hugging Face authentication token support

Usage example:
python text2image.py --model 'runwayml/stable-diffusion-v1-5' \
    --prompt "a beautiful landscape, sunrise over mountains" \
    --outdir ./outputs --num_inference_steps 30 --guidance_scale 7.5

"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

try:
    # diffusers and transformers are external dependencies
    from diffusers import StableDiffusionPipeline
    from diffusers.models import AutoencoderKL
except Exception as e:  # pragma: no cover - real imports required at runtime
    StableDiffusionPipeline = None  # type: ignore
    AutoencoderKL = None  # type: ignore


# --------------------------- Configuration & Logging ---------------------------
LOG = logging.getLogger("text2image")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for consistent output."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers in interactive environments
    if not any(isinstance(h, type(handler)) and h.formatter._fmt == handler.formatter._fmt for h in root.handlers):
        root.addHandler(handler)


# --------------------------- Utilities & Validation ---------------------------

def validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def ensure_multiple_of_eight(value: int, name: str) -> int:
    """Stable Diffusion expects dimensions that are multiples of 8 (commonly 8 or 64).

    This helper rounds down to the nearest multiple of 8 and warns when changed.
    """
    validate_positive_int(value, name)
    if value % 8 == 0:
        return value
    adjusted = (value // 8) * 8
    if adjusted <= 0:
        raise ValueError(f"{name} too small after rounding to multiple of 8")
    LOG.warning("%s=%d is not multiple of 8; rounding down to %d", name, value, adjusted)
    return adjusted


def seed_worker(seed: Optional[int] = None) -> torch.Generator:
    g = torch.Generator()
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") >> 1
    g.manual_seed(seed)
    return g


# --------------------------- Core Generation Class ---------------------------

@dataclass
class Text2ImageConfig:
    model_id: str
    device: str = "cuda" if torch and torch.cuda.is_available() else "cpu"
    safety_checker: bool = True
    torch_dtype: Optional[torch.dtype] = None
    hf_token: Optional[str] = None


class Text2ImageGenerator:
    """Wrapper around Hugging Face Diffusers Stable Diffusion pipeline.

    Responsibilities:
    - Load model in an optimal dtype for the available device (fp16 for CUDA)
    - Provide a generate_images API that validates inputs and saves outputs
    - Provide small, predictable memory footprint by unloading when necessary
    """

    def __init__(self, config: Text2ImageConfig):
        """Initialize the generator.

        Args:
            config: Text2ImageConfig instance containing model and device settings.
        """
        if StableDiffusionPipeline is None:
            raise RuntimeError(
                "diffusers package not available. Please install requirements: see requirements.txt"
            )

        self.config = config
        self.pipeline: Optional[StableDiffusionPipeline] = None

    def _select_dtype(self) -> Optional[torch.dtype]:
        if self.config.torch_dtype is not None:
            return self.config.torch_dtype
        if self.config.device == "cuda":
            # Use fp16 on CUDA to reduce memory and speed up inference
            if torch.cuda.is_available():
                return torch.float16
        # Use default (float32) on CPU
        return torch.float32

    def load_pipe(self, revision: Optional[str] = None) -> None:
        """Load the Stable Diffusion pipeline with configuration aware settings.

        Args:
            revision: Optional model revision (tag or commit) to load.
        """
        LOG.info("Loading model %s on device=%s", self.config.model_id, self.config.device)
        dtype = self._select_dtype()

        pipe_kwargs = {
            "torch_dtype": dtype,
        }

        if self.config.hf_token:
            # use_auth_token is accepted by diffusers and transformers to access gated models
            pipe_kwargs["use_auth_token"] = self.config.hf_token

        if revision:
            pipe_kwargs["revision"] = revision

        # Use from_pretrained; device mapping handled by .to(device) afterwards.
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.config.model_id, **pipe_kwargs)

        # Optional: disable safety checker if requested (must consider ethics/security implications).
        if not self.config.safety_checker:
            try:
                # Newer diffusers versions might have .safety_checker attribute
                self.pipeline.safety_checker = None  # type: ignore[attr-defined]
                LOG.info("Disabled safety checker on pipeline (developer requested)")
            except Exception:
                LOG.debug("No safety checker attribute to disable on the pipeline")

        # Move pipeline to desired device
        self.pipeline = self.pipeline.to(self.config.device)

        # Enable attention slicing for memory constrained devices
        try:
            self.pipeline.enable_attention_slicing()
            LOG.debug("Enabled attention slicing for memory efficiency")
        except Exception:
            LOG.debug("Pipeline does not support attention slicing")

        LOG.info("Model loaded and moved to device")

    def unload_pipe(self) -> None:
        """Free pipeline from memory (useful in long-running services)."""
        if self.pipeline is not None:
            try:
                del self.pipeline
            except Exception:
                LOG.exception("Error when unloading pipeline")
            finally:
                self.pipeline = None
                # Force garbage collection in long-running processes
                try:
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    LOG.debug("Failed to run GC/empty_cache")

    def generate_images(
        self,
        prompts: Sequence[str],
        outdir: Path,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        batch_size: int = 1,
    ) -> List[Path]:
        """Generate images for the provided prompts.

        Args:
            prompts: A sequence of text prompts. If >1 and batch_size>1, will generate multiple per call.
            outdir: Directory where images will be saved.
            num_inference_steps: Number of diffusion steps (trade-off quality vs speed).
            guidance_scale: Classifier-free guidance scale.
            height: Image height in pixels (will be rounded to multiple of 8).
            width: Image width in pixels (will be rounded to multiple of 8).
            seed: Optional deterministic seed.
            negative_prompt: Optional negative prompt to avoid elements.
            batch_size: How many prompts to synthesize at once (efficient for GPU).

        Returns:
            List of Path objects pointing to saved images.
        """
        if not prompts:
            raise ValueError("At least one prompt is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        num_inference_steps = validate_positive_int(int(num_inference_steps), "num_inference_steps")
        guidance_scale = float(guidance_scale)
        height = ensure_multiple_of_eight(int(height), "height")
        width = ensure_multiple_of_eight(int(width), "width")

        outdir.mkdir(parents=True, exist_ok=True)

        if self.pipeline is None:
            self.load_pipe()

        gen = seed_worker(seed)

        results: List[Path] = []

        device = torch.device(self.config.device)

        # Generate in chunks to avoid OOM for large prompt lists
        prompts = list(prompts)
        total = len(prompts)

        LOG.info(
            "Starting generation: total_prompts=%d, batch_size=%d, device=%s, seed=%s",
            total,
            batch_size,
            device,
            seed,
        )

        # Use inference_mode for faster inference and lower memory when available
        inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

        with inference_ctx():
            for start in range(0, total, batch_size):
                batch_prompts = prompts[start : start + batch_size]

                try:
                    LOG.debug("Generating batch prompts=%s", batch_prompts)

                    # The pipeline accepts a list of prompts and will return a list of PIL images
                    output = self.pipeline(
                        prompt=batch_prompts,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt,
                        generator=gen,
                    )

                    images = output.images
                except Exception as ex:
                    LOG.exception("Model inference failed for batch starting at %d", start)
                    raise RuntimeError("Model inference failed") from ex

                for i, img in enumerate(images):
                    unique_suffix = uuid.uuid4().hex[:8]
                    timestamp = int(time.time())
                    filename = f"img_{timestamp}_{start + i}_{unique_suffix}.png"
                    file_path = outdir / filename
                    try:
                        img.save(file_path, format="PNG")
                        results.append(file_path)
                        LOG.info("Saved generated image to %s", file_path)
                    except Exception:
                        LOG.exception("Failed to save image to %s", file_path)
                        raise

        return results


# --------------------------- CLI Entrypoint ---------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using Hugging Face Diffusers (Stable Diffusion)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model ID or path (e.g. 'runwayml/stable-diffusion-v1-5' or local path)",
    )
    parser.add_argument("--prompt", type=str, nargs="+", help="One or more prompts", required=True)
    parser.add_argument("--outdir", type=Path, default=Path("./outputs"), help="Output directory")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps (higher -> slower/higher quality)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Image height (pixels).")
    parser.add_argument("--width", type=int, default=512, help="Image width (pixels).")
    parser.add_argument("--batch_size", type=int, default=1, help="Prompts to generate per batch")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic outputs")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to avoid unwanted features")
    parser.add_argument("--no_safety_checker", action="store_true", help="Disable the safety checker (use responsibly)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models or private repos")
    parser.add_argument("--revision", type=str, default=None, help="Optional model revision tag")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (e.g. 'cpu' or 'cuda')")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    config = Text2ImageConfig(
        model_id=args.model,
        device=device,
        safety_checker=not args.no_safety_checker,
        torch_dtype=None,  # autodetect in loader
        hf_token=args.hf_token or os.getenv("HF_ACCESS_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
    )

    generator = Text2ImageGenerator(config=config)

    try:
        generator.load_pipe(revision=args.revision)

        results = generator.generate_images(
            prompts=args.prompt,
            outdir=args.outdir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            batch_size=args.batch_size,
        )

        LOG.info("Completed generation. %d images saved.", len(results))
        return 0
    except Exception:
        LOG.exception("Generation failed")
        return 2
    finally:
        # Ensure memory is freed if this script runs in a persistent process
        try:
            generator.unload_pipe()
        except Exception:
            LOG.debug("Error during pipeline unload")


if __name__ == "__main__":
    raise SystemExit(main())
