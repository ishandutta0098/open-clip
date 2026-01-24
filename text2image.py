#!/usr/bin/env python3
"""
text2image.py

Command-line utility to generate images from text prompts using Hugging Face Diffusers

Features:
- Loads a Stable Diffusion pipeline from Hugging Face Hub (or local path)
- Automatic device selection (CUDA if available, otherwise CPU)
- Mixed precision when running on CUDA for performance
- Input validation, deterministic seeding option
- Basic NSFW filtering handling via pipeline's safety checker
- Saves outputs with sanitized filenames, stores metadata in a JSON sidecar
- Comprehensive logging and error handling

Usage (example):
  python text2image.py --prompt "A cozy cottage in the woods, watercolor" --model "runwayml/stable-diffusion-v1-5" --num_images 2 --steps 30 --out_dir ./outputs

Environment:
- If accessing private HF models, set HUGGINGFACE_TOKEN or run `huggingface-cli login`.

References:
- https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
except Exception as e:  # pragma: no cover - import/runtime errors are handled below
    # We'll handle missing deps at runtime; keep script importable for static analysis
    torch = None  # type: ignore


# --------- Logging setup ---------
LOG = logging.getLogger("text2image")
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOG.addHandler(ch)


# --------- Utility data classes ---------
@dataclass
class GenerationConfig:
    prompt: str
    model: str = "runwayml/stable-diffusion-v1-5"
    num_images: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    output_dir: str = "outputs"
    dtype: Optional[str] = None  # 'fp16' or 'fp32'
    device: Optional[str] = None  # 'cuda' or 'cpu'


# --------- Helper functions ---------
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def sanitize_filename(s: str, max_len: int = 200) -> str:
    """Sanitize a string to be used as a filename.

    Keeps only alphanumerics, dot, underscore and dash. Truncates if too long.
    """
    safe = _SANITIZE_RE.sub("_", s)
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def pick_device(prefer_gpu: bool = True) -> str:
    """Return the best device string available: 'cuda' if GPU is available and preferred, otherwise 'cpu'."""
    if prefer_gpu and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dtype_for_device(device: str) -> torch.dtype:
    if device == "cuda":
        # use mixed precision on GPU to reduce memory and improve perf
        return torch.float16
    return torch.float32


def load_pipeline(model_id: str, device: str, dtype: torch.dtype, use_auth_token: Optional[str] = None):
    """Load and return a StableDiffusionPipeline configured for inference.

    Falls back safely if scheduler is not set, and enables attention slicing for low memory.
    """
    if torch is None:
        raise RuntimeError("torch is not available. Install required packages as listed in requirements.txt")

    LOG.info("Loading pipeline: %s on device=%s dtype=%s", model_id, device, dtype)
    try:
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_auth_token=use_auth_token,
        )

        # Replace scheduler with a performant multistep solver if available
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            LOG.debug("Could not move to DPMSolverMultistepScheduler; using default scheduler")

        # Performance tuning
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                LOG.debug("xFormers not available or failed to enable")

        pipe.to(device)
        # Enable attention slicing to reduce peak memory usage (trade-off: slower)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            LOG.debug("Attention slicing could not be enabled")

        return pipe
    except Exception as exc:
        LOG.error("Failed to load pipeline: %s", exc)
        raise


def _nsfw_check_and_handle(pipeline, images: List[Image.Image]) -> Tuple[List[Image.Image], List[bool]]:
    """Use the pipeline's safety checker to inspect NSFW flags and handle accordingly.

    Returns tuple(images_to_save, nsfw_flags)
    If images are flagged as NSFW, we will still return them but set their corresponding flag True.
    This function intentionally does not delete anything automatically; instead, it logs and returns flags
    so caller can decide how to handle.
    """
    nsfw_flags: List[bool] = []
    try:
        checker = getattr(pipeline, "safety_checker", None)
        if checker is None:
            LOG.debug("No safety checker found on pipeline; skipping NSFW check")
            nsfw_flags = [False] * len(images)
            return images, nsfw_flags

        # Convert to required inputs
        imgs_for_check = [np.array(img.convert("RGB")) for img in images]
        checked, has_nsfw_concepts = checker(imgs_for_check, [pipeline.feature_extractor(images=img, return_tensors="pt") for img in images])
        # Many safety checkers return a tuple; unify to flags
        if isinstance(has_nsfw_concepts, list):
            nsfw_flags = [bool(v) for v in has_nsfw_concepts]
        elif isinstance(has_nsfw_concepts, (tuple,)):
            nsfw_flags = [bool(v) for v in has_nsfw_concepts]
        else:
            # Fallback: assume False
            nsfw_flags = [False] * len(images)
    except Exception:
        # If safety check fails, don't block generation, but warn
        LOG.warning("Safety checker failed - continuing without NSFW checks")
        nsfw_flags = [False] * len(images)
    return images, nsfw_flags


def generate_images(
    pipe,
    cfg: GenerationConfig,
) -> List[Tuple[Path, dict]]:
    """Generate images from prompt according to cfg. Returns list of (image_path, metadata) tuples.

    Saves images under cfg.output_dir.
    """
    ensure_dir(cfg.output_dir)
    out_paths_and_meta: List[Tuple[Path, dict]] = []

    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(cfg.seed)
        LOG.info("Using deterministic seed: %d", cfg.seed)

    LOG.info("Generating %d image(s) ...", cfg.num_images)

    # For batch generation, generate images in a single call where supported
    try:
        # Dtype context for CPU/FP16 handling
        if pipe.device.type == "cuda":
            context = torch.cuda.amp.autocast(pipe.device.type, enabled=True)
        else:
            # No-op context manager
            from contextlib import nullcontext

            context = nullcontext()

        with context:
            result = pipe(
                prompt=cfg.prompt,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                num_images_per_prompt=cfg.num_images,
                generator=generator,
            )

        images = result.images
        # Some pipelines return images in a list-of-lists if prompt is batched; flatten
        if isinstance(images[0], list):
            flat_images = []
            for item in images:
                flat_images.extend(item)
            images = flat_images

        # NSFW checking - optional handling
        try:
            # Lazy import numpy and use pipeline safety checker if available
            import numpy as np
            images, nsfw_flags = _nsfw_check_and_handle(pipe, images)
        except Exception:
            nsfw_flags = [False] * len(images)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for idx, img in enumerate(images):
            safe_prompt = sanitize_filename(cfg.prompt)[:150]
            fname_base = f"{timestamp}_{safe_prompt}_{idx}"
            img_path = Path(cfg.output_dir) / f"{fname_base}.png"
            img.save(img_path)

            meta = {
                "prompt": cfg.prompt,
                "model": cfg.model,
                "seed": cfg.seed,
                "height": cfg.height,
                "width": cfg.width,
                "num_inference_steps": cfg.num_inference_steps,
                "guidance_scale": cfg.guidance_scale,
                "nsfw": bool(nsfw_flags[idx]) if idx < len(nsfw_flags) else False,
                "file": str(img_path.name),
                "generated_at": timestamp,
            }

            # Save metadata sidecar
            meta_path = img_path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)

            out_paths_and_meta.append((img_path, meta))
            LOG.info("Saved image: %s (nsfw=%s)", img_path, meta["nsfw"])

        return out_paths_and_meta

    except Exception as exc:
        LOG.exception("Image generation failed: %s", exc)
        raise


# --------- CLI ---------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from text prompts using HF Diffusers")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate images from")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="HF model id or local path")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps (inference steps)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for determinism")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for generated images")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to run the pipeline on")
    parser.add_argument("--auth_token", type=str, default=None, help="Hugging Face auth token (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        LOG.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)

    try:
        # Validate inputs
        if not args.prompt.strip():
            LOG.error("Prompt is empty")
            return 2
        if args.num_images < 1 or args.num_images > 8:
            LOG.error("num_images must be between 1 and 8 (got: %d)", args.num_images)
            return 2
        if args.steps < 1 or args.steps > 500:
            LOG.error("steps must be between 1 and 500")
            return 2

        prefer_gpu = True
        if args.device == "cpu":
            prefer_gpu = False
        elif args.device == "cuda":
            prefer_gpu = True

        resolved_device = pick_device(prefer_gpu)
        dtype = dtype_for_device(resolved_device)

        cfg = GenerationConfig(
            prompt=args.prompt,
            model=args.model,
            num_images=args.num_images,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
            output_dir=args.out_dir,
            dtype="fp16" if dtype == torch.float16 else "fp32",
            device=resolved_device,
        )

        # Load pipeline
        pipe = load_pipeline(cfg.model, cfg.device, dtype, use_auth_token=args.auth_token)

        # Generate
        outputs = generate_images(pipe, cfg)

        LOG.info("Generation complete. %d files saved to %s", len(outputs), cfg.output_dir)
        return 0

    except Exception as exc:
        LOG.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
