#!/usr/bin/env python3
"""
text2image.py

A production-ready command-line utility to generate images from text prompts using
Hugging Face Diffusers (Stable Diffusion). Supports device selection (cuda/mps/cpu),
FP16 for GPUs, scheduler selection, reproducible seeding, input validation, and
secure model access via HF token.

Usage examples:

  python text2image.py --prompt "A cozy cabin in a snowy forest, warm lights" --out output.png

  python text2image.py --prompt-file prompts.txt --batch 3 --model "runwayml/stable-diffusion-v1-5" \
    --hf-token $HF_TOKEN --steps 30 --guidance 7.5 --width 768 --height 512

"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from typing import Optional, Tuple, List

try:
    import torch
except Exception as e:  # pragma: no cover - informative error
    raise RuntimeError("PyTorch is required. Install via pip install torch") from e

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler
except Exception as e:  # pragma: no cover - informative error
    raise RuntimeError("diffusers is required. Install via pip install diffusers") from e

from PIL import Image

# Module-level logger
logger = logging.getLogger("text2image")

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"


def configure_logging(verbosity: int = 1) -> None:
    """
    Configure module-level logging.

    Args:
        verbosity: 0 = WARNING, 1 = INFO (default), 2+ = DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    # prevent duplicated handlers in interactive environments
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    else:
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level)


def _select_device(prefer_cuda: bool = True) -> str:
    """
    Select the best available device for inference.

    Returns:
        device string recognized by torch ("cuda", "mps", or "cpu").
    """
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    # Apple silicon support
    if hasattr(torch, "has_mps") and torch.has_mps:
        return "mps"
    return "cpu"


def _validate_dimensions(width: int, height: int) -> Tuple[int, int]:
    """
    Stable Diffusion models usually require image dimensions to be multiples of 8.
    Adjusts to nearest multiple of 8 (rounding down) and ensures positive values.

    Args:
        width: requested width
        height: requested height

    Returns:
        Tuple of validated (width, height)
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")

    def down_to_multiple_of_8(x: int) -> int:
        return max(8, (x // 8) * 8)

    w = down_to_multiple_of_8(width)
    h = down_to_multiple_of_8(height)
    if (w, h) != (width, height):
        logger.warning("Adjusted image dimensions from (%d,%d) to nearest multiples of 8: (%d,%d)", width, height, w, h)
    return w, h


def _get_scheduler(name: str):
    """
    Map a friendly scheduler name to a diffusers scheduler class instance.
    """
    name = (name or "dpmsolver").lower()
    if "dpmsolver" in name:
        return DPMSolverMultistepScheduler.from_config
    if "lms" in name:
        return LMSDiscreteScheduler.from_config
    if "euler" in name:
        return EulerAncestralDiscreteScheduler.from_config
    # default as DPMSolver
    return DPMSolverMultistepScheduler.from_config


def load_pipeline(model_id: str, device: str, hf_token: Optional[str], fp16: bool, scheduler: Optional[str]):
    """
    Load the Stable Diffusion pipeline with sensible defaults.

    Args:
        model_id: HF model repo identifier
        device: "cuda" | "mps" | "cpu"
        hf_token: optional HF access token (recommended for gated models)
        fp16: whether to use float16 for torch dtype on CUDA
        scheduler: optional scheduler name

    Returns:
        Initialized StableDiffusionPipeline
    """
    # Choose dtype
    torch_dtype = None
    if device == "cuda" and fp16:
        torch_dtype = torch.float16
    elif device == "mps":
        # mps works best with float32 in many setups
        torch_dtype = torch.float32

    logger.info("Loading model '%s' on device=%s fp16=%s", model_id, device, bool(torch_dtype == torch.float16))

    try:
        # Load base pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_auth_token=hf_token,
            safety_checker=None,  # explicit: we will provide opt-in for safety
            revision=None,
        )

        # Optionally swap scheduler
        if scheduler:
            scheduler_ctor = _get_scheduler(scheduler)
            pipeline.scheduler = scheduler_ctor(pipeline.scheduler.config)

        if device == "cuda":
            pipeline.to("cuda")
        elif device == "mps":
            pipeline.to("mps")
        else:
            pipeline.to("cpu")

        # Enable memory efficient attention if available
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            # Optional optimization; ignore if not available
            logger.debug("xformers not available or failed to enable; continuing without it")

        return pipeline
    except Exception as exc:  # pragma: no cover - handle runtime environment issues
        logger.exception("Failed to load pipeline: %s", exc)
        raise


def generate_image(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    out_path: str,
    seed: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    batch: int = 1,
) -> List[str]:
    """
    Generate one or more images from a text prompt and save them.

    Args:
        pipeline: Loaded Stable Diffusion pipeline
        prompt: Prompt string to render
        out_path: Output file path template (e.g., out.png or out_%d.png)
        seed: Optional integer seed for reproducibility
        num_inference_steps: Sampling steps
        guidance_scale: Classifier-free guidance scale
        width: Image width
        height: Image height
        batch: How many images to produce

    Returns:
        List of saved file paths
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    width, height = _validate_dimensions(width, height)

    device = next(pipeline.parameters()).device if hasattr(pipeline, "parameters") else None

    # Prepare generator for reproducibility
    generator = None
    if seed is not None:
        # The generator should be created on the target device
        gen_device = str(device) if device is not None else "cpu"
        try:
            generator = torch.Generator(device=gen_device).manual_seed(seed)
        except Exception:
            # fallback: cpu generator
            generator = torch.Generator(device="cpu").manual_seed(seed)

    outputs: List[str] = []

    # Support templated filenames so user can pass out_%d.png
    base, ext = os.path.splitext(out_path)
    if batch == 1 and "%d" not in out_path:
        filenames = [out_path]
    else:
        # ensure file pattern
        if "%d" not in out_path:
            filenames = [f"{base}_{i}{ext}" for i in range(batch)]
        else:
            filenames = [out_path % i for i in range(batch)]

    logger.info("Generating %d image(s) with prompt: %s", batch, (prompt if len(prompt) < 256 else prompt[:250] + "..."))

    for i in range(batch):
        try:
            result = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = result.images[0]
            out_file = filenames[i]
            os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
            image.save(out_file)
            outputs.append(out_file)
            logger.info("Saved image: %s", out_file)
        except Exception as exc:  # pragma: no cover - catch runtime generation errors
            logger.exception("Failed to generate/save image #%d: %s", i, exc)
            raise

    return outputs


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-to-Image generation using Hugging Face Diffusers")
    parser.add_argument("--prompt", type=str, help="Prompt text to generate an image for", default=None)
    parser.add_argument("--prompt-file", type=str, help="Path to a file with prompts, one per line (mutually exclusive with --prompt)")
    parser.add_argument("--out", type=str, default="output.png", help="Output filename or pattern (supports %%d for batch: e.g. out_%%d.png)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model repo id")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Device to use; default 'auto' picks cuda->mps->cpu")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 (only effective on CUDA)")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width (will be adjusted to multiple of 8 if needed)")
    parser.add_argument("--height", type=int, default=512, help="Image height (will be adjusted to multiple of 8 if needed)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible results")
    parser.add_argument("--batch", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--scheduler", type=str, default=None, help="Optional scheduler name: dpmsolver|lms|euler")
    parser.add_argument("--verbosity", type=int, default=1, help="Logging verbosity: 0=warning,1=info,2=debug")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.verbosity)

    # Validate prompt input
    prompts: List[str] = []
    if args.prompt and args.prompt_file:
        logger.error("--prompt and --prompt-file are mutually exclusive")
        return 2

    if args.prompt_file:
        if not os.path.isfile(args.prompt_file):
            logger.error("Prompt file does not exist: %s", args.prompt_file)
            return 2
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            prompts = [line.strip() for line in fh if line.strip()]
        if not prompts:
            logger.error("Prompt file is empty: %s", args.prompt_file)
            return 2
    elif args.prompt:
        prompts = [args.prompt]
    else:
        logger.error("Either --prompt or --prompt-file must be provided")
        return 2

    # Device selection
    device = args.device
    if device == "auto":
        device = _select_device(prefer_cuda=True)
    logger.info("Using device: %s", device)

    if device == "cpu" and args.fp16:
        logger.warning("fp16 requested but unavailable on CPU. Ignoring fp16 flag.")

    # Load pipeline once and reuse for all prompts
    try:
        pipeline = load_pipeline(args.model, device=device, hf_token=args.hf_token, fp16=args.fp16, scheduler=args.scheduler)
    except Exception as exc:
        logger.error("Failed to initialize model pipeline: %s", exc)
        return 3

    all_outputs: List[str] = []
    start_time = time.time()
    try:
        for idx, p in enumerate(prompts):
            out_template = args.out
            # If multiple prompts are provided, create unique outputs
            if len(prompts) > 1:
                base, ext = os.path.splitext(out_template)
                if "%d" not in out_template:
                    out_template = f"{base}_{idx}%d{ext}"
            try:
                outs = generate_image(
                    pipeline=pipeline,
                    prompt=p,
                    out_path=out_template,
                    seed=args.seed,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    width=args.width,
                    height=args.height,
                    batch=args.batch,
                )
                all_outputs.extend(outs)
            except Exception as e:
                logger.exception("Failed generating for prompt index %d: %s", idx, e)
                # continue to next prompt instead of aborting entire run
                continue
    finally:
        elapsed = time.time() - start_time
        logger.info("Generation completed in %.2fs. Generated %d image(s)", elapsed, len(all_outputs))

    if not all_outputs:
        logger.error("No images were generated")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
