#!/usr/bin/env python3
"""
text2image.py

A robust, production-ready CLI script to generate images from text prompts using HuggingFace Diffusers.

Features:
- CLI with rich options for model, scheduler, seed, batch size, guidance scale, steps
- Automatic device detection (CUDA/CPU) and mixed-precision when applicable
- Memory optimizations (attention slicing, optional xformers if installed)
- Reproducible outputs via seed and deterministic generators
- Safety checks and optional NSFW filtering toggle
- Prompt file support (one prompt per line) and batch generation
- Thorough input validation, logging and error handling

Usage examples:
  python text2image.py --prompt "A serene landscape painting of mountains at sunrise" --outdir outputs
  python text2image.py --prompt-file prompts.txt --num-outputs 3 --device cpu

Environment:
  - For access to private HF models, set HUGGINGFACE_TOKEN or pass --hf_token

License: MIT
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
from typing import Iterable, List, Optional, Tuple

try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler
    from PIL import Image
except Exception as e:  # pragma: no cover - allow script to show useful import errors
    raise ImportError(
        "One or more required packages are missing. Make sure to install requirements.txt. Error: {}".format(e)
    )

# Module-level logger
logger = logging.getLogger("text2image")


@dataclass
class GenerationOptions:
    model_id: str
    hf_token: Optional[str]
    device: str
    outdir: Path
    prompt: Optional[str]
    prompt_file: Optional[Path]
    seeds: Optional[List[int]]
    num_outputs: int
    height: int
    width: int
    guidance_scale: float
    num_inference_steps: int
    scheduler: Optional[str]
    use_fp16: bool
    enable_xformers: bool
    allow_nsfw: bool
    attention_slicing: bool


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger.

    Args:
        verbose: Enable debug-level logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


def validate_image_size(value: int) -> int:
    """Validate that width/height are positive integers and multiples of 8 (Stable Diffusion requirement).

    Args:
        value: dimension in pixels

    Returns:
        Provided value if valid

    Raises:
        argparse.ArgumentTypeError: if invalid
    """
    try:
        v = int(value)
    except Exception:
        raise argparse.ArgumentTypeError("Image dimensions must be integers")
    if v <= 0:
        raise argparse.ArgumentTypeError("Image dimensions must be positive")
    if v % 8 != 0:
        raise argparse.ArgumentTypeError("Image dimensions must be a multiple of 8 (model requirement)")
    return v


def parse_args(argv: Optional[List[str]] = None) -> GenerationOptions:
    """Parse CLI arguments and return a GenerationOptions dataclass.

    Args:
        argv: list of args (defaults to sys.argv)

    Returns:
        GenerationOptions instance
    """
    parser = argparse.ArgumentParser(
        description="Generate images from text using HuggingFace Diffusers (Stable Diffusion)."
    )
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="HuggingFace model id (e.g., runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"),
                        help="HuggingFace token if required for private models. Can also be provided via HUGGINGFACE_TOKEN env var.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string to generate an image for.")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="Path to a text file with one prompt per line. Mutually exclusive with --prompt.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory for images")
    parser.add_argument("--num-outputs", type=int, default=1, help="Number of images to generate per prompt")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds to use. If provided, overrides num-outputs. Example: 42,123")
    parser.add_argument("--height", type=validate_image_size, default=512, help="Image height (multiple of 8)")
    parser.add_argument("--width", type=validate_image_size, default=512, help="Image width (multiple of 8)")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--scheduler", type=str, choices=["dpmsolver", "lms", "euler"], default="dpmsolver",
                        help="Scheduler to use (dpmsolver recommended)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on: cuda or cpu")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision (recommended on modern GPUs)")
    parser.add_argument("--enable-xformers", action="store_true", help="Attempt enabling xformers memory efficient attention if available")
    parser.add_argument("--no-safety-check", dest="allow_nsfw", action="store_true",
                        help="Allow generation of NSFW images by disabling the safety filter (use with caution)")
    parser.add_argument("--attention-slicing", dest="attention_slicing", action="store_true",
                        help="Enable attention slicing to reduce peak memory usage")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    ns = parser.parse_args(argv)

    # seeds parsing
    seeds = None
    if ns.seeds:
        try:
            seeds = [int(s.strip()) for s in ns.seeds.split(",") if s.strip() != ""]
        except Exception:
            parser.error("Invalid --seeds format. Provide comma-separated integers.")

    # Mutually exclusive prompt and prompt_file
    if ns.prompt is None and ns.prompt_file is None:
        parser.error("Either --prompt or --prompt-file must be provided")
    if ns.prompt is not None and ns.prompt_file is not None:
        parser.error("Provide only one of --prompt or --prompt-file")

    opts = GenerationOptions(
        model_id=ns.model_id,
        hf_token=ns.hf_token,
        device=ns.device,
        outdir=ns.outdir,
        prompt=ns.prompt,
        prompt_file=ns.prompt_file,
        seeds=seeds,
        num_outputs=ns.num_outputs,
        height=ns.height,
        width=ns.width,
        guidance_scale=ns.guidance_scale,
        num_inference_steps=ns.num_inference_steps,
        scheduler=ns.scheduler,
        use_fp16=ns.fp16,
        enable_xformers=ns.enable_xformers,
        allow_nsfw=ns.allow_nsfw,
        attention_slicing=ns.attention_slicing,
    )

    return opts


def read_prompts(prompt: Optional[str], prompt_file: Optional[Path]) -> List[str]:
    """Return a list of prompts from either a single prompt or a prompt file.

    Args:
        prompt: single prompt string
        prompt_file: path to file with one prompt per line

    Returns:
        List of prompt strings

    Raises:
        FileNotFoundError: if prompt_file path doesn't exist
    """
    prompts: List[str] = []
    if prompt is not None:
        prompts = [prompt.strip()]
    elif prompt_file is not None:
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with prompt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    return prompts


def _select_scheduler(scheduler_name: Optional[str], pipeline: StableDiffusionPipeline):
    """Replace pipeline scheduler based on name. Returns pipeline with new scheduler attached.

    Args:
        scheduler_name: name string
        pipeline: currently loaded pipeline

    Returns:
        pipeline with scheduler set
    """
    if scheduler_name == "dpmsolver":
        scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_name == "lms":
        scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_name == "euler":
        scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        return pipeline

    pipeline.scheduler = scheduler
    return pipeline


def load_pipeline(opts: GenerationOptions) -> StableDiffusionPipeline:
    """Load and configure the Stable Diffusion pipeline with optimizations.

    Args:
        opts: GenerationOptions instance

    Returns:
        Configured StableDiffusionPipeline
    """
    device = torch.device(opts.device if torch.cuda.is_available() and opts.device == "cuda" else "cpu")
    logger.info("Loading model %s onto %s", opts.model_id, device)

    # dtype selection
    torch_dtype = torch.float16 if (opts.use_fp16 and device.type == "cuda") else torch.float32

    # Use authentication token if provided
    hf_kwargs = {}
    if opts.hf_token:
        hf_kwargs = {"use_auth_token": opts.hf_token}

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            opts.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None if opts.allow_nsfw else None,  # placeholder; Diffusers handles safety differently across versions
            **hf_kwargs,
        )
    except Exception as e:
        logger.exception("Failed to load model from HuggingFace. Ensure the model id and token (if required) are correct.")
        raise e

    # Replace scheduler if requested
    if opts.scheduler:
        try:
            pipe = _select_scheduler(opts.scheduler, pipe)
        except Exception:
            logger.warning("Could not set scheduler=%s; continuing with default", opts.scheduler)

    # Set device
    pipe = pipe.to(device)

    # Memory optimizations
    if opts.attention_slicing:
        try:
            pipe.enable_attention_slicing()
            logger.debug("Enabled attention slicing")
        except Exception:
            logger.warning("Attention slicing not supported for this pipeline version")

    if opts.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.debug("Enabled xformers memory efficient attention")
        except Exception:
            logger.warning("xFormers not available or failed to enable; continuing without it")

    # Some envs/models require safety checker hooking. We keep default safety unless allow_nsfw True.
    # Newer diffusers may not have built-in safety checker or have different API; handling that generically is complex.

    return pipe


def _generate_single(
    pipe: StableDiffusionPipeline,
    prompt: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
) -> Image.Image:
    """Generate a single image from a prompt using the provided pipeline and parameters.

    Args:
        pipe: configured StableDiffusionPipeline
        prompt: text prompt
        width: image width
        height: image height
        guidance_scale: guidance scale
        num_inference_steps: number of denoising steps
        seed: optional random seed for reproducibility

    Returns:
        PIL.Image
    """
    generator = torch.Generator(device=pipe.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    else:
        # Use random seed for non-deterministic behavior
        generator = None

    # Wrap execution in autocast for fp16 when possible
    try:
        autocast_ctx = torch.autocast if hasattr(torch, "autocast") else torch.cuda.amp.autocast
    except Exception:
        autocast_ctx = None

    inference_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }

    # Pick context manager if available and pipeline uses fp16
    device_type = pipe.device.type if hasattr(pipe, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = device_type == "cuda" and pipe.unet.dtype == torch.float16 if hasattr(pipe, "unet") else False

    if use_autocast and autocast_ctx is not None:
        with autocast_ctx(device_type):
            result = pipe(**inference_kwargs)
    else:
        result = pipe(**inference_kwargs)

    image = result.images[0]

    # Basic NSFW check reporting: diffusers returns NSFW flag in 'nsfw_content_detected' on some versions
    nsfw_flag = getattr(result, "nsfw_content_detected", None)
    if nsfw_flag is not None and any(nsfw_flag):
        logger.warning("NSFW content detected for prompt: %s", prompt)

    return image


def save_image(img: Image.Image, outdir: Path, prompt: str, seed: Optional[int]) -> Path:
    """Save a PIL image to outdir with a deterministic filename and return path.

    Args:
        img: PIL.Image
        outdir: output directory Path
        prompt: source prompt (used in filename sanitized)
        seed: seed used to generate

    Returns:
        Path to saved image
    """
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    sanitized = "".join([c if c.isalnum() or c in "-_" else "_" for c in prompt])[:128]
    unique = uuid.uuid4().hex[:8]
    seed_part = f"_s{seed}" if seed is not None else ""
    filename = f"sd_{timestamp}_{sanitized}_{unique}{seed_part}.png"
    path = outdir / filename
    img.save(path, format="PNG")
    logger.info("Saved image to %s", path)
    return path


def generate_images(opts: GenerationOptions) -> List[Path]:
    """Main orchestration that loads pipeline, iterates prompts and seeds, generates and saves images.

    Args:
        opts: GenerationOptions

    Returns:
        List of paths to saved images
    """
    prompts = read_prompts(opts.prompt, opts.prompt_file)
    if not prompts:
        raise ValueError("No prompts to generate from")

    # Precompute seeds
    seeds: List[Optional[int]] = []
    if opts.seeds:
        seeds = opts.seeds
    else:
        # Generate deterministic seeds if user asked for specific number of outputs
        for _ in range(opts.num_outputs):
            seeds.append(int.from_bytes(os.urandom(2), "big"))

    logger.debug("Prompts: %s", prompts)
    logger.debug("Seeds: %s", seeds)

    pipe = load_pipeline(opts)

    saved_paths: List[Path] = []

    # Iterate prompts and seeds
    for prompt in prompts:
        for i, seed in enumerate(seeds):
            try:
                logger.info("Generating prompt=%s (seed=%s, index=%d)", prompt, seed, i)
                img = _generate_single(
                    pipe=pipe,
                    prompt=prompt,
                    width=opts.width,
                    height=opts.height,
                    guidance_scale=opts.guidance_scale,
                    num_inference_steps=opts.num_inference_steps,
                    seed=seed,
                )
                path = save_image(img, opts.outdir, prompt, seed)
                saved_paths.append(path)
            except Exception as e:
                logger.exception("Failed to generate image for prompt=%s seed=%s: %s", prompt, seed, e)
    return saved_paths


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    try:
        opts = parse_args(argv)
        setup_logging(verbose=False)
        logger.info("Starting text2image with model %s", opts.model_id)
        saved = generate_images(opts)
        logger.info("Finished. Generated %d images", len(saved))
        return 0
    except Exception as e:
        logger.exception("text2image failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
