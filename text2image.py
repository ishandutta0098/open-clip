import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
handler.setFormatter(_formatter)
logger.addHandler(handler)


def _validate_size(value: int) -> int:
    """Validate that the image dimension is reasonable and multiple of 8.

    Stable Diffusion requires width/height divisible by 8.

    Args:
        value: dimension value (px)
    Returns:
        The same value if valid.
    Raises:
        argparse.ArgumentTypeError: if invalid.
    """
    if value <= 0:
        raise argparse.ArgumentTypeError("Dimension must be > 0")
    if value % 8 != 0:
        raise argparse.ArgumentTypeError("Dimension must be divisible by 8 for Stable Diffusion (e.g. 512)")
    if value > 2048:
        raise argparse.ArgumentTypeError("Dimension too large, risk of running out of memory. Keep <= 2048")
    return value


def _get_device(prefer_cuda: bool = True) -> torch.device:
    """Select computation device.

    Args:
        prefer_cuda: prefer GPU if available.
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        logger.debug("CUDA available - using GPU")
        return torch.device("cuda")
    logger.debug("CUDA not available or not preferred - using CPU")
    return torch.device("cpu")


def _load_pipeline(model_id: str, device: torch.device, torch_dtype: Optional[torch.dtype], hf_token: Optional[str], enable_safety: bool = True) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline from Hugging Face diffusers.

    This encapsulates safe loading and device placement.

    Args:
        model_id: huggingface model repo id
        device: torch.device to put the pipeline on
        torch_dtype: target dtype (e.g. torch.float16) or None
        hf_token: optional HF token for private models
        enable_safety: whether to keep the safety checker enabled (default True). Some models or workflows may remove it.
    Returns:
        Initialized StableDiffusionPipeline
    """
    logger.info("Loading model: %s", model_id)

    from diffusers import DPMSolverMultistepScheduler

    # Use the DPM solver scheduler by default for speed/quality tradeoff
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_safetensors=True,
        revision="fp16" if torch_dtype == torch.float16 else None,
        torch_dtype=torch_dtype,
        safety_checker=None if not enable_safety else None,  # keep as default (diffusers may set internally)
        local_files_only=False,
        use_auth_token=hf_token,
    )

    # Replace scheduler with a high-quality fast scheduler
    try:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    except Exception:
        logger.debug("Failed to set DPMSolver scheduler; continuing with default scheduler")

    # Move to device
    pipeline.to(device)

    # Performance: enable attention slicing and optional memory-efficient attention
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        logger.debug("enable_attention_slicing not available for this pipeline")

    # If running on GPU with float16 support, enable it
    if device.type == "cuda" and torch_dtype == torch.float16:
        try:
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.debug("xformers not available or failed to enable: %s", e)

    logger.info("Model loaded and moved to device: %s", device)
    return pipeline


def _sanitize_prompt(prompt: str) -> str:
    """Basic prompt sanitation to avoid trivially empty prompts.

    Extend this to implement more advanced safety/validation as needed.

    Args:
        prompt: raw prompt string
    Returns:
        sanitized prompt
    Raises:
        ValueError: when prompt is invalid
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    p = prompt.strip()
    if len(p) > 2000:
        logger.warning("Prompt length exceeds 2000 characters; truncating")
        p = p[:2000]
    return p


def generate(
    prompt: str,
    output: Path,
    model: str = "runwayml/stable-diffusion-v1-5",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    hf_token: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> Path:
    """Generate an image from text using a diffusion model.

    Args:
        prompt: text prompt to condition the image
        output: output file path to save result (PNG/JPEG)
        model: Hugging Face model repo id
        width: output width in pixels (multiple of 8)
        height: output height in pixels (multiple of 8)
        num_inference_steps: number of diffusion steps
        guidance_scale: classifier-free guidance scale
        seed: optional random seed for reproducibility
        device: torch.device to run on; autodetected if None
        hf_token: Hugging Face token for private models or higher rate limits
        torch_dtype: torch.float16 for GPU memory savings, torch.float32 for CPU
    Returns:
        Path to saved image
    """
    if device is None:
        device = _get_device()

    prompt = _sanitize_prompt(prompt)

    # Validate dims
    _validate_size(width)
    _validate_size(height)

    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
        logger.debug("No seed provided - generated random seed: %d", seed)

    if torch_dtype is None:
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipeline = _load_pipeline(model, device, torch_dtype, hf_token)

    # Build generator for reproducibility
    generator = torch.manual_seed(seed) if device.type == "cpu" else torch.Generator(device=device).manual_seed(seed)

    logger.info("Generating image with prompt: %s", prompt)

    # The pipeline call handles the autocast context internally for many diffusers builds;
    # however, we add a manual autocast for additional safety when using float16 on CUDA.
    try:
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                result = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
        else:
            result = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        image = result.images[0]
    except Exception as exc:
        logger.exception("Image generation failed")
        raise RuntimeError("Image generation failed: %s" % exc) from exc

    # Ensure output directory exists
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Atomic-ish save: write to temp file then rename
    tmp_path = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        image.save(tmp_path, format="PNG")
        tmp_path.replace(output)
        logger.info("Saved image to %s", output)
    except Exception as exc:
        logger.exception("Failed to save image")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text→Image generator using Hugging Face diffusers (Stable Diffusion)")
    parser.add_argument("prompt", type=str, help="Text prompt to generate an image from")
    parser.add_argument("--output", "-o", type=Path, default=Path("./out.png"), help="Output image path (PNG)")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="HF model id (default: runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--width", type=_validate_size, default=512, help="Image width (pixels, divisible by 8)")
    parser.add_argument("--height", type=_validate_size, default=512, help="Image height (pixels, divisible by 8)")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps (e.g. 20-50)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (classifier-free guidance)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-cuda", dest="no_cuda", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="Use fp16 where supported (recommended on modern GPUs)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # HF token precedence: CLI -> env
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token is None:
        logger.warning("No Hugging Face token provided. Public models may work but rate limits apply or model access may be denied.")

    # Choose device
    device = _get_device(prefer_cuda=not args.no_cuda)
    torch_dtype = torch.float16 if (device.type == "cuda" and args.fp16) else torch.float32

    try:
        out_path = generate(
            prompt=args.prompt,
            output=args.output,
            model=args.model,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            device=device,
            hf_token=hf_token,
            torch_dtype=torch_dtype,
        )
        logger.info("Done. Output saved to %s", out_path)
    except Exception as exc:
        logger.error("Failed to generate image: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
