import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

try:
    # Stable Diffusion pipeline
    from diffusers import StableDiffusionPipeline
except Exception as e:  # pragma: no cover - informative import error handling
    raise ImportError(
        "diffusers is required to run this script. Install with `pip install diffusers`"
    ) from e


# Configure module-level logger
logger = logging.getLogger("text2image")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _validate_prompt(prompt: str) -> str:
    """
    Validate and sanitize a user-provided prompt.

    Args:
        prompt: Raw user input prompt.

    Returns:
        Sanitized prompt string.

    Raises:
        ValueError: If prompt is empty or suspiciously long.
    """
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string")
    prompt = prompt.strip()
    if len(prompt) == 0:
        raise ValueError("Prompt cannot be empty")
    # Prevent extremely long prompts that could cause unexpected behavior
    if len(prompt) > 2000:
        raise ValueError("Prompt too long (max 2000 characters)")
    return prompt


def _get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Determine the torch device to use.

    If the user requests 'cpu' it will be used. If 'cuda' is requested but not
    available, we fall back to CPU with a warning.

    Args:
        preferred: Optional user preference, e.g., 'cuda' or 'cpu'.

    Returns:
        torch.device instance.
    """
    if preferred:
        pref = preferred.lower()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    # Default smart choice
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unique_filename(prefix: str, seed: Optional[int]) -> str:
    """
    Create a unique filename using timestamp + seed hash.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    seed_str = str(seed) if seed is not None else "none"
    hashed = hashlib.sha1((seed_str + ts).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{ts}_{hashed}.png"


def load_pipeline(
    model_id: str,
    device: torch.device,
    hf_token: Optional[str] = None,
    use_safety_checker: bool = True,
    dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
) -> StableDiffusionPipeline:
    """
    Load the Stable Diffusion pipeline from Hugging Face diffusers.

    Args:
        model_id: Hugging Face model repo id.
        device: torch device to move the pipeline to.
        hf_token: Optional HF token for access to gated models.
        use_safety_checker: Whether to enable the built-in safety checker.
        dtype: torch dtype for model weights, e.g., torch.float16 for mixed precision.
        cache_dir: Optional cache directory for model files.

    Returns:
        Initialized StableDiffusionPipeline.
    """
    logger.info("Loading model %s on device %s", model_id, device)
    try:
        # Important: use_auth_token param is accepted in older/newer APIs; pass None if not provided
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            revision=None,
            use_auth_token=hf_token,
            cache_dir=cache_dir,
        )
    except TypeError:
        # Some diffusers versions don't accept use_auth_token kwarg name
        kwargs = {"torch_dtype": dtype, "cache_dir": cache_dir}
        if hf_token is not None:
            kwargs["use_auth_token"] = hf_token
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

    # Move to device
    pipe = pipe.to(device)

    # Performance: enable memory-efficient settings when possible
    try:
        pipe.enable_attention_slicing()
    except Exception:
        logger.debug("enable_attention_slicing not available for this pipeline version")

    # Try to enable xformers efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        logger.debug("xformers not available or failed to enable")

    if not use_safety_checker:
        # Disable the safety checker with a warning
        logger.warning(
            "Safety checker has been disabled. Generated images may contain unsafe content."
        )
        try:
            pipe.safety_checker = None  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Could not disable safety checker for this pipeline")

    return pipe


def generate_image(
    prompt: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    output_dir: str = "outputs",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    device_name: Optional[str] = None,
    hf_token: Optional[str] = None,
    mixed_precision: bool = True,
    cache_dir: Optional[str] = None,
) -> Path:
    """
    Generate an image for a given text prompt using a Stable Diffusion pipeline.

    Args:
        prompt: Text prompt describing desired image.
        model_id: Hugging Face model repo id for the pipeline.
        output_dir: Directory where generated image will be saved.
        num_inference_steps: Number of denoising steps (tradeoff quality/time).
        guidance_scale: Classifier-free guidance scale.
        width: Output image width in pixels.
        height: Output image height in pixels.
        seed: Optional random seed for deterministic outputs.
        device_name: Optional device override ('cuda' or 'cpu').
        hf_token: Optional Hugging Face token for gated models.
        mixed_precision: Use float16 on CUDA for faster generation.
        cache_dir: Optional cache dir for huggingface models.

    Returns:
        Path to the saved image file.

    Raises:
        RuntimeError: On pipeline or generation failures.
    """
    prompt = _validate_prompt(prompt)
    device = _get_device(device_name)

    dtype = torch.float16 if (mixed_precision and device.type == "cuda") else torch.float32

    try:
        pipe = load_pipeline(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            use_safety_checker=True,
            dtype=dtype,
            cache_dir=cache_dir,
        )
    except Exception as e:
        logger.exception("Failed to load pipeline: %s", e)
        raise RuntimeError("Model loading failed") from e

    # Prepare generator for deterministic output if seed provided
    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
        except Exception:
            # If device-specific generator fails, use CPU generator as a fallback
            generator = torch.Generator(device="cpu").manual_seed(seed)

    # Validate image size moderately to prevent abuse
    if not (64 <= width <= 2048 and 64 <= height <= 2048):
        raise ValueError("Width and height must be between 64 and 2048")

    # Ensure output dir exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating image with prompt=%s model=%s steps=%d guidance=%.2f size=%dx%d",
        (prompt if len(prompt) <= 120 else prompt[:117] + "..."),
        model_id,
        num_inference_steps,
        guidance_scale,
        width,
        height,
    )

    # Run inference
    start = time.time()
    try:
        # Use autocast for mixed precision on CUDA
        if device.type == "cuda" and dtype == torch.float16:
            autocast = torch.cuda.amp.autocast
        else:
            # no-op context manager
            from contextlib import nullcontext

            autocast = nullcontext

        with autocast():
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        # result can be an Image or dict depending on pipeline version
        if hasattr(result, "images"):
            images = result.images
        elif isinstance(result, dict) and "images" in result:
            images = result["images"]
        else:
            # Try to treat result as single PIL image
            if isinstance(result, Image.Image):
                images = [result]
            else:
                raise RuntimeError("Unexpected pipeline output format")

        # Save first image (expandable to multi-image saving if required)
        filename = _unique_filename("sd", seed)
        save_path = out_dir / filename
        images[0].save(save_path)

        elapsed = time.time() - start
        logger.info("Image saved to %s (%.2fs)", str(save_path), elapsed)
        return save_path

    except Exception as e:
        logger.exception("Image generation failed: %s", e)
        raise RuntimeError("Image generation failed") from e


def _parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using Hugging Face Diffusers"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt to generate an image for"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model id (default: runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps (higher -> slower, often better)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (higher increases adherence to prompt)",
    )
    parser.add_argument("--width", type=int, default=512, help="Output width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Output height in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto detect)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face token or set HF_TOKEN/HUGGINGFACE_TOKEN in env",
    )
    parser.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
        help="Disable FP16 mixed precision even if CUDA is available",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HF cache directory for model artifacts",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Basic security note for automated environments
    if args.hf_token is None:
        logger.info(
            "No Hugging Face token provided. Public models will load without a token; gated models require a token."
        )

    try:
        out_path = generate_image(
            prompt=args.prompt,
            model_id=args.model,
            output_dir=args.output_dir,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
            device_name=args.device,
            hf_token=args.hf_token,
            mixed_precision=args.mixed_precision,
            cache_dir=args.cache_dir,
        )
        logger.info("Finished successfully. Output: %s", out_path)
        return 0
    except Exception as exc:  # pragma: no cover - top-level runtime handling
        logger.error("Failed to generate image: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
