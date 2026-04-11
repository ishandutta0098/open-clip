import os
import logging
import math
import warnings
from functools import lru_cache
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class Text2ImageConfig:
    """
    Configuration for the text2image pipeline.

    Attributes:
        model_id: Hugging Face Diffusers model id to load.
        device: Device to move pipeline to. If None, device will be auto-detected.
        cache_dir: HF cache directory to use when downloading weights.
        enable_xformers: Whether to try to enable xformers for memory-efficient attention.
        torch_dtype: dtype used when loading weights (e.g. torch.float16)
    """

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    enable_xformers: bool = False
    torch_dtype: Optional[torch.dtype] = None


# Internal cached pipeline so we don't reload model repeatedly
_cached_pipeline: Optional[StableDiffusionPipeline] = None
_cached_config: Optional[Text2ImageConfig] = None


def _auto_device() -> str:
    """Detect the best device to use (cuda if available otherwise cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _validate_dimensions(width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    """Validate and sanitize width/height, returning defaults if needed.

    Stable Diffusion typically expects multiples of 8 and reasonable bounds.
    """
    default_w, default_h = 512, 512
    min_size, max_size = 64, 2048

    if width is None and height is None:
        return default_w, default_h

    if width is None:
        width = default_w
    if height is None:
        height = default_h

    if not (isinstance(width, int) and isinstance(height, int)):
        raise ValueError("width and height must be integers")

    if width < min_size or height < min_size or width > max_size or height > max_size:
        raise ValueError(f"width and height must be between {min_size} and {max_size}")

    # Make sure they are multiples of 8
    def _round8(x: int) -> int:
        return int(math.ceil(x / 8.0) * 8)

    width = _round8(width)
    height = _round8(height)

    return width, height


def _validate_steps(steps: int) -> int:
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer")
    if steps > 200:
        warnings.warn("num_inference_steps > 200 may be slow and not significantly improve quality")
    return steps


def _validate_guidance(guidance: float) -> float:
    if guidance < 1.0:
        raise ValueError("guidance_scale (classifier-free guidance) should be >= 1.0")
    if guidance > 30.0:
        warnings.warn("guidance_scale > 30 may produce unexpected artifacts")
    return guidance


def _get_hf_token() -> Optional[str]:
    """Securely fetch Hugging Face token from environment.

    This avoids hard-coding tokens in source code.
    """
    return os.getenv("HF_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def load_pipeline(config: Text2ImageConfig) -> StableDiffusionPipeline:
    """
    Load and prepare the Stable Diffusion pipeline with sensible defaults and performance options.

    This function caches the pipeline instance in-memory to avoid repeated downloads and cold starts.

    Args:
        config: Text2ImageConfig containing model and runtime options.

    Returns:
        a prepared StableDiffusionPipeline instance

    Raises:
        RuntimeError: If the model cannot be loaded or device memory is insufficient.
    """
    global _cached_pipeline, _cached_config

    # If we already loaded a pipeline with the same config, return it
    if _cached_pipeline is not None and _cached_config == config:
        logger.debug("Reusing cached pipeline")
        return _cached_pipeline

    model_id = config.model_id
    device = config.device or _auto_device()
    cache_dir = config.cache_dir
    torch_dtype = config.torch_dtype

    hf_token = _get_hf_token()

    logger.info("Loading model %s to device=%s cache_dir=%s", model_id, device, cache_dir)

    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            use_safetensors=True,
            revision=None,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Keep it optional for deployments; set to None to avoid extra deps
            use_auth_token=hf_token,
        )
    except Exception as exc:
        logger.exception("Failed to load model %s: %s", model_id, exc)
        raise RuntimeError(f"Failed to load model {model_id}: {exc}")

    # Move to device
    try:
        pipeline = pipeline.to(device)
    except Exception as exc:
        logger.exception("Failed to move pipeline to device %s: %s", device, exc)
        # fallback: keep on cpu if GPU move fails
        pipeline = pipeline.to("cpu")
        device = "cpu"

    # Performance options
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        logger.debug("enable_attention_slicing not supported for this pipeline")

    if config.enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception:
            logger.warning("xformers is not available or failed to enable; continuing without it")

    # Cache
    _cached_pipeline = pipeline
    _cached_config = config

    logger.info("Model loaded and ready")
    return pipeline


def generate_image(
    prompt: str,
    config: Optional[Text2ImageConfig] = None,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Image.Image:
    """
    Generate an image from a text prompt using HuggingFace Diffusers Stable Diffusion.

    Args:
        prompt: Text prompt describing the desired image. Required and must be non-empty.
        config: Optional Text2ImageConfig (model selection and runtime settings). If None, defaults will be used.
        negative_prompt: Optional text to discourage certain features.
        num_inference_steps: Number of denoising steps. Typical values: 20-50.
        guidance_scale: Classifier-free guidance scale (>=1.0). Higher values means stronger adherence to prompt.
        seed: Optional random seed for reproducibility. If None, nondeterministic behavior occurs.
        width: Target width in pixels (multiple of 8). Defaults to 512.
        height: Target height in pixels (multiple of 8). Defaults to 512.

    Returns:
        A PIL.Image instance with the generated result.

    Raises:
        ValueError: if inputs are invalid.
        RuntimeError: on model or runtime errors (OOM, missing model, etc.).
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt must be a non-empty string")

    cfg = config or Text2ImageConfig()
    pipeline = load_pipeline(cfg)

    width, height = _validate_dimensions(width, height)
    num_inference_steps = _validate_steps(num_inference_steps)
    guidance_scale = _validate_guidance(guidance_scale)

    device = cfg.device or _auto_device()

    # Seed handling for reproducibility
    generator = None
    if seed is not None:
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    logger.info(
        "Generating image with prompt='%s' steps=%d guidance=%.2f size=%dx%d seed=%s",
        (prompt if len(prompt) < 200 else prompt[:200] + "..."),
        num_inference_steps,
        guidance_scale,
        width,
        height,
        str(seed),
    )

    try:
        # The pipeline API expects width/height keys depending on model; we pass them explicitly
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
    except RuntimeError as exc:
        # Common case: OOM or CUDA error
        logger.exception("Runtime error while generating image: %s", exc)
        # Provide actionable suggestions
        if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
            raise RuntimeError(
                "RuntimeError during inference (likely OOM). Try reducing width/height, lowering batch size, enabling attention_slicing, use CPU or a bigger GPU. Original error: "
                + str(exc)
            )
        raise
    except Exception as exc:
        logger.exception("Unexpected error while generating image: %s", exc)
        raise

    images = output.images if hasattr(output, "images") else output
    if isinstance(images, list):
        image = images[0]
    elif isinstance(images, Image.Image):
        image = images
    else:
        raise RuntimeError("Unexpected output type from pipeline: %s" % type(images))

    return image


def save_image(image: Image.Image, output_path: str, fmt: Optional[str] = None) -> str:
    """
    Save a PIL.Image to disk, creating parent directories if necessary.

    Args:
        image: PIL image to save
        output_path: path where the image will be saved
        fmt: optional format string (e.g. "PNG", "JPEG"). If None the format is inferred from file extension.

    Returns:
        The absolute path to the saved file.
    """
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("output_path must be a non-empty string")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image.save(output_path, format=fmt)
    logger.info("Saved image to %s", output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    # Simple CLI for quick testing without extra deps (argparse is stdlib)
    import argparse

    parser = argparse.ArgumentParser(description="Text2Image using HuggingFace Diffusers")
    parser.add_argument("prompt", type=str, help="Prompt describing the desired image")
    parser.add_argument("--output", type=str, default="out.png", help="Output image path")
    parser.add_argument("--model", type=str, default=None, help="HF model id (diffusers)")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--enable-xformers", action="store_true", help="Attempt to enable xformers for memory savings")

    args = parser.parse_args()

    config = Text2ImageConfig(
        model_id=args.model or Text2ImageConfig().model_id,
        device=("cpu" if args.cpu else None),
        enable_xformers=args.enable_xformers,
    )

    try:
        img = generate_image(
            prompt=args.prompt,
            config=config,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
        save_image(img, args.output)
        print(f"Saved generated image to {os.path.abspath(args.output)}")
    except Exception as e:
        logger.exception("Failed to generate image: %s", e)
        raise
