import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
import numpy as np

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import login as hf_login


LOGGER = logging.getLogger("text2image")


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root and module logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format=fmt)


def _validate_image_dimensions(width: int, height: int) -> None:
    """Validate image dimensions required by most diffusion models (multiples of 8).

    Args:
        width: requested width
        height: requested height

    Raises:
        ValueError: if dims are not multiples of 8 or out-of-range
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("width and height must be multiples of 8 for Stable Diffusion models")
    if width > 2048 or height > 2048:
        # Protect memory blowups
        raise ValueError("width and height must be <= 2048 to avoid enormous memory usage")


def _resolve_device_and_dtype(force_cpu: bool = False) -> Tuple[torch.device, torch.dtype]:
    """Return the best device and dtype to run inference with.

    Args:
        force_cpu: if True, always use CPU.
    Returns:
        device and dtype
    """
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        # float32 is safer on CPU
        dtype = torch.float32
    LOGGER.debug("Selected device=%s dtype=%s", device, dtype)
    return device, dtype


def _login_if_token(hf_token: Optional[str]) -> None:
    """Login to Hugging Face Hub if token provided. This helps load gated models.

    Args:
        hf_token: token string or None
    """
    if hf_token:
        try:
            hf_login(token=hf_token)
            LOGGER.info("Logged into Hugging Face Hub using token")
        except Exception as exc:
            LOGGER.warning("Failed to log in to Hugging Face Hub: %s", exc)


def create_pipeline(model_id: str, device: torch.device, dtype: torch.dtype, hf_token: Optional[str] = None,
                    enable_xformers: bool = False) -> StableDiffusionPipeline:
    """Instantiate and return a StableDiffusion pipeline tuned for inference.

    The pipeline is configured with DPMSolverMultistepScheduler for faster stable convergence.

    Args:
        model_id: Hugging Face model id (eg. "runwayml/stable-diffusion-v1-5")
        device: torch device
        dtype: torch dtype
        hf_token: optional Hugging Face token for private/gated models
        enable_xformers: if True, attempt to enable xformers (if installed) for memory optimization

    Returns:
        Configured StableDiffusionPipeline
    """
    # Ensure auth if provided
    _login_if_token(hf_token)

    LOGGER.info("Loading model '%s' on device=%s dtype=%s", model_id, device, dtype)

    # Load the pipeline
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_auth_token=hf_token if hf_token else None,
        )
    except TypeError:
        # Some diffusers versions accept token= instead of use_auth_token=
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token if hf_token else None,
        )

    # Use a performant scheduler (DPMSolverMultistep)
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception:
        LOGGER.debug("Failed to switch scheduler to DPMSolverMultistepScheduler; using default scheduler")

    # Move to device
    pipe.to(device)

    # Optionally enable xformers memory efficient attention
    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            LOGGER.info("Enabled xformers memory efficient attention")
        except Exception:
            LOGGER.debug("xformers not available or failed to enable")

    # Do NOT disable the safety checker here in production. We're keeping the default.
    return pipe


def generate_images(
    prompt: str,
    output_dir: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    hf_token: Optional[str] = None,
    num_images: int = 1,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    negative_prompt: Optional[str] = None,
    force_cpu: bool = False,
    enable_xformers: bool = False,
) -> List[Path]:
    """Generate image(s) from text using Hugging Face diffusers Stable Diffusion.

    Args:
        prompt: main text prompt describing the scene
        output_dir: directory where images will be written
        model_id: HF model id to use
        hf_token: optional HF token for gated models
        num_images: number of images to generate
        num_inference_steps: diffusion steps (higher -> better quality, slower)
        guidance_scale: classifier-free guidance scale
        seed: optional PRNG seed to make outputs deterministic
        width: output width (must be multiple of 8)
        height: output height (must be multiple of 8)
        negative_prompt: optional negative prompt
        force_cpu: if True, avoid GPU even if available
        enable_xformers: try to enable xformers if present

    Returns:
        list of Paths written

    Raises:
        ValueError: for invalid inputs
        RuntimeError: when generation fails
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    _validate_image_dimensions(width=width, height=height)

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    device, dtype = _resolve_device_and_dtype(force_cpu=force_cpu)

    pipe = create_pipeline(model_id=model_id, device=device, dtype=dtype, hf_token=hf_token,
                           enable_xformers=enable_xformers)

    # Ensure deterministic behavior if seed provided
    generator = None
    if seed is not None:
        if device.type == "cuda":
            generator = torch.Generator(device=device).manual_seed(int(seed))
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        LOGGER.info("Using seed=%s", seed)

    LOGGER.info("Generating %d image(s) with model=%s", num_images, model_id)

    saved_paths: List[Path] = []
    try:
        # The pipeline supports batched generation; do images in a single call if possible
        images = pipe(
            prompt=[prompt] * num_images,
            negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        for i, img in enumerate(images):
            # Attach prompt metadata into PNG info (Pillow limited support)
            file_name = f"sd_{i}.png"
            dest = out_dir_path / file_name

            # Pillow allow simple text info
            metadata = {"prompt": prompt}

            img.save(dest, pnginfo=None)
            LOGGER.info("Saved image %s", dest)
            saved_paths.append(dest)

        return saved_paths

    except Exception as exc:
        LOGGER.exception("Failed to generate images: %s", exc)
        raise RuntimeError(f"Image generation failed: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from text using Hugging Face Diffusers")

    parser.add_argument("--prompt", required=True, help="Text prompt to render into image(s)")
    parser.add_argument("--output_dir", default="outputs", help="Directory to write generated images")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face model id to use")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate (batch)")

    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Diffusion steps (higher => better quality, slower)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Optional PRNG seed for deterministic outputs")
    parser.add_argument("--width", type=int, default=512, help="Output width (multiple of 8)")
    parser.add_argument("--height", type=int, default=512, help="Output height (multiple of 8)")
    parser.add_argument("--negative_prompt", default=None, help="Negative prompt to avoid undesired elements")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--enable_xformers", action="store_true", help="Try to enable xformers for memory savings")

    parser.add_argument("--log_level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    # Prefer env var if provided
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        saved = generate_images(
            prompt=args.prompt,
            output_dir=args.output_dir,
            model_id=args.model_id,
            hf_token=hf_token,
            num_images=args.num_images,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            width=args.width,
            height=args.height,
            negative_prompt=args.negative_prompt,
            force_cpu=args.force_cpu,
            enable_xformers=args.enable_xformers,
        )
        LOGGER.info("Generation completed. Saved files: %s", saved)
    except Exception as exc:
        LOGGER.error("Generation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
