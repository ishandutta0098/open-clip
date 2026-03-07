import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


@dataclass
class GenerationConfig:
    """Configuration for text-to-image generation.

    Attributes:
        prompt: The text prompt to generate images for.
        model_id: The Hugging Face Diffusers model identifier.
        out_dir: Output directory where images will be written.
        height: Image height in pixels. Must be divisible by 8.
        width: Image width in pixels. Must be divisible by 8.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        num_images: Number of images to generate for the prompt.
        seed: Optional random seed for reproducibility.
        device: Device string ("cpu" or "cuda").
        use_auth_token: Optional HF token to download gated models.
    """

    prompt: str
    model_id: str = "runwayml/stable-diffusion-v1-5"
    out_dir: str = "outputs"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_auth_token: Optional[str] = None


def validate_config(cfg: GenerationConfig) -> None:
    """Validate the generation configuration and raise ValueError on invalid input.

    Ensures sizes are reasonable and inputs safe. Keep validations conservative to
    avoid OOM and long-running jobs.
    """
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    if cfg.height <= 0 or cfg.width <= 0:
        raise ValueError("Image dimensions must be positive")

    if cfg.height % 8 != 0 or cfg.width % 8 != 0:
        raise ValueError("Image width and height must be divisible by 8")

    # Keep a reasonable upper bound to avoid accidental OOM in production
    if cfg.height * cfg.width > 1024 * 1024:  # e.g., > 1 megapixel
        logger.warning(
            "Requested image resolution is large and may run out of GPU memory"
        )

    if cfg.num_inference_steps <= 0 or cfg.num_inference_steps > 200:
        raise ValueError("num_inference_steps must be between 1 and 200")

    if cfg.num_images <= 0 or cfg.num_images > 16:
        raise ValueError("num_images must be between 1 and 16")

    if cfg.device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")


def prepare_pipeline(model_id: str, device: str, use_auth_token: Optional[str] = None) -> StableDiffusionPipeline:
    """Load and prepare the Stable Diffusion pipeline.

    This function loads the diffusers pipeline with recommended settings. If a CUDA device is available
    it will use mixed precision to reduce memory usage. Use_auth_token is passed through for gated models.

    Args:
        model_id: HF model repo id.
        device: 'cuda' or 'cpu'.
        use_auth_token: Optional HF token for private or gated models.

    Returns:
        A ready-to-use StableDiffusionPipeline.
    """
    logger.info("Loading model '%s' on device=%s", model_id, device)

    # Choose dtype for GPU inference to reduce memory usage
    torch_dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_auth_token=use_auth_token,
        )
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        raise

    # Move to device
    pipeline = pipeline.to(device)

    # Disable NSFW safety checker if not present or if you want to manage filtering
    # For newer diffusers versions the safety checker may be optional / removed.
    # We will not remove it programmatically here; rely on upstream defaults.

    # Configure progress bar off by default for programmatic usage
    pipeline.set_progress_bar_config(disable=True)

    logger.info("Model loaded and pipeline prepared")
    return pipeline


def _seed_generator(seed: Optional[int], device: str) -> torch.Generator:
    """Create a torch Generator optionally seeded for reproducibility.

    Args:
        seed: Optional integer seed.
        device: 'cuda' or 'cpu'.

    Returns:
        A torch.Generator object.
    """
    generator = torch.Generator(device=device)
    if seed is None:
        # Use non-deterministic seed
        seed = int.from_bytes(os.urandom(2), "big")
        logger.debug("No seed provided; using random seed=%s", seed)
    else:
        logger.debug("Using provided seed=%s", seed)

    generator.manual_seed(seed)
    return generator


def generate_images(cfg: GenerationConfig) -> List[Path]:
    """Generate images from a text prompt using Stable Diffusion.

    Args:
        cfg: GenerationConfig instance containing generation parameters.

    Returns:
        A list of pathlib.Path pointing to the saved images.
    """
    validate_config(cfg)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = prepare_pipeline(cfg.model_id, cfg.device, cfg.use_auth_token)

    # Prepare deterministic generator
    generator = _seed_generator(cfg.seed, cfg.device)

    saved_paths: List[Path] = []

    # The pipeline returns a list of PIL images for batch sizes > 1
    try:
        with torch.autocast(cfg.device) if cfg.device == "cuda" else torch.no_grad():
            logger.info(
                "Generating %d image(s): prompt='%s' (height=%d,width=%d, steps=%d, guidance=%.2f)",
                cfg.num_images,
                (cfg.prompt[:120] + "...") if len(cfg.prompt) > 120 else cfg.prompt,
                cfg.height,
                cfg.width,
                cfg.num_inference_steps,
                cfg.guidance_scale,
            )

            images = pipeline(
                prompt=cfg.prompt,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                num_images_per_prompt=cfg.num_images,
                generator=generator,
            ).images

            # Save images to disk
            for i, img in enumerate(images):
                safe_prompt = "".join(c for c in cfg.prompt if c.isalnum() or c in (" ", "-", "_"))[:120].strip()
                filename = f"img_{i:03d}__{safe_prompt.replace(' ', '_') or 'image'}.png"
                out_path = out_dir / filename

                # Convert to RGB if needed and save with quality/safety
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img.save(out_path, format="PNG")
                saved_paths.append(out_path)
                logger.info("Saved image: %s", out_path)

    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        raise

    return saved_paths


def _parse_args(argv: Optional[List[str]] = None) -> GenerationConfig:
    """Parse CLI args into a GenerationConfig object.

    Args:
        argv: Optional list of arguments (for testing). If None, reads from sys.argv.

    Returns:
        GenerationConfig instance.
    """
    parser = argparse.ArgumentParser(
        prog="text2image",
        description="Generate images from text prompts using Hugging Face Diffusers Stable Diffusion",
    )

    parser.add_argument("prompt", type=str, help="Text prompt to render")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id for the diffusers model")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory to save generated images")
    parser.add_argument("--height", type=int, default=512, help="Image height (must be divisible by 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width (must be divisible by 8)")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None, help="Optional integer seed for reproducibility")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to run on: 'cpu' or 'cuda'")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"), help="Hugging Face access token or set HF_TOKEN env var")

    args = parser.parse_args(argv)

    return GenerationConfig(
        prompt=args.prompt,
        model_id=args.model_id,
        out_dir=args.out_dir,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        num_images=args.num_images,
        seed=args.seed,
        device=args.device,
        use_auth_token=args.hf_token,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns exit code (0 success, non-zero failure)."""
    try:
        cfg = _parse_args(argv)
        logger.info("Starting generation with config: %s", cfg)
        paths = generate_images(cfg)
        logger.info("Generation finished. %d images saved to %s", len(paths), cfg.out_dir)
        return 0
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
