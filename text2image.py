import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging as diffusers_logging

# Reduce diffusers logging verbosity; we still log important info via our logger
diffusers_logging.set_verbosity_error()


LOGGER = logging.getLogger("text2image")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root and module loggers.

    Args:
        level: Logging level (e.g., logging.INFO)
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )
    root = logging.getLogger()
    if root.handlers:
        root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)
    LOGGER.setLevel(level)


def sanitize_filename(filename: str) -> str:
    """Return a safe filename by removing unsafe characters.

    This is intentionally conservative. It will replace disallowed characters with
    underscore. This avoids directory traversal and other filesystem surprises.
    """
    filename = filename.strip()
    # Remove path separators and keep alphanumerics, dash, underscore, space, and dot
    filename = re.sub(r"[^A-Za-z0-9._ \-]", "_", filename)
    # Collapse repeated underscores/spaces
    filename = re.sub(r"[ \_]+", "_", filename)
    return filename


def validate_image_dimensions(width: int, height: int) -> None:
    """Validate that the width and height are acceptable for Stable Diffusion.

    Stable Diffusion models typically expect dimensions divisible by 8 or 64 depending
    on model. We enforce divisible-by-8 here which is safe for common models.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("Width and height must be divisible by 8 for model compatibility")
    if width * height > 1024 * 1024 * 3:
        # crude heuristic to avoid huge allocations (3MP)
        LOGGER.warning("Requested output resolution is large and may exceed available memory")


def choose_device(preferred: Optional[str] = None) -> str:
    """Choose a device string for torch based on availability and preference.

    Preference order: explicit arg -> cuda if available -> mps (Apple) -> cpu
    """
    if preferred:
        # Basic validation
        if preferred not in ("cpu", "cuda", "mps"):
            raise ValueError("Invalid device choice. Choose from 'cpu', 'cuda', or 'mps'.")
        if preferred == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        if preferred == "mps" and not getattr(torch, "has_mps", False):
            LOGGER.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "has_mps", False):
        return "mps"
    return "cpu"


def load_pipeline(
    model_id: str, device: str, use_auth_token: Optional[str], torch_dtype: Optional[torch.dtype]
) -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline with reasonable defaults.

    Args:
        model_id: Hugging Face Hub model id (e.g., "runwayml/stable-diffusion-v1-5").
        device: Device string for .to(device).
        use_auth_token: Optional HF token for private or gated models.
        torch_dtype: dtype to load model in (e.g., torch.float16) or None.

    Returns:
        Instantiated StableDiffusionPipeline loaded on the requested device.
    """
    # Keep the import local to speed up CLI responsiveness when listing help
    from diffusers import DPMSolverMultistepScheduler

    try:
        LOGGER.info("Loading model '%s' on device '%s'", model_id, device)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_auth_token=use_auth_token,
        )

        # Use a more performant scheduler if available
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            LOGGER.debug("Could not replace scheduler with DPMSolver; using default scheduler")

        # Move to desired device
        pipe.to(device)

        # Enable memory-efficient attention if available
        try:
            pipe.enable_attention_slicing()
        except Exception:
            LOGGER.debug("enable_attention_slicing not available for this pipeline")

        # If running on CUDA, enable CUDA graph or half precision where appropriate
        return pipe
    except Exception as exc:
        LOGGER.exception("Failed to load pipeline: %s", exc)
        raise


def generate_images(
    prompt: str,
    out_dir: Path,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    num_images: int = 1,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    device_choice: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Tuple[Path, list]:
    """Generate images from text using diffusers.

    Args:
        prompt: The text prompt.
        out_dir: Directory to write images to (will be created if missing).
        model_id: Hugging Face model id to use.
        num_images: How many images to produce.
        height: Output image height (px). Must be divisible by 8.
        width: Output image width (px). Must be divisible by 8.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Optional random seed for reproducibility.
        device_choice: Preferred compute device (cpu, cuda, mps) or None to auto-select.
        hf_token: Optional Hugging Face token for private models.

    Returns:
        Tuple of (output directory path, list of saved image paths)
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    if num_images <= 0 or num_images > 16:
        raise ValueError("num_images must be between 1 and 16")

    validate_image_dimensions(width=width, height=height)

    device = choose_device(device_choice)

    # Choose dtype for model loading
    torch_dtype = None
    if device == "cuda":
        # Use fp16 on GPUs for better throughput and memory savings
        torch_dtype = torch.float16
    elif device == "mps":
        # MPS currently works best with float32
        torch_dtype = torch.float32

    pipe = load_pipeline(model_id=model_id, device=device, use_auth_token=hf_token, torch_dtype=torch_dtype)

    # Prepare generator for reproducibility if seed provided
    generator = None
    if seed is not None:
        # Generator should be on the same device
        gen_device = device if device != "mps" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    try:
        LOGGER.info("Generating %d image(s)", num_images)

        for i in range(num_images):
            LOGGER.debug("Generating image %d/%d", i + 1, num_images)
            # The pipeline returns PIL images
            with torch.autocast(device) if device == "cuda" else torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )

            images = result.images
            if not images:
                raise RuntimeError("No image returned by the pipeline")

            # Save image(s)
            for j, img in enumerate(images):
                safe_prompt = sanitize_filename(prompt)[:80] or "image"
                base_name = f"{safe_prompt}_{seed or 'r'}_{i}_{j}.png"
                out_path = out_dir / base_name
                # Prevent writing outside the working directory for security (basic check)
                out_path = out_path.resolve()
                cwd = Path.cwd().resolve()
                if not str(out_path).startswith(str(cwd)):
                    raise RuntimeError("Refusing to write outside the current working directory")

                img.save(out_path, format="PNG")
                saved_paths.append(out_path)
                LOGGER.info("Saved image to %s", out_path)

    except Exception:
        LOGGER.exception("Failed to generate images")
        raise

    return out_dir, saved_paths


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text2Image CLI using HuggingFace Diffusers (Stable Diffusion)"
    )

    parser.add_argument("--prompt", required=True, help="Text prompt to generate the image(s)")
    parser.add_argument(
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model id (default: runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument("--out", default="outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate (1-16)")
    parser.add_argument("--height", type=int, default=512, help="Image height in px (divisible by 8)")
    parser.add_argument("--width", type=int, default=512, help="Image width in px (divisible by 8)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Preferred device; auto-detect if omitted",
    )
    parser.add_argument("--hf_token", default=None, help="Hugging Face token for gated models (or set HF_TOKEN env var)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    out_dir = Path(args.out)

    try:
        _, saved = generate_images(
            prompt=args.prompt,
            out_dir=out_dir,
            model_id=args.model,
            num_images=args.num_images,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            device_choice=args.device,
            hf_token=hf_token,
        )
        LOGGER.info("Generation complete. %d images saved.", len(saved))
        return 0
    except Exception as exc:
        LOGGER.error("Error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
