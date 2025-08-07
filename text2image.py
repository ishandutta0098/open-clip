import argparse
import logging
import os
import sys
from typing import List, Optional

import torch
from PIL import Image  # Required for image I/O

try:
    from diffusers import StableDiffusionPipeline
except Exception as exc:  # pragma: no cover - friendly error for missing dependencies
    raise SystemExit(
        "Dependency error: diffusers is not installed. Install via 'pip install diffusers[torch]'."
    ) from exc


def _setup_logger() -> None:
    """Configure a simple logger for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("text2image").setLevel(logging.INFO)


def _load_pipeline(model_id: str, device: str, dtype: torch.dtype) -> "StableDiffusionPipeline":
    """Load the diffusion pipeline from Hugging Face hub.

    This function attempts to load with the provided dtype and disables the safety checker
    for environments where safety checker integration is not desired.
    """
    logging.info("Loading model '%s' on device '%s' with dtype %s", model_id, device, dtype)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
    except TypeError:
        # Fallback for older diffusers versions that may not accept safety_checker arg
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
    pipe = pipe.to(device)
    return pipe


def _generate_images(
    pipe: "StableDiffusionPipeline",
    prompts: List[str],
    output_dir: str,
    seed: Optional[int],
    num_images: int,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
) -> List[str]:
    """Generate images from prompts using the provided pipeline.

    Args:
        pipe: The loaded StableDiffusionPipeline.
        prompts: List of prompts to render; will cycle if fewer prompts than images.
        output_dir: Directory to save generated images.
        seed: Optional seed for reproducibility.
        num_images: Number of images to generate.
        steps: Number of diffusion steps for generation.
        guidance_scale: CFG guidance scale.
        width: Image width in pixels (must be multiple of 8 for most models).
        height: Image height in pixels (must be multiple of 8 for most models).

    Returns:
        List of file paths to the saved images.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    for i in range(num_images):
        prompt = prompts[i % len(prompts)]
        generator = None  # Optional RNG seed for reproducibility per image
        if seed is not None:
            generator = torch.Generator(device=pipe.device)
            generator.manual_seed(seed + i)
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        out_path = os.path.join(output_dir, f"image_{i+1:04d}.png")
        image.save(out_path)
        saved_paths.append(out_path)
        logging.info("Saved image %s", out_path)

    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-to-image generation using HuggingFace Diffusers."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model identifier from HuggingFace hub.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale (CFG) for image generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (multiples of 8).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (multiples of 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--dtype",
        dest="dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Floating point precision to use during inference.",
    )
    return parser.parse_args()


def _choose_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available. Falling back to CPU.")
        return "cpu"
    if choice == "cpu" and torch.cuda.is_available():
        logging.info("CUDA is available but CPU was requested. Using CPU anyway.")
    return choice


def main() -> int:
    _setup_logger()
    args = _parse_args()

    device = _choose_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    try:
        pipe = _load_pipeline(args.model_id, device, dtype)
    except Exception as exc:
        logging.exception("Failed to load model: %s", exc)
        return 1

    prompts = [args.prompt]

    try:
        _generate_images(
            pipe=pipe,
            prompts=prompts,
            output_dir=args.output_dir,
            seed=args.seed,
            num_images=args.num_images,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
        )
    except Exception as exc:
        logging.exception("Image generation failed: %s", exc)
        return 2

    logging.info("Generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
