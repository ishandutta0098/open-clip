#!/usr/bin/env python3
"""text2image.py: A production-grade CLI tool to generate images from text prompts using HuggingFace Diffusers.

Features:
- Lightweight, dependency-conscious design with clear separation of concerns
- GPU-accelerated path (FP16) when available; CPU fallback with reasonable defaults
- Deterministic output via seeds for reproducible results
- Robust error handling and logging
- Output directory management with deterministic file naming
- Environment-based HF Hub token support (HF_HUB_TOKEN or HF_TOKEN)

This script is designed for ease of integration in larger systems (e.g., CI jobs, server apps) while remaining approachable as a standalone CLI tool.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import diffusers. Ensure you have diffusers installed. Run: pip install diffusers[torch]\n"
        f"Original error: {e}"
    )


# Module-level logger for consistent logging across the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger(__name__)


class Text2ImageGenerator:
    """Encapsulates loading a Stable Diffusion pipeline and generating images.

    This class is intentionally lightweight and focused on production readiness:
    - Robust error handling when loading the model
    - GPU-accelerated path with FP16 where available
    - Safe defaults for device selection and token handling
    - Simple API to generate one or multiple images from a prompt
    """

    def __init__(self, model_id: str, device: Optional[str] = None, token: Optional[str] = None):
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.token = token or os.environ.get("HF_HUB_TOKEN") or None
        self.pipe = self._load_pipeline()

    @staticmethod
    def _resolve_device(requested: Optional[str]) -> str:
        if requested in {"cpu", "cuda"}:
            return requested
        # Auto-detect: CUDA if available, else CPU
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_pipeline(self) -> StableDiffusionPipeline:
        """Load the Stable Diffusion pipeline with memory-aware defaults."""
        try:
            if self.device == "cuda":
                LOGGER.info("Loading pipeline on CUDA with FP16 for memory efficiency.")
                pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    use_auth_token=self.token,
                )
            else:
                LOGGER.info("Loading pipeline on CPU/CPU-compatible device with FP32.")
                pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    use_auth_token=self.token,
                )
            pipe = pipe.to(self.device)
            LOGGER.info("Model loaded successfully: %s", self.model_id)
            return pipe
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load model '{self.model_id}': {exc}") from exc

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> List[Image.Image]:
        """Generate one or more images from a text prompt.

        Args:
            prompt: Primary text description for image generation.
            negative_prompt: Optional negative prompt to steer away from undesired features.
            steps: Inference steps; higher is slower but potentially higher quality.
            guidance_scale: CFG scale controlling the strength of the prompt guidance.
            seed: Optional seed for deterministic results.
            num_images: How many images to generate.

        Returns:
            List of PIL.Image.Image objects.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("'prompt' must be a non-empty string.")
        if steps <= 0:
            raise ValueError("'steps' must be a positive integer.")
        if num_images <= 0:
            raise ValueError("'num_images' must be a positive integer.")

        images: List[Image.Image] = []

        # Seed handling for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            try:
                import numpy as np
                np.random.seed(seed)
            except Exception:  # pragma: no cover
                pass

        for i in range(num_images):
            current_seed = seed + i if seed is not None else None
            if current_seed is not None:
                torch.manual_seed(current_seed)
                try:
                    import numpy as np
                    np.random.seed(current_seed)
                except Exception:  # pragma: no cover
                    pass

            with torch.no_grad():
                result = self.pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                )
            images.append(result.images[0])
        return images


def save_images(images: List[Image.Image], output_dir: str, base_filename: str = "image") -> List[str]:
    """Persist generated images to disk with deterministic naming."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for idx, img in enumerate(images, start=1):
        filename = f"{base_filename}_{timestamp}_{idx:04d}.png"
        path = os.path.join(output_dir, filename)
        img.save(path)
        saved_paths.append(path)
    return saved_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="text2image",
        description="Generate images from text prompts using HuggingFace Diffusers (Stable Diffusion).",
        epilog="Example: python text2image.py --prompt 'a futuristic city skyline at sunset' --num-images 2 --steps 60 --output-dir ./outputs",
    )

    parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-2-1-base",
                        help="Pretrained model id from HuggingFace Hub.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate image(s).")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt.")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (quality/speed trade-off).")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG scale for guidance strength.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic outputs.")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save generated images.")
    parser.add_argument("--base-filename", type=str, default="image", help="Base filename pattern for saved images.")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace Hub token (optional).")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                        help="Computation device: 'auto' (detect), 'cpu', or 'cuda'.")

    args = parser.parse_args()

    LOGGER.info("Starting text2image with model=%s on device=%s", args.model_id, args.device)

    # Normalize device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("Requested CUDA but CUDA is not available. Falling back to CPU.")
        device = "cpu"

    try:
        generator = Text2ImageGenerator(model_id=args.model_id, device=device, token=args.token)
        images = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            num_images=args.num_images,
        )
        paths = save_images(images, args.output_dir, base_filename=args.base_filename)
        LOGGER.info("Generated %d image(s). Saved to:\n%s", len(paths), "\n".join(paths))
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Operation cancelled by user.")
        return 130
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
