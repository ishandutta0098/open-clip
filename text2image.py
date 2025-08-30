#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text2ImageGenerator: produces images from text prompts using HuggingFace diffusers
"""

import os
import sys
import argparse
import logging
from typing import Optional, List
from pathlib import Path

import torch
from PIL import Image  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

class Text2ImageGenerator:
    """Text-to-image generator using HuggingFace diffusers Stable Diffusion pipeline."""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1", device: Optional[str] = None, safety_checker: bool = True):
        """Initialize the generator.

        Parameters:
            model_id: HF model identifier to load from the hub.
            device: Computation device to use (e.g., 'cpu' or 'cuda'). If None, auto-detect.
            safety_checker: If True, enable the default safety checker; if False, disable safety checks.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.safety_checker = safety_checker
        self.pipeline = None

    def load_pipeline(self) -> None:
        """Load the Diffusers Stable Diffusion pipeline from the specified model_id.

        This method lazy-loads the model on first use and configures device placement.
        """
        if self.pipeline is not None:
            return
        try:
            from diffusers import StableDiffusionPipeline
        except Exception as e:
            logger.error("Failed to import diffusers: %s", e)
            raise

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        try:
            if self.safety_checker:
                self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=dtype, safety_checker=None)
            self.pipeline = self.pipeline.to(self.device)
            logger.info("Loaded model '%s' on device '%s'", self.model_id, self.device)
        except Exception as e:
            logger.error("Error loading model '%s': %s", self.model_id, e)
            raise

    @staticmethod
    def _build_generator(device: str, seed: Optional[int]) -> Optional[torch.Generator]:
        """Create a PyTorch random generator with the given seed for reproducibility."""
        if seed is None:
            return None
        try:
            g = torch.Generator(device).manual_seed(seed)
            return g
        except Exception:
            return None

    def generate(self, prompt: str, height: int = 512, width: int = 512, num_images: int = 1, guidance_scale: float = 7.5, num_inference_steps: int = 50, seed: Optional[int] = None, negative_prompt: Optional[str] = None) -> List[Image.Image]:
        """Generate images from a text prompt using the loaded pipeline.

        Parameters:
            prompt: Text prompt describing the desired image.
            height, width: Image dimensions (multiples of 8).
            num_images: How many images to generate.
            guidance_scale: CFG scale controlling adherence to the prompt.
            num_inference_steps: Inference steps for diffusion process.
            seed: Optional random seed for reproducibility.
            negative_prompt: Optional prompt to suppress unwanted features.

        Returns:
            List of PIL.Image.Image instances.
        """
        if self.pipeline is None:
            self.load_pipeline()
        generator = self._build_generator(self.device, seed)
        try:
            result = self.pipeline(
                prompt,
                height=height,
                width=width,
                num_images=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                generator=generator,
            )
            images = getattr(result, "images", None)
            if images is None and isinstance(result, (list, tuple)) and len(result) > 0:
                images = result[0]
            if not isinstance(images, list):
                images = [images]
            return images
        except Exception as e:
            logger.error("Image generation failed: %s", e)
            raise

def sanitize_filename(name: str) -> str:
    invalid = r'[^-_.() A-Za-z0-9]'
    import re
    sanitized = re.sub(invalid, "_", name)
    if len(sanitized) > 128:
        sanitized = sanitized[:128]
    return sanitized

def main():
    parser = argparse.ArgumentParser(description="Text-to-image generator using HuggingFace diffusers.")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the image.")
    parser.add_argument("--output-dir", default="./outputs", help="Directory to save generated images.")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1", help="Diffusers model id to load.")
    parser.add_argument("--num-images", type=int, default=1, help="How many images to generate.")
    parser.add_argument("--width", type=int, default=512, help="Image width (must be multiple of 8).")
    parser.add_argument("--height", type=int, default=512, help="Image height (must be multiple of 8).")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale (CFG)." )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt to suppress unwanted features.")
    parser.add_argument("--device", default=None, help="Computation device to use (cpu/cuda).")
    parser.add_argument("--disable-safety-check", action="store_true", help="Disable safety checker (not recommended).")
    args = parser.parse_args()

    if args.width % 8 != 0 or args.height % 8 != 0:
        raise SystemExit("Width and height must be multiples of 8.")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting text-to-image generation with prompt: '%s'", args.prompt)
    generator = Text2ImageGenerator(model_id=args.model_id, device=args.device, safety_checker=not args.disable_safety_check)
    try:
        images = generator.generate(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_images=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
        )
        base_name = sanitize_filename(args.prompt) if args.prompt else "image"
        for i, im in enumerate(images, start=1):
            out_path = Path(args.output_dir) / f"{base_name}_#{i}.png"
            im.save(out_path)
            logger.info("Saved image %d to %s", i, out_path)
        logger.info("Generation complete. Total images: %d", len(images))
    except Exception as e:
        logger.exception("Failed to generate images: %s", e)
        sys.exit(1)
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

if __name__ == "__main__":
    main()
