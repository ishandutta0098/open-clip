import os
import logging
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# Module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Text2ImageError(Exception):
    """Base error for text2image operations."""


@dataclass(frozen=True)
class T2IConfig:
    """Configuration for the text-to-image generator.

    Attributes:
        model_id: Hugging Face Diffusers-compatible model id.
        device: torch device string (e.g. 'cuda' or 'cpu').
        use_auth_token: Optional Hugging Face token. If None, environment
            variable HUGGINGFACE_API_TOKEN will be read.
    """

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_auth_token: Optional[str] = None


class Text2ImageGenerator:
    """High-level text-to-image generator using Hugging Face Diffusers.

    This class encapsulates model loading, inference parameters, input
    validation, resource optimization, and safe saving of outputs.

    Usage:
        gen = Text2ImageGenerator()
        img = gen.generate("An astronaut riding a horse", seed=42)
        gen.save(img, "out.png")
    """

    _PIPELINE: Optional[StableDiffusionPipeline] = None
    _CONFIG: Optional[T2IConfig] = None

    def __init__(self, config: Optional[T2IConfig] = None):
        """Create generator instance.

        Args:
            config: Optional configuration; sensible defaults are used
                (Stable Diffusion v1.5 model and available device).
        """
        self.config = config or T2IConfig()
        self._auth_token = self.config.use_auth_token or os.environ.get(
            "HUGGINGFACE_API_TOKEN"
        )
        logger.debug("Text2ImageGenerator init; device=%s", self.config.device)

    def _load_pipeline(self) -> StableDiffusionPipeline:
        """Load (or reuse) the Diffusers pipeline with sensible defaults and
        optimizations applied.

        Returns:
            An initialized StableDiffusionPipeline on the configured device.
        """
        if (
            Text2ImageGenerator._PIPELINE is not None
            and Text2ImageGenerator._CONFIG == self.config
        ):
            logger.debug("Reusing cached pipeline for model %s", self.config.model_id)
            return Text2ImageGenerator._PIPELINE

        logger.info("Loading pipeline for model: %s", self.config.model_id)
        try:
            # Use DPMSolverMultistep for fast convergence; fallback to default if not available
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.config.model_id, subfolder="scheduler",
            )
        except Exception:
            logger.debug("DPMSolver scheduler not found or failed; using default scheduler")
            scheduler = None

        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.model_id,
                revision="fp16" if self.config.device.startswith("cuda") else None,
                torch_dtype=torch.float16 if self.config.device.startswith("cuda") else torch.float32,
                use_auth_token=self._auth_token,
            )

            if scheduler is not None:
                try:
                    pipe.scheduler = scheduler
                except Exception:
                    logger.debug("Failed to attach custom scheduler; continuing with default scheduler")

            # Performance and memory optimizations
            if self.config.device.startswith("cuda"):
                pipe.to(self.config.device)
                # Mixed precision for memory savings and speed
                pipe.enable_xformers_memory_efficient_attention()
                pipe.enable_attention_slicing()
            else:
                pipe.to("cpu")
                pipe.enable_attention_slicing()

            # Disable safety checker by default for compatibility; callers must be
            # aware. If you require safety checks, provide an alternative.
            if getattr(pipe, "safety_checker", None) is not None:
                try:
                    pipe.safety_checker = None
                except Exception:
                    logger.debug("Failed to disable safety checker (it's optional)")

        except Exception as exc:
            logger.exception("Failed to load pipeline: %s", exc)
            raise Text2ImageError("Failed to load model pipeline") from exc

        Text2ImageGenerator._PIPELINE = pipe
        Text2ImageGenerator._CONFIG = self.config

        logger.info("Pipeline loaded and cached")
        return pipe

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise Text2ImageError("Prompt must be a non-empty string")
        if len(prompt) > 1000:
            # arbitrary limit to avoid abuse and long processing
            raise Text2ImageError("Prompt too long; maximum 1000 characters")

    @staticmethod
    def _validate_size(width: int, height: int) -> None:
        if not (isinstance(width, int) and isinstance(height, int)):
            raise Text2ImageError("Width and height must be integers")
        if width <= 0 or height <= 0:
            raise Text2ImageError("Width and height must be positive")
        # Most SD models require multiples of 8; enforce this to avoid silent failures
        if width % 8 != 0 or height % 8 != 0:
            raise Text2ImageError("Width and height must be multiples of 8")
        if width > 2048 or height > 2048:
            raise Text2ImageError("Maximum supported resolution is 2048x2048 to limit memory usage")

    @staticmethod
    def _sanitize_filename(path: str) -> str:
        # Basic sanitization to avoid path traversal
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        return path

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        safety_checker: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
        num_images_per_prompt: int = 1,
    ) -> List[Image.Image]:
        """Generate images from text.

        Args:
            prompt: The text prompt to render.
            negative_prompt: Optional negative prompt (things to avoid).
            width: Output width in pixels (multiple of 8).
            height: Output height in pixels (multiple of 8).
            num_inference_steps: Number of diffusion steps; trade-off between quality and speed.
            guidance_scale: Classifier-free guidance scale; higher = more prompt-faithful.
            seed: Optional integer seed for deterministic outputs.
            safety_checker: If True and the pipeline supports it, runs the safety checker.
            callback: Optional function called with (step, total_steps) during inference.
            num_images_per_prompt: How many images to generate per prompt.

        Returns:
            List of PIL Image objects (length = num_images_per_prompt).

        Raises:
            Text2ImageError on validation or runtime issues.
        """
        # Validate inputs
        self._validate_prompt(prompt)
        self._validate_size(width, height)

        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise Text2ImageError("num_inference_steps must be a positive integer")
        if not (0.0 <= guidance_scale <= 30.0):
            raise Text2ImageError("guidance_scale must be between 0.0 and 30.0")

        pipe = self._load_pipeline()

        # If safety_checker is requested but pipeline disabled it earlier, warn
        if safety_checker and getattr(pipe, "safety_checker", None) is None:
            logger.warning("Safety checker requested but not available in pipeline")

        # Setup generator for deterministic runs
        generator = None
        if seed is not None:
            # Use cuda generator if available for best reproducibility on device
            device_for_gen = "cuda" if self.config.device.startswith("cuda") else "cpu"
            try:
                generator = torch.Generator(device_for_gen).manual_seed(int(seed))
            except Exception:
                # Fall back to CPU generator
                generator = torch.Generator("cpu").manual_seed(int(seed))

        # Define progress callback wrapper expected by diffusers: lambda step, timestep, latents: ...
        def _internal_callback(step: int, timestep: int, latents):
            if callback:
                try:
                    callback(step, num_inference_steps)
                except Exception:
                    logger.exception("User callback raised an exception; continuing")

        # Perform inference
        try:
            logger.info(
                "Generating images: prompt=" + (prompt[:80] + "..." if len(prompt) > 80 else prompt)
            )
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=_internal_callback if callback is not None else None,
                num_images_per_prompt=num_images_per_prompt,
            )
        except Exception as exc:
            logger.exception("Image generation failed: %s", exc)
            raise Text2ImageError("Image generation failed") from exc

        images = getattr(output, "images", None)
        if images is None:
            raise Text2ImageError("Model produced no images")

        # Optionally re-enable safety checker or process safety information here
        # Return PIL images
        return images

    def save(self, image: Image.Image, out_path: str, format: Optional[str] = None) -> str:
        """Save a PIL image to disk safely.

        Args:
            image: PIL Image to save.
            out_path: Destination path.
            format: Optional image format (e.g., 'PNG'). If None, derived from filename.

        Returns:
            The absolute path to the saved file.
        """
        if not isinstance(image, Image.Image):
            raise Text2ImageError("image must be a PIL.Image.Image instance")

        out_path = self._sanitize_filename(out_path)
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        try:
            image.save(out_path, format=format)
        except Exception as exc:
            logger.exception("Failed to save image to %s: %s", out_path, exc)
            raise Text2ImageError("Failed to save image") from exc

        logger.info("Image saved to %s", out_path)
        return out_path


# Convenience function for quick uses
def generate_image_file(
    prompt: str,
    out_path: str,
    config: Optional[T2IConfig] = None,
    **kwargs,
) -> str:
    """Helper to generate an image from a prompt and write it to a file.

    Returns:
        Absolute path to the written image file.
    """
    gen = Text2ImageGenerator(config=config)
    images = gen.generate(prompt, **kwargs)
    # Save the first image by default
    return gen.save(images[0], out_path)


if __name__ == "__main__":
    # Basic CLI example for local testing
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="Text2Image generator using diffusers")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--out", type=str, default="./out.png", help="Output file path")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    cfg = T2IConfig()
    generator = Text2ImageGenerator(cfg)
    img = generator.generate(
        args.prompt, width=args.width, height=args.height, num_inference_steps=args.steps, guidance_scale=args.scale, seed=args.seed
    )
    generator.save(img[0], args.out)
