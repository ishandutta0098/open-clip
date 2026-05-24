import io
import logging
import os
import threading
from typing import List, Optional, Union

from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
)
logger.addHandler(_handler)


class Text2ImageError(RuntimeError):
    """Generic error for the Text2Image generator."""


class Text2ImageGenerator:
    """
    Text2ImageGenerator wraps Hugging Face diffusers Stable Diffusion pipeline
    and provides a safe, reusable API to generate images from text prompts.

    Key features:
    - Automatic device selection (CUDA if available, otherwise CPU)
    - Reuse of the underlying pipeline (cached) for multiple inferences
    - Type hints, validation, and detailed logging
    - Optional saving of generated images to disk

    Usage example:
        gen = Text2ImageGenerator()
        images = gen.generate("a photo of a futuristic city at sunset", num_images=2)
        images[0].save("out0.png")
        gen.close()

    Args:
        model_id: HuggingFace model repo id (default: runwayml/stable-diffusion-v1-5)
        device: 'cuda'|'cpu' or None to auto-detect
        cache_dir: optional path to cache models
        enable_torch_cuda_benchmark: toggle torch.backends.cudnn.benchmark for perf on fixed size
    """

    _global_lock = threading.Lock()

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_torch_cuda_benchmark: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._pipe: Optional[StableDiffusionPipeline] = None
        self._closed = False

        # Decide device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device not in ("cuda", "cpu"):
                raise ValueError("device must be 'cuda' or 'cpu' or None")
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("Requested CUDA but torch.cuda.is_available() is False. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = device

        if enable_torch_cuda_benchmark and self.device == "cuda":
            # Can improve perf on fixed-size inputs
            torch.backends.cudnn.benchmark = True

        # Default dtype for weights when loading on GPU
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        logger.info("Text2ImageGenerator initializing: device=%s, model=%s", self.device, self.model_id)
        # Lazy load pipeline on first inference for faster init.

    def _load_pipeline(self) -> StableDiffusionPipeline:
        """
        Load and cache the StableDiffusionPipeline. Thread-safe.
        """
        if self._pipe is not None:
            return self._pipe

        with Text2ImageGenerator._global_lock:
            if self._pipe is not None:
                return self._pipe

            try:
                logger.info("Loading pipeline for model '%s' (device=%s)...", self.model_id, self.device)

                # Use DPMSolverMultistepScheduler which is a performant sampler
                pipe = (
                    StableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        revision=None,
                        torch_dtype=self.torch_dtype,
                        cache_dir=self.cache_dir,
                    )
                )

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

                # Move to device
                pipe = pipe.to(self.device)

                # Enable attention slicing on limited memory devices (optional)
                if self.device == "cuda":
                    try:
                        pipe.enable_attention_slicing()
                        logger.debug("Enabled attention slicing for model to reduce memory.")
                    except Exception:
                        logger.debug("Failed to enable attention slicing; continuing.")

                # Safety: Note that some pipelines include a safety checker. Keep default behavior.

                self._pipe = pipe
                logger.info("Pipeline loaded successfully.")
                return self._pipe
            except Exception as exc:
                logger.exception("Failed to load pipeline: %s", exc)
                raise Text2ImageError("Failed to load Stable Diffusion pipeline") from exc

    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        save_to_dir: Optional[str] = None,
        callback: Optional[callable] = None,
    ) -> List[Image.Image]:
        """
        Generate images from a text prompt.

        Args:
            prompt: Text prompt describing the image.
            num_images: Number of images to generate (1-8 recommended).
            guidance_scale: Classifier-free guidance scale. Higher = more prompt adherence.
            num_inference_steps: Number of denoising steps (tradeoff quality vs speed).
            height: Image height in pixels.
            width: Image width in pixels.
            seed: Optional RNG seed for deterministic outputs.
            save_to_dir: Optional directory to save PNG images. Directory will be created if missing.
            callback: Optional callback called during generation: callback(step, timestep, latents)

        Returns:
            List of PIL.Image instances.

        Raises:
            Text2ImageError on failure.
        """
        if self._closed:
            raise Text2ImageError("Generator is closed")

        # Basic validations
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not (1 <= num_images <= 16):
            raise ValueError("num_images must be between 1 and 16")
        if not (1 <= num_inference_steps <= 500):
            raise ValueError("num_inference_steps must be between 1 and 500")
        if not (64 <= height <= 2048 and height % 8 == 0):
            raise ValueError("height must be a multiple of 8 between 64 and 2048")
        if not (64 <= width <= 2048 and width % 8 == 0):
            raise ValueError("width must be a multiple of 8 between 64 and 2048")

        # Load pipeline
        pipe = self._load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Convert prompt list form if multiple images
        prompts = [prompt] * num_images

        logger.info(
            "Generating %d image(s) for prompt='%s' (steps=%d, guidance=%.2f, size=%dx%d)",
            num_images,
            prompt if len(prompt) < 200 else prompt[:196] + "...",
            num_inference_steps,
            guidance_scale,
            width,
            height,
        )

        try:
            # Use mixed precision on CUDA for speed & mem savings
            with torch.autocast(device_type=self.device) if self.device == "cuda" else _null_context():
                result = pipe(
                    prompts,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback=callback,
                )

            # The pipeline returns a StableDiffusionPipelineOutput with .images (PIL.Image)
            images = result.images
            if save_to_dir:
                os.makedirs(save_to_dir, exist_ok=True)
                for idx, img in enumerate(images):
                    out_path = os.path.join(save_to_dir, f"sd_out_{idx}.png")
                    try:
                        img.save(out_path, format="PNG")
                        logger.info("Saved image to %s", out_path)
                    except Exception:
                        logger.exception("Failed to save image to %s", out_path)

            return images
        except Exception as exc:
            logger.exception("Image generation failed: %s", exc)
            raise Text2ImageError("Image generation failed") from exc

    def close(self) -> None:
        """
        Dispose of the pipeline and free any GPU memory. After calling close(),
        the generator should not be used unless a new instance is created.
        """
        if self._closed:
            return
        logger.info("Closing Text2ImageGenerator and freeing resources.")
        if self._pipe is not None:
            try:
                # Move to CPU then delete to free GPU memory
                try:
                    self._pipe.to("cpu")
                except Exception:
                    pass
                del self._pipe
            except Exception:
                logger.debug("Exception while deleting pipeline")
        # free GPU memory if present
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to empty CUDA cache")

        self._closed = True


# small helper context manager for CPU no-op autocast
class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    # Simple CLI for quick local testing
    import argparse

    parser = argparse.ArgumentParser(description="Generate images from text using Stable Diffusion")
    parser.add_argument("prompt", type=str, help="Prompt text")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--num", type=int, default=1, help="Number of images")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--width", type=int, default=512, help="Width (multiple of 8)")
    parser.add_argument("--height", type=int, default=512, help="Height (multiple of 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic output")
    args = parser.parse_args()

    gen = Text2ImageGenerator()
    try:
        result_images = gen.generate(
            args.prompt,
            num_images=args.num,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            seed=args.seed,
            save_to_dir=args.outdir,
        )
        for i, im in enumerate(result_images):
            print(f"Generated image {i}: size={im.size}")
    finally:
        gen.close()
