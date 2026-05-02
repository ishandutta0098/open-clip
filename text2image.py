from __future__ import annotations

import os
import io
import threading
import logging
import tempfile
from typing import Optional, Callable, Dict, Any
from pathlib import Path

from PIL import Image

import torch
from diffusers import DiffusionPipeline

# Module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
logger.addHandler(_ch)


class Text2ImageError(Exception):
    """Generic exception for Text2ImageGenerator failures."""


class Text2ImageGenerator:
    """
    High-level text-to-image generator wrapper around Hugging Face Diffusers.

    This class provides a production-ready, thread-safe, cached interface to
    load a diffusion pipeline (Stable Diffusion or other compatible pipeline)
    and generate images from text prompts.

    Security and operational notes:
    - Model weights may require an access token. Supply via `hf_token` or set
      the `HUGGINGFACE_TOKEN` environment variable.
    - Avoid untrusted prompt injection in multi-tenant environments. Sanitize
      user prompts before passing them here if you accept user-provided text.

    Example:
        gen = Text2ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")
        image = gen.generate("A scenic landscape at sunrise", seed=42)
        image.save("out.png")

    Args:
        model_id: Hugging Face model ID (e.g. "runwayml/stable-diffusion-v1-5").
        device: explicit device string, e.g. "cuda" or "cpu". If None, device
            will be auto-selected (cuda if available else cpu).
        cache_dir: where to cache model weights.
        hf_token: optional token for private models. If None, will use
            HUGGINGFACE_TOKEN environment variable.
    """

    # Shared cache across instances to reuse loaded pipelines
    _pipeline_cache: Dict[str, DiffusionPipeline] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")

        self.model_id = model_id
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        self.cache_dir = cache_dir
        # Choose device: prefer provided, else cuda if available
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Initializing Text2ImageGenerator for model_id=%s on device=%s", self.model_id, self.device)

        # Ensure pipeline loaded (lazy load) but create reference now
        self._pipeline = None

    def _load_pipeline(self) -> DiffusionPipeline:
        """
        Load and cache the DiffusionPipeline instance for the configured model_id.
        Thread-safe: will only instantiate pipeline once per model_id.
        """
        with Text2ImageGenerator._cache_lock:
            if self.model_id in Text2ImageGenerator._pipeline_cache:
                pipeline = Text2ImageGenerator._pipeline_cache[self.model_id]
                logger.debug("Reusing cached pipeline for %s", self.model_id)
            else:
                logger.info("Loading pipeline for %s (this may take a while)", self.model_id)

                # Determine dtype: use float16 for CUDA to save memory and speed
                torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

                # Using device_map='auto' may be helpful for multi-GPU / offloading
                # but to keep deterministic behaviour we move pipeline to device explicitly.
                # Many SD pipelines work well with `torch_dtype=torch.float16` on CUDA.
                try:
                    pipeline = DiffusionPipeline.from_pretrained(
                        self.model_id,
                        revision=None,
                        torch_dtype=torch_dtype,
                        use_auth_token=self.hf_token,
                        cache_dir=self.cache_dir,
                    )
                except TypeError:
                    # Older diffusers may not accept use_auth_token positional; try without or with token
                    pipeline = DiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch_dtype,
                        cache_dir=self.cache_dir,
                    )

                # If attention slicing available, enable it to reduce VRAM usage by default
                if hasattr(pipeline, "enable_attention_slicing"):
                    try:
                        pipeline.enable_attention_slicing()
                        logger.debug("Enabled attention slicing for memory optimization")
                    except Exception:
                        logger.debug("Could not enable attention slicing")

                # Move to the desired device
                try:
                    pipeline.to(self.device)
                except Exception as exc:  # pragma: no cover - device move may fail in unusual envs
                    logger.exception("Failed to move pipeline to device %s: %s", self.device, exc)
                    raise Text2ImageError("Failed to initialize model on device") from exc

                Text2ImageGenerator._pipeline_cache[self.model_id] = pipeline
                logger.info("Pipeline loaded and cached for %s", self.model_id)

            return Text2ImageGenerator._pipeline_cache[self.model_id]

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        callback: Optional[Callable[[int, int, Any], None]] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: text prompt to render. Must be non-empty.
            seed: optional random seed for reproducibility.
            num_inference_steps: number of denoising steps (higher improves quality/time).
            guidance_scale: classifier-free guidance scale; typical 7.5.
            height: pixel height. If None, uses model default.
            width: pixel width. If None, uses model default.
            negative_prompt: optional negative prompt to steer outputs away from undesired concepts.
            output_path: if provided, save the generated image to this path.
            callback: optional callback(progress_step, num_inference_steps, info) invoked during generation

        Returns:
            PIL.Image.Image instance of the generated image.

        Raises:
            Text2ImageError: for recoverable or expected failures.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        if num_inference_steps <= 0 or num_inference_steps > 500:
            raise ValueError("num_inference_steps must be between 1 and 500")

        if guidance_scale < 1.0 or guidance_scale > 20.0:
            raise ValueError("guidance_scale suspicious; expected in range [1.0, 20.0]")

        # Load pipeline (cached)
        pipeline = self._load_pipeline()

        # Respect provided sizes if supported by pipeline's scheduler/vae
        gen_kwargs: Dict[str, Any] = {
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
        }
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt

        if height is not None:
            gen_kwargs["height"] = int(height)
        if width is not None:
            gen_kwargs["width"] = int(width)

        # Setup reproducible generator
        generator = None
        if seed is not None:
            try:
                # Use device-specific generator for deterministic results on GPU/CPU
                device_str = str(self.device)
                if self.device.type == "cuda":
                    generator = torch.Generator(device="cuda")
                else:
                    generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed))
            except Exception:
                logger.exception("Failed to configure deterministic generator; proceeding without seed")
                generator = None

        if generator is not None:
            gen_kwargs["generator"] = generator

        # Optional progress reporting wrapper if pipeline supports callbacks
        # diffusers pipelines can accept callback and callback_steps in some versions
        if callback is not None:
            gen_kwargs["callback"] = callback
            # default to calling back every step if callback_steps not set
            gen_kwargs.setdefault("callback_steps", 1)

        # Perform generation with optimal autocast if CUDA available
        try:
            if self.device.type == "cuda":
                # Mixed precision inference on CUDA
                with torch.autocast(device_type="cuda"):
                    result = pipeline(prompt=prompt, **gen_kwargs)
            else:
                result = pipeline(prompt=prompt, **gen_kwargs)
        except Exception as exc:
            logger.exception("Text-to-image generation failed: %s", exc)
            raise Text2ImageError("Generation failed") from exc

        # Different diffusers versions return different structures; prefer .images
        image = None
        if isinstance(result, dict) and "images" in result:
            images = result["images"]
            if not images:
                raise Text2ImageError("Model returned no images")
            image = images[0]
        elif hasattr(result, "images"):
            images = getattr(result, "images")
            if not images:
                raise Text2ImageError("Model returned no images")
            image = images[0]
        elif isinstance(result, list) and result:
            image = result[0]
        else:
            # Last attempt: if pipeline returns PIL directly
            image = result

        if not isinstance(image, Image.Image):
            # Some pipelines return numpy arrays
            try:
                image = Image.fromarray(image)
            except Exception:
                logger.exception("Unable to coerce pipeline output to PIL.Image")
                raise Text2ImageError("Unexpected pipeline output type")

        # Optionally save output atomically
        if output_path:
            self._atomic_save(image, output_path)

        return image

    @staticmethod
    def _atomic_save(image: Image.Image, path: str) -> None:
        """
        Atomically save image to path to avoid race conditions / partial files.
        Creates parent directories as needed.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Save into a temporary file in the same directory then rename
        fd, tmp = tempfile.mkstemp(prefix=p.stem, suffix=p.suffix or ".png", dir=str(p.parent))
        os.close(fd)
        try:
            # Use PNG quality-preserving default; let PIL infer format from suffix
            image.save(tmp)
            os.replace(tmp, str(p))
            logger.info("Saved image to %s", p)
        except Exception as exc:
            logger.exception("Failed to save image to %s: %s", p, exc)
            # Attempt to remove temp file
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise Text2ImageError("Failed to save generated image") from exc


def _demo_cli() -> None:
    """
    Minimal CLI demo for local testing. Not intended as a production CLI.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate an image from text using diffusers.")
    parser.add_argument("prompt", type=str, help="Text prompt to render")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="HF model id")
    parser.add_argument("--out", default="out.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    gen = Text2ImageGenerator(model_id=args.model)
    img = gen.generate(
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        output_path=args.out,
    )
    print(f"Image generated and saved to {args.out}")


if __name__ == "__main__":
    _demo_cli()
