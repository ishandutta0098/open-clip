import io
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

import torch
from diffusers import StableDiffusionPipeline


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Text2ImageError(Exception):
    """Custom error for Text2ImageGenerator operations."""


class Text2ImageGenerator:
    """
    Text2ImageGenerator encapsulates a Hugging Face diffusers Stable Diffusion pipeline
    and exposes a safe, production-ready generate() method.

    Usage example:
        gen = Text2ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")
        images = gen.generate("A cute corgi wearing a spacesuit")
        images[0].save("out.png")

    Design/Features:
    - Pipeline caching per model_id to avoid repeated downloads / memory churn.
    - Device auto-detection with safe fallbacks (GPU FP16 when available).
    - Memory optimizations enabled (attention slicing).
    - Input validation for sizes and parameters.
    - Thread-safe lazy initialization and reuse of pipelines.
    - Context manager support to guarantee resource cleanup.

    Args:
        model_id: Hugging Face model id for the Stable Diffusion model.
        device: torch device string like "cuda" or "cpu". If None auto-selects.
        use_fp16: Use float16 precision when running on CUDA for reduced memory.
        hf_token: Optional Hugging Face token for private models.

    """

    # Class-level cache; holds loaded pipelines by model id
    _pipelines: dict = {}
    _lock = threading.Lock()

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        use_fp16: bool = True,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.user_provided_device = device
        self._use_fp16_requested = use_fp16

        # Determine actual device
        self.device = self._select_device(device)

        # Use appropriate dtype
        self.torch_dtype = torch.float16 if (self.device.startswith("cuda") and use_fp16 and torch.cuda.is_available()) else torch.float32

        # Pipeline instance will be lazily loaded
        self._pipeline = None

        logger.debug("Text2ImageGenerator initialized: model=%s device=%s dtype=%s", self.model_id, self.device, self.torch_dtype)

    def _select_device(self, device: Optional[str]) -> str:
        """Select compute device safely. Defaults to CUDA if available.

        Args:
            device: optional user requested device string.
        Returns:
            Resolved device string.
        """
        if device:
            # Normalize
            d = device.lower()
            if d in ("cuda", "cpu") or d.startswith("cuda:"):
                # If CUDA requested but unavailable, fallback to CPU
                if d.startswith("cuda") and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    return "cpu"
                return d
            else:
                logger.warning("Unrecognized device '%s', falling back to auto-detection.", device)

        # Auto-detect
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_pipeline(self):
        """Internal: load or get from cache the Stable Diffusion pipeline.

        Uses a class-level cache and a lock to ensure thread-safe singletons per model id.
        """
        with Text2ImageGenerator._lock:
            if self.model_id in Text2ImageGenerator._pipelines:
                logger.debug("Reusing cached pipeline for model %s", self.model_id)
                self._pipeline = Text2ImageGenerator._pipelines[self.model_id]
                # ensure pipeline on desired device
                try:
                    self._pipeline.to(self.device)
                except Exception:
                    logger.exception("Failed to move cached pipeline to device %s; reloading.", self.device)
                    # If moving failed, remove cached pipeline and reload
                    try:
                        del Text2ImageGenerator._pipelines[self.model_id]
                    except Exception:
                        pass
                else:
                    return

            logger.info("Loading Stable Diffusion pipeline for model %s on %s (dtype=%s)", self.model_id, self.device, self.torch_dtype)
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    use_auth_token=self._hf_token,
                )

                # Performance: attention slicing reduces peak memory. Always enable.
                pipe.enable_attention_slicing()

                # Try to enable memory efficient attention if available (xformers)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    logger.debug("xformers not available or failed to enable; continuing without it.")

                # Move pipeline to device
                pipe = pipe.to(self.device)

                # Save into cache
                Text2ImageGenerator._pipelines[self.model_id] = pipe
                self._pipeline = pipe
                logger.info("Pipeline loaded and cached for model %s", self.model_id)
            except Exception as exc:
                logger.exception("Failed to instantiate pipeline for model %s: %s", self.model_id, exc)
                raise Text2ImageError(f"Could not load model {self.model_id}: {exc}") from exc

    def close(self) -> None:
        """Free resources by moving pipeline to CPU and deleting references.

        Note: this does not remove the cached model files from disk.
        """
        with Text2ImageGenerator._lock:
            if self._pipeline is None:
                return
            try:
                # Move to CPU then delete to free GPU memory
                try:
                    self._pipeline.to("cpu")
                except Exception:
                    pass
                # delete local ref and remove from cache
                if self.model_id in Text2ImageGenerator._pipelines:
                    try:
                        del Text2ImageGenerator._pipelines[self.model_id]
                    except Exception:
                        pass
                self._pipeline = None
                logger.info("Pipeline for model %s closed and removed from cache.", self.model_id)
            except Exception:
                logger.exception("Error while closing pipeline for model %s", self.model_id)

    def __enter__(self) -> "Text2ImageGenerator":
        # ensure pipeline loaded
        if self._pipeline is None:
            self._load_pipeline()
        return self

    def __exit__(self, exc_type, exc, tb):
        # best-effort cleanup
        try:
            self.close()
        except Exception:
            logger.exception("Error closing pipeline on context exit")

    @staticmethod
    def _validate_image_dims(height: int, width: int) -> None:
        # Many Stable Diffusion models expect multiples of 8 or 16 depending on architecture.
        if height % 8 != 0 or width % 8 != 0:
            raise Text2ImageError("Height and width must be multiples of 8 to match model's VAE requirements.")
        max_pixels = 1024 * 1024 * 6  # arbitrary 6M pixel guard
        if height * width > max_pixels:
            raise Text2ImageError(f"Requested resolution {width}x{height} is too large and may OOM.")

    @staticmethod
    def _safe_save_image(img: Image.Image, path: Path) -> None:
        # Ensure the directory exists and the path is safe
        base_dir = Path.cwd().resolve()
        target = path.resolve()
        if base_dir not in target.parents and target != base_dir:
            # This is a simple check to prevent writing outside repo working dir by default
            logger.warning("Saving outside current working directory. Ensure this is intended: %s", target)

        target.parent.mkdir(parents=True, exist_ok=True)
        # Pillow handles atomicity poorly; write to temp file then rename
        temp = target.with_suffix(target.suffix + ".tmp")
        img.save(temp)
        temp.replace(target)
        logger.info("Saved image to %s", target)

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
        output_path: Optional[Union[str, Path]] = None,
        return_type: str = "pil",
    ) -> List[Union[Image.Image, bytes]]:
        """
        Generate images from text prompt.

        Args:
            prompt: Main text prompt (non-empty).
            negative_prompt: Negative prompt to discourage features.
            height: Height in pixels (must be multiple of 8).
            width: Width in pixels (must be multiple of 8).
            num_inference_steps: Number of diffusion steps (1-200 recommended).
            guidance_scale: Classifier free guidance scale.
            seed: Optional RNG seed for reproducibility.
            num_images: Number of images to generate (num_images_per_prompt).
            output_path: Optional path or directory to save images. If directory provided, multiple outputs are saved.
            return_type: 'pil' (PIL.Image) or 'bytes' (PNG bytes).

        Returns:
            List of PIL.Image or bytes depending on return_type.

        Raises:
            Text2ImageError on invalid input or generation failure.
        """
        # Basic validations
        if not prompt or not prompt.strip():
            raise Text2ImageError("Prompt must be a non-empty string.")
        if num_images < 1 or num_images > 8:
            raise Text2ImageError("num_images must be between 1 and 8 to avoid excessive memory use.")
        if not (1 <= num_inference_steps <= 200):
            raise Text2ImageError("num_inference_steps must be between 1 and 200.")
        if guidance_scale < 1.0 or guidance_scale > 30.0:
            raise Text2ImageError("guidance_scale should be between 1.0 and 30.0.")
        Text2ImageGenerator._validate_image_dims(height, width)

        # Lazy load pipeline
        if self._pipeline is None:
            self._load_pipeline()

        # Prepare generator
        device_for_generator = "cuda" if self.device.startswith("cuda") else "cpu"
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device_for_generator).manual_seed(int(seed))
            except Exception:
                # Fallback to CPU generator then move
                generator = torch.Generator().manual_seed(int(seed))

        # Assemble kwargs for pipeline
        pipe_kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            num_images_per_prompt=num_images,
        )

        try:
            # Inference; run in autocast if float16 pipeline on CUDA
            if self.torch_dtype == torch.float16 and self.device.startswith("cuda"):
                with torch.autocast(self.device):
                    result = self._pipeline(**pipe_kwargs)
            else:
                result = self._pipeline(**pipe_kwargs)

            images: List[Image.Image] = result.images
            outputs: List[Union[Image.Image, bytes]] = []

            # Save if requested
            if output_path:
                out = Path(output_path)
                if out.exists() and out.is_file() and num_images > 1:
                    raise Text2ImageError("When generating multiple images, output_path must be a directory, not a file.")
                if num_images == 1 and out.suffix:  # single file
                    # ensure parent exists
                    Text2ImageGenerator._safe_save_image(images[0], out)
                else:
                    # treat as directory
                    dir_path = out if out.suffix == "" or out.is_dir() else out.parent
                    dir_path.mkdir(parents=True, exist_ok=True)
                    for i, img in enumerate(images, start=1):
                        target = dir_path / f"{self.model_id.replace('/', '_')}_{i}.png"
                        Text2ImageGenerator._safe_save_image(img, target)

            # Return requested format
            if return_type == "pil":
                return images
            elif return_type == "bytes":
                for img in images:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    outputs.append(buf.getvalue())
                return outputs
            else:
                raise Text2ImageError(f"Unsupported return_type '{return_type}'. Use 'pil' or 'bytes'.")

        except Exception as exc:
            logger.exception("Error during image generation: %s", exc)
            raise Text2ImageError(f"Image generation failed: {exc}") from exc


if __name__ == "__main__":
    # Simple CLI example for developer testing. Not a production CLI.
    logging.basicConfig(level=logging.INFO)
    prompt = "A futuristic city skyline at sunset, ultra-detailed"
    gen = Text2ImageGenerator()
    images = gen.generate(prompt, num_images=1)
    out_file = Path.cwd() / "example_output.png"
    images[0].save(out_file)
    logger.info("Wrote example image to %s", out_file)
