import os
import io
import logging
import base64
import tempfile
from typing import Optional, List, Union, Dict, Any
from contextlib import nullcontext

from PIL import Image

import torch
from diffusers import StableDiffusionPipeline

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_ch.setFormatter(_formatter)
logger.addHandler(_ch)


# Simple in-memory cache for loaded pipelines to avoid re-loading in the same process
_PIPELINE_CACHE: Dict[str, StableDiffusionPipeline] = {}


class Text2ImageError(Exception):
    """Base exception for text2image module."""


def _validate_prompt(prompt: str) -> None:
    if not prompt or not isinstance(prompt, str):
        raise Text2ImageError("Prompt must be a non-empty string")


def _validate_resolution(width: int, height: int) -> None:
    # Stable Diffusion typically requires multiples of 8 (or 64 in some variants); 8 is safe
    if width % 8 != 0 or height % 8 != 0:
        raise Text2ImageError("width and height must be multiples of 8 for the model")
    if width <= 0 or height <= 0:
        raise Text2ImageError("width and height must be positive integers")


def _validate_num_images(n: int) -> None:
    if n <= 0 or n > 8:
        # limit to 8 to avoid OOM in typical settings; adjust as needed for your infra
        raise Text2ImageError("num_images must be between 1 and 8")


def _sanitize_filename(path: str) -> str:
    # Ensure we don't write outside intended directories; only basename is allowed
    return os.path.basename(path)


def _get_device_and_dtype(force_device: Optional[str] = None) -> (str, torch.dtype):
    """Decide runtime device and dtype. If CUDA is available, use float16 for perf.

    Args:
        force_device: Optional override like 'cpu' or 'cuda'
    Returns:
        tuple of (device_str, torch_dtype)
    """
    if force_device is not None:
        device = force_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def _load_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    torch_device: Optional[torch.device] = None,
    revision: Optional[str] = None,
    use_auth_token: Optional[str] = None,
    safety_checker: Optional[bool] = True,
) -> StableDiffusionPipeline:
    """Load or retrieve cached StableDiffusionPipeline.

    This function wraps loading logic to keep load/reload behavior consistent.
    """
    cache_key = f"{model_id}:{device}:{dtype}:{revision}:{safety_checker}"
    if cache_key in _PIPELINE_CACHE:
        logger.debug("Using cached pipeline for %s", cache_key)
        return _PIPELINE_CACHE[cache_key]

    logger.info("Loading pipeline %s on %s with dtype=%s", model_id, device, dtype)

    try:
        # Use device_map="auto" for automatic offloading if available; otherwise set device
        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
        }
        if revision:
            kwargs["revision"] = revision
        if use_auth_token:
            kwargs["use_auth_token"] = use_auth_token

        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

        # Move to device
        if device == "cuda":
            pipeline = pipeline.to("cuda")
        else:
            pipeline = pipeline.to("cpu")

        # Optionally disable NSFW filter here (not recommended unless you have a separate safety flow)
        if not safety_checker:
            pipeline.safety_checker = None

        # Cache it
        _PIPELINE_CACHE[cache_key] = pipeline
        return pipeline
    except Exception as e:
        logger.exception("Failed to load model %s: %s", model_id, e)
        raise Text2ImageError(f"Failed to load model {model_id}: {e}")


class Text2Image:
    """Simple text-to-image generator wrapper around Hugging Face Diffusers.

    This class focuses on stability, safety checks, resource management, and
    providing a convenient API for synchronous generation of images from text.

    Usage example:
        t2i = Text2Image()
        images = t2i.generate("a red fox in a meadow", num_images=2)

    Important:
        - Model weights are large and will be downloaded the first time the model is loaded.
        - For production, consider running on a GPU-equipped host and pinning model to GPU.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        revision: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        safety_checker: bool = True,
    ) -> None:
        """Initialize the text2image generator.

        Args:
            model_id: Hugging Face model identifier for a Stable Diffusion model.
            device: 'cuda' or 'cpu'. If None, auto-detect (GPU preferred).
            revision: Optional model revision to load.
            use_auth_token: Optional auth token for private or gated models.
            safety_checker: Whether to keep the pipeline's NSFW safety checker enabled.
        """
        self.model_id = model_id
        self.device, self.dtype = _get_device_and_dtype(device)
        self.revision = revision
        self.use_auth_token = use_auth_token
        self.safety_checker = safety_checker

        # Pipeline will be loaded on demand
        self._pipeline: Optional[StableDiffusionPipeline] = None

    def _ensure_pipeline(self) -> StableDiffusionPipeline:
        if self._pipeline is None:
            self._pipeline = _load_pipeline(
                self.model_id,
                device=self.device,
                dtype=self.dtype,
                revision=self.revision,
                use_auth_token=self.use_auth_token,
                safety_checker=self.safety_checker,
            )
        return self._pipeline

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        output_path: Optional[str] = None,
        return_type: str = "pil",
    ) -> Union[List[Image.Image], List[bytes], List[str]]:
        """Generate images from a text prompt.

        Args:
            prompt: Prompt text describing the desired image(s).
            negative_prompt: Text describing things to avoid (optional).
            num_images: Number of images to generate (1-8).
            height: Output image height (multiple of 8).
            width: Output image width (multiple of 8).
            seed: Optional integer seed for reproducible outputs.
            guidance_scale: CFG scale (higher -> more prompt-conforming).
            num_inference_steps: Number of denoising steps.
            output_path: If provided, each image will be written to this directory (basename safe).
            return_type: 'pil' (PIL Images), 'bytes' (PNG bytes), or 'base64' (base64-encoded PNG strings).

        Returns:
            List of images in the requested format.
        """
        _validate_prompt(prompt)
        _validate_resolution(width, height)
        _validate_num_images(num_images)

        pipeline = self._ensure_pipeline()

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Select appropriate autocast context for dtype performance on CUDA
        amp_ctx = nullcontext()
        if self.device == "cuda":
            try:
                amp_ctx = torch.cuda.amp.autocast()
            except Exception:
                amp_ctx = nullcontext()

        images: List[Image.Image] = []
        try:
            with torch.inference_mode():
                with amp_ctx:
                    logger.info(
                        "Generating %d image(s) with model %s (size=%dx%d, steps=%d)",
                        num_images,
                        self.model_id,
                        width,
                        height,
                        num_inference_steps,
                    )

                    result = pipeline(
                        prompt=[prompt] * num_images,
                        negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )

                    # Result typically contains 'images' (list of PIL Images)
                    raw_images = getattr(result, "images", None)
                    if raw_images is None:
                        raise Text2ImageError("Model did not return images")

                    images = raw_images

        except Exception as e:
            logger.exception("Error during generation: %s", e)
            # Try to free GPU memory in case of OOM
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            raise

        # Post-process: save to disk optionally and convert to requested return type
        outputs: List[Union[Image.Image, bytes, str]] = []
        for idx, img in enumerate(images):
            if output_path:
                safe_name = _sanitize_filename(output_path)
                # If output_path is a dir, construct filename
                if os.path.isdir(safe_name) or output_path.endswith(os.path.sep):
                    # Use a temporary name
                    out_name = os.path.join(safe_name, f"{idx}.png")
                else:
                    # If user provided a file path, append index
                    base, ext = os.path.splitext(safe_name)
                    if not ext:
                        ext = ".png"
                    out_name = os.path.join(os.getcwd(), f"{base}_{idx}{ext}")

                # Ensure we don't write outside cwd by using basename
                out_name = os.path.join(os.getcwd(), os.path.basename(out_name))
                logger.debug("Saving generated image to %s", out_name)
                img.save(out_name, format="PNG")

            if return_type == "pil":
                outputs.append(img)
            elif return_type == "bytes":
                with io.BytesIO() as bio:
                    img.save(bio, format="PNG")
                    outputs.append(bio.getvalue())
            elif return_type == "base64":
                with io.BytesIO() as bio:
                    img.save(bio, format="PNG")
                    b = bio.getvalue()
                    outputs.append(base64.b64encode(b).decode("utf-8"))
            else:
                raise Text2ImageError(f"Unsupported return_type: {return_type}")

        # Optionally free GPU memory if desired (may slow down repeated calls if you reload pipeline)
        # torch.cuda.empty_cache()

        return outputs


def _cli_example():
    """Small CLI for quick manual testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Text2Image CLI (diffusers wrapper)")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="HF model id")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="Output filename or directory")
    args = parser.parse_args()

    t2i = Text2Image(model_id=args.model)
    images = t2i.generate(
        prompt=args.prompt,
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        seed=args.seed,
        output_path=args.out,
        return_type="pil",
    )

    # If images saved to disk via --out, they are persisted; otherwise show them
    if args.out is None:
        for i, im in enumerate(images):
            im.show(title=f"result_{i}")


if __name__ == "__main__":
    _cli_example()
