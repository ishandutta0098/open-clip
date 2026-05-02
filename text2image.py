import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


class Text2ImageError(Exception):
    """Generic error for Text2Image operations."""


@dataclass
class GenerationOptions:
    """Options for image generation.

    Attributes:
        prompt: The text prompt to guide generation.
        negative_prompt: (Optional) text used to discourage elements.
        height: Image height (pixels). Must be multiple of 8.
        width: Image width (pixels). Must be multiple of 8.
        num_inference_steps: Diffusion steps (higher -> slower but often better quality).
        guidance_scale: Classifier-free guidance scale (higher -> more prompt adherence).
        seed: Optional RNG seed for deterministic output.
    """

    prompt: str
    negative_prompt: Optional[str] = None
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None


class Text2Image:
    """High-level helper for text -> image generation using Hugging Face Diffusers.

    This class wraps StableDiffusionPipeline for convenient, safe, and efficient
    generation. It handles device selection, dtype selection, model loading and
    exposes a simple generate API that returns a PIL.Image.

    Example:
        t2i = Text2Image(model_id="runwayml/stable-diffusion-v1-5")
        img = t2i.generate("A painting of a fox in a forest")
        img.save("out.png")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        disable_safety_checker: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize Text2Image.

        Args:
            model_id: HF model id for the diffusion model.
            device: Optional device spec (e.g. "cuda", "cuda:0", "cpu").
                    If None, a best-effort device is chosen.
            use_auth_token: Optional Hugging Face token if required to access model.
            disable_safety_checker: If True, disable the pipeline safety checker.
            torch_dtype: Optional dtype to load model with. If None, selects float16 for CUDA, float32 otherwise.
        """
        self.model_id = model_id
        self.use_auth_token = use_auth_token
        self.disable_safety_checker = disable_safety_checker

        # Device selection
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info("Selected device: %s", self.device)

        # dtype selection
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        else:
            # Use fp16 on CUDA for speed/memory, otherwise fp32
            self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        logger.info("Using torch dtype: %s", self.torch_dtype)

        self.pipeline: Optional[StableDiffusionPipeline] = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load and configure the Stable Diffusion pipeline.

        The pipeline is configured for inference: attention slicing is enabled to
        reduce peak memory, and the safety checker can be optionally disabled.
        """
        try:
            logger.info("Loading pipeline for model_id=%s", self.model_id)

            # Choose appropriate kwargs for pipeline construction
            pipeline_kwargs = {
                "torch_dtype": self.torch_dtype,
            }
            if self.use_auth_token:
                pipeline_kwargs["use_auth_token"] = self.use_auth_token

            pipe = StableDiffusionPipeline.from_pretrained(self.model_id, **pipeline_kwargs)

            # Move to device
            pipe = pipe.to(self.device)

            # Memory optimizations
            try:
                pipe.enable_attention_slicing()
            except Exception:
                # Older/newer diffusers may not have this function
                logger.debug("enable_attention_slicing not supported by this pipeline")

            # Optionally try enabling xformers for efficient attention if installed
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.debug("xFormers optimization not enabled or not installed")

            # Optionally disable safety checker (explicit opt-in required)
            if self.disable_safety_checker:
                try:
                    pipe.safety_checker = None  # type: ignore[attr-defined]
                except Exception:
                    logger.debug("Pipeline does not expose a safety_checker or cannot be disabled")

            self.pipeline = pipe
            logger.info("Pipeline loaded successfully")
        except Exception as exc:  # broad catch to wrap details
            logger.exception("Failed to load pipeline: %s", exc)
            raise Text2ImageError(f"Failed to load model '{self.model_id}': {exc}")

    @staticmethod
    def _validate_options(opts: GenerationOptions) -> None:
        if not opts.prompt or not isinstance(opts.prompt, str):
            raise Text2ImageError("prompt must be a non-empty string")

        if opts.height % 8 != 0 or opts.width % 8 != 0:
            raise Text2ImageError("height and width must be multiples of 8")

        if not (64 <= opts.height <= 2048) or not (64 <= opts.width <= 2048):
            # These bounds are conservative; adjust as model supports.
            raise Text2ImageError("height and width must be between 64 and 2048")

        if not (1 <= opts.num_inference_steps <= 500):
            raise Text2ImageError("num_inference_steps must be between 1 and 500")

        if not (0.0 <= opts.guidance_scale <= 50.0):
            raise Text2ImageError("guidance_scale must be between 0.0 and 50.0")

    def generate(self, opts: GenerationOptions) -> Image.Image:
        """Generate a single image from the provided prompt options.

        Args:
            opts: GenerationOptions dataclass instance.

        Returns:
            A PIL.Image instance with the generated result.

        Raises:
            Text2ImageError: on validation, runtime, or model errors.
        """
        if self.pipeline is None:
            raise Text2ImageError("Model pipeline is not loaded")

        self._validate_options(opts)

        generator = None
        if opts.seed is not None:
            try:
                # Create a torch.Generator for reproducible results
                generator = torch.Generator(device=self.device).manual_seed(int(opts.seed))
            except Exception as exc:
                logger.warning("Failed to create generator for seed %s: %s", opts.seed, exc)
                generator = None

        logger.info(
            "Generating image (prompt=%s, height=%d, width=%d, steps=%d, guidance_scale=%.2f)",
            (opts.prompt[:80] + "..." if len(opts.prompt) > 80 else opts.prompt),
            opts.height,
            opts.width,
            opts.num_inference_steps,
            opts.guidance_scale,
        )

        # Use inference_mode for better performance and to avoid gradients
        try:
            # Newer diffusers accept negative_prompt directly in the __call__
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=opts.prompt,
                    negative_prompt=opts.negative_prompt,
                    height=opts.height,
                    width=opts.width,
                    num_inference_steps=opts.num_inference_steps,
                    guidance_scale=opts.guidance_scale,
                    generator=generator,
                )

            # The pipeline returns a Image or a dict depending on version; handle common cases
            if isinstance(result, dict):
                images = result.get("images") or result.get("image")
                if not images:
                    raise Text2ImageError("Pipeline returned no images")
                image = images[0]
            elif isinstance(result, list) or isinstance(result, tuple):
                image = result[0]
            else:
                image = result

            if not isinstance(image, Image.Image):
                # Some pipeline returns numpy arrays, handle that
                try:
                    image = Image.fromarray(image)
                except Exception:
                    raise Text2ImageError("Generated output is not an image and cannot be converted")

            logger.info("Image generation completed successfully")
            return image

        except Exception as exc:
            logger.exception("Image generation failed: %s", exc)
            raise Text2ImageError(f"Image generation failed: {exc}")

    def generate_to_file(self, opts: GenerationOptions, output_path: str, overwrite: bool = False) -> str:
        """Generate an image and save it to disk.

        Args:
            opts: GenerationOptions describing generation parameters.
            output_path: Destination file path. Parent directories will be created.
            overwrite: If False and file exists, raises error.

        Returns:
            The absolute path to the saved image.
        """
        if os.path.exists(output_path) and not overwrite:
            raise Text2ImageError(f"Output file already exists: {output_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        image = self.generate(opts)
        try:
            # Use high-quality saving defaults
            image.save(output_path, format="PNG")
            logger.info("Saved generated image to %s", output_path)
            return os.path.abspath(output_path)
        except Exception as exc:
            logger.exception("Failed to save image to disk: %s", exc)
            raise Text2ImageError(f"Failed to save image: {exc}")


if __name__ == "__main__":
    # Simple CLI for quick local testing. Not intended as a production CLI.
    import argparse

    parser = argparse.ArgumentParser(description="Text -> Image (Stable Diffusion) helper")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("output", type=str, help="Output file path (PNG)")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="HF model id")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable-safety", action="store_true")
    args = parser.parse_args()

    # Basic parameter validation and run
    gen_opts = GenerationOptions(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    t2i = Text2Image(model_id=args.model, disable_safety_checker=args.disable_safety)
    out = t2i.generate_to_file(gen_opts, args.output, overwrite=False)
    print(f"Saved image to: {out}")
