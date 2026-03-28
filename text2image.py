import os
import uuid
import time
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
from PIL import Image

from diffusers import DiffusionPipeline
from huggingface_hub import login


# Configure module-level logger
logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
if not logger.handlers:
    logger.addHandler(handler)


class Text2ImageError(Exception):
    """Base exception for text2image module."""


class ModelLoadError(Text2ImageError):
    """Raised when a model cannot be loaded."""


class GenerationError(Text2ImageError):
    """Raised when image generation fails."""


@dataclass
class GenerationConfig:
    """Configuration for generation.

    Attributes:
        prompt: Prompt text to generate image from.
        num_inference_steps: Number of diffusion steps. Higher -> better quality but slower.
        guidance_scale: Classifier-free guidance scale.
        height: Output height (pixels).
        width: Output width (pixels).
        seed: Optional seed for deterministic results.
        num_images_per_prompt: Number of images to generate per prompt.
        output_dir: Directory to write generated images.
    """

    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    num_images_per_prompt: int = 1
    output_dir: str = "./outputs"


class Text2ImageGenerator:
    """Service for generating images from text using HuggingFace diffusers.

    This class wraps model loading, device selection, and generation with
    sensible defaults, input validation, and error handling.

    Example:
        gen = Text2ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")
        out_paths = gen.generate(GenerationConfig(prompt="A cute cat"))

    Notes on security and licensing:
        - Models are not bundled; the user must provide a model_id that they are
          authorized to use. If using gated models, set HF_TOKEN environment
          variable or pass hf_token to the constructor.
        - This module performs minimal safety checking; production deployments
          should integrate application-level content moderation and rate limiting.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        """Initialize the generator.

        Args:
            model_id: HF model repo id for a text-to-image pipeline.
            device: torch device string (e.g., "cuda" or "cpu"). If None, auto-selects.
            hf_token: Optional Hugging Face token to access gated models.
            torch_dtype: Optional torch dtype to use (e.g., "auto", "float16").
            trust_remote_code: Whether to set trust_remote_code=True for model loading.
        """
        self.model_id = model_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.trust_remote_code = trust_remote_code

        # device selection
        self.device = device or self._select_device()

        # dtype selection
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)

        self.pipeline: Optional[DiffusionPipeline] = None
        self._load_pipeline()

    def _select_device(self) -> str:
        """Select device automatically. Prefer CUDA if available."""
        if torch.cuda.is_available():
            logger.info("CUDA detected. Using GPU for generation.")
            return "cuda"
        # Consider MPS for Apple silicon
        if getattr(torch, "has_mps", False) and torch.has_mps:
            logger.info("MPS detected (Apple Silicon). Using mps device for generation.")
            return "mps"
        logger.info("No GPU detected. Falling back to CPU.")
        return "cpu"

    def _resolve_torch_dtype(self, dtype: Optional[str]) -> Optional[torch.dtype]:
        if not dtype or dtype == "auto":
            # Use float16 on CUDA to improve speed and memory usage
            if self.device == "cuda":
                return torch.float16
            # MPS currently works best with float32
            if self.device == "mps":
                return torch.float32
            return torch.float32
        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        resolved = mapping.get(dtype)
        if resolved is None:
            raise ValueError(f"Unsupported torch_dtype: {dtype}")
        return resolved

    def _login_if_needed(self) -> None:
        if self.hf_token:
            try:
                login(self.hf_token, add_to_git_credential=False)
                logger.info("Logged into Hugging Face hub with provided token.")
            except Exception as e:
                logger.warning("Hugging Face login failed: %s", e)

    def _load_pipeline(self) -> None:
        """Load the Diffusers pipeline with caching and device placement.

        The pipeline is stored on self.pipeline for reuse across generate() calls.
        """
        if self.pipeline is not None:
            return

        self._login_if_needed()

        try:
            logger.info("Loading model %s on device %s", self.model_id, self.device)
            # For stable diffusion text->image
            load_kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": True,
                "trust_remote_code": self.trust_remote_code,
            }
            # If no dtype is provided, remove the key
            if self.torch_dtype is None:
                load_kwargs.pop("torch_dtype", None)

            pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                revision=None,
                use_auth_token=self.hf_token,
                **load_kwargs,
            )

            # Move to device
            pipe.to(self.device)

            # Enable attention slicing to reduce memory when needed
            try:
                pipe.enable_attention_slicing()
            except Exception:
                # Not critical
                pass

            self.pipeline = pipe
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load model %s", self.model_id)
            raise ModelLoadError(str(e))

    def _validate_config(self, cfg: GenerationConfig) -> None:
        if not cfg.prompt or not cfg.prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if not (1 <= cfg.num_inference_steps <= 200):
            raise ValueError("num_inference_steps must be between 1 and 200.")
        if not (1.0 <= cfg.guidance_scale <= 30.0):
            raise ValueError("guidance_scale must be between 1.0 and 30.0.")
        if not (64 <= cfg.height <= 2048 and cfg.height % 8 == 0):
            raise ValueError("height must be a multiple of 8 in range [64, 2048].")
        if not (64 <= cfg.width <= 2048 and cfg.width % 8 == 0):
            raise ValueError("width must be a multiple of 8 in range [64, 2048].")
        if not (1 <= cfg.num_images_per_prompt <= 8):
            raise ValueError("num_images_per_prompt must be between 1 and 8.")

    def generate(
        self,
        cfg: GenerationConfig,
    ) -> List[str]:
        """Generate images for the given configuration and save them to disk.

        Args:
            cfg: GenerationConfig with prompt and generation parameters.

        Returns:
            List of file paths to generated images.

        Raises:
            GenerationError on failure.
        """
        self._validate_config(cfg)

        # Ensure pipeline is loaded
        if self.pipeline is None:
            self._load_pipeline()
        pipe = self.pipeline
        if pipe is None:
            raise GenerationError("Pipeline is not available")

        os.makedirs(cfg.output_dir, exist_ok=True)

        # Set seed
        generator = None
        if cfg.seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
            except Exception:
                # Fallback to CPU generator
                generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

        logger.info(
            "Generating %d image(s) for prompt: %s",
            cfg.num_images_per_prompt,
            (cfg.prompt[:120] + "...") if len(cfg.prompt) > 120 else cfg.prompt,
        )

        try:
            # Use autocast for mixed precision on CUDA to speed up and reduce memory
            autocast_context = torch.autocast(self.device) if self.device == "cuda" else torch.no_grad()

            with autocast_context:
                output = pipe(
                    [cfg.prompt] * cfg.num_images_per_prompt,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    generator=generator,
                )

            images: List[Image.Image] = output.images

            file_paths: List[str] = []
            for img in images:
                filename = self._unique_filename(cfg.prompt)
                path = os.path.join(cfg.output_dir, filename)
                img.save(path, format="PNG")
                file_paths.append(path)
                logger.info("Saved generated image: %s", path)

            return file_paths

        except RuntimeError as e:
            # OOM handling: try a fallback strategy
            logger.exception("RuntimeError during generation: %s", e)
            if "out of memory" in str(e).lower() and self.device == "cuda":
                torch.cuda.empty_cache()
                logger.warning("CUDA out-of-memory: consider lowering steps/resolution or using fewer images.")
            raise GenerationError(str(e))
        except Exception as e:
            logger.exception("Failed to generate images: %s", e)
            raise GenerationError(str(e))

    def _unique_filename(self, prompt: str) -> str:
        # Create human-readable but unique filename using safe short prompt, timestamp, uuid
        safe = (
            "".join(c for c in prompt if c.isalnum() or c in (" ", "-", "_")).strip()
        )[:50]
        safe = safe.replace(" ", "_") or "img"
        ts = int(time.time())
        uid = uuid.uuid4().hex[:8]
        return f"{safe}_{ts}_{uid}.png"


if __name__ == "__main__":
    # Simple CLI example for manual testing
    import argparse

    parser = argparse.ArgumentParser("Text2Image generator using diffusers")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--out", type=str, default="./outputs")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num", type=int, default=1)
    args = parser.parse_args()

    gen = Text2ImageGenerator(model_id=args.model)
    cfg = GenerationConfig(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_images_per_prompt=args.num,
        output_dir=args.out,
    )
    try:
        paths = gen.generate(cfg)
        print("Generated:")
        for p in paths:
            print(" -", p)
    except Text2ImageError as e:
        logger.error("Generation failed: %s", e)
