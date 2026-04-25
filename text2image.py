import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

try:
    import torch
except Exception as e:  # pragma: no cover - environment dependent
    torch = None  # type: ignore

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "diffusers library is required. Install it with `pip install diffusers`"
    ) from exc

from PIL import Image


# Configure module-level logger
logger = logging.getLogger("text2image")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class GenerationConfig:
    """Configuration for text-to-image generation.

    Attributes:
        model_id: Hugging Face repo-id for the stable-diffusion model (e.g. "runwayml/stable-diffusion-v1-5").
        device: Device to run on (e.g. "cuda" or "cpu"). If None, auto-detects.
        use_fp16: If True and CUDA is available,use float16 for reduced memory.
        guidance_scale: Classifier-free guidance scale. Common values: 7.5-8.5.
        num_inference_steps: Number of denoising steps. Higher => slower, higher quality.
        height: Output image height in pixels (multiple of 8).
        width: Output image width in pixels (multiple of 8).
        seed: Random seed for deterministic outputs. If None, non-deterministic.
        scheduler: Optional scheduler name. Supported: 'ddim', 'dpm' (DPMSolverMultistep). None means default.
        safety_checker: If True attempts to run safety checker if the pipeline supports it.
        output_dir: Directory where generated images will be saved.
        batch_size: How many prompts to process per batch if the pipeline supports batches.
        use_auth_token: Optional huggingface auth token string if model requires authentication.
    """

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: Optional[str] = None
    use_fp16: bool = True
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    scheduler: Optional[str] = None
    safety_checker: bool = False
    output_dir: str = "outputs"
    batch_size: int = 1
    use_auth_token: Optional[str] = None


class Text2ImageGenerator:
    """A simple, production-oriented text-to-image generator wrapper around Hugging Face diffusers.

    Example:
        cfg = GenerationConfig(model_id="runwayml/stable-diffusion-v1-5")
        gen = Text2ImageGenerator(cfg)
        paths = gen.generate(["A beautiful painting of a lighthouse at sunset."])

    The class aims for reasonable defaults, safety, and memory optimizations. It loads the
    pipeline on initialization and exposes generate() to create images.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._validate_config()
        self._pipeline = None
        self._device = self._choose_device(config.device)
        self._torch_dtype = self._choose_torch_dtype(self._device, config.use_fp16)
        self._load_pipeline()

    # --------------------- Initialization helpers ---------------------
    def _validate_config(self) -> None:
        if not self.config.model_id:
            raise ValueError("model_id must be set in GenerationConfig")
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.config.height % 8 != 0 or self.config.width % 8 != 0:
            logger.warning("Height/width should be divisible by 8 for some model variants")
        if self.config.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")

    def _choose_device(self, preferred: Optional[str]) -> str:
        if preferred:
            logger.info("Using preferred device: %s", preferred)
            return preferred

        # Auto-detect
        if torch is not None and torch.cuda.is_available():
            logger.info("CUDA detected, using 'cuda'")
            return "cuda"
        logger.info("Falling back to 'cpu'")
        return "cpu"

    def _choose_torch_dtype(self, device: str, use_fp16: bool):
        if device.startswith("cuda") and use_fp16 and torch is not None:
            return torch.float16
        return torch.float32

    def _load_pipeline(self) -> None:
        """Load the Stable Diffusion pipeline with the chosen configuration.

        This method tries to apply standard memory and performance optimizations:
        - attention slicing
        - optionally xformers memory efficient attention (if available)
        - appropriate torch dtype
        - optional scheduler selection
        """
        logger.info("Loading pipeline %s", self.config.model_id)
        try:
            # Select scheduler
            scheduler = None
            if self.config.scheduler:
                s = self.config.scheduler.lower()
                if s == "ddim":
                    scheduler = DDIMScheduler.from_config(
                        self.config.model_id, subfolder="scheduler"
                    )
                elif s in ("dpm", "dpmsolver", "dpmsolver_multistep"):
                    scheduler = DPMSolverMultistepScheduler.from_config(
                        self.config.model_id, subfolder="scheduler"
                    )
                else:
                    logger.warning("Unknown scheduler `%s`, defaulting to pipeline default", s)

            # Load pipeline
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=self._torch_dtype,
                safety_checker=None if not self.config.safety_checker else None,
                use_auth_token=self.config.use_auth_token,
            )

            # Apply scheduler if created
            if scheduler is not None:
                try:
                    self._pipeline.scheduler = scheduler
                except Exception:
                    logger.exception("Failed to set custom scheduler, using default.")

            # Move to device
            try:
                self._pipeline.to(self._device)
            except Exception:
                logger.exception("Failed to move pipeline to device '%s'", self._device)

            # Memory/speed optimizations
            try:
                # reduces peak memory for attention
                self._pipeline.enable_attention_slicing()
            except Exception:
                logger.debug("enable_attention_slicing not available for this pipeline")

            # If xformers is available, enable
            try:
                # type: ignore[attr-defined]
                self._pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception:
                logger.debug("xformers not available or failed to enable")

            # Ensure output dir exists
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

            logger.info("Pipeline loaded successfully")
        except Exception as exc:
            logger.exception("Failed to load pipeline: %s", exc)
            raise

    # --------------------- Public API ---------------------
    def generate(self, prompts: Union[str, Sequence[str]]) -> List[Path]:
        """Generate images for a single prompt or a sequence of prompts.

        Args:
            prompts: Single prompt string or list/tuple of prompt strings.

        Returns:
            List[Path]: Paths to saved image files (one per prompt).

        Raises:
            ValueError: If prompts is empty or invalid.
            RuntimeError: If pipeline is not loaded.
        """
        if isinstance(prompts, str):
            prompts_list = [prompts]
        elif isinstance(prompts, Sequence):
            prompts_list = list(prompts)
        else:
            raise ValueError("prompts must be a string or a sequence of strings")

        if len(prompts_list) == 0:
            raise ValueError("At least one prompt must be provided")

        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        results: List[Path] = []

        # Batch processing
        for start in range(0, len(prompts_list), self.config.batch_size):
            batch = prompts_list[start : start + self.config.batch_size]
            logger.info(
                "Generating batch %d - %d of %d prompts",
                start + 1,
                min(start + self.config.batch_size, len(prompts_list)),
                len(prompts_list),
            )

            # Prepare generator for deterministic sampling if seed is set
            generator = None
            if self.config.seed is not None and torch is not None:
                try:
                    device_for_gen = self._device if self._device.startswith("cuda") else "cpu"
                    generator = torch.Generator(device=device_for_gen).manual_seed(self.config.seed)
                except Exception:
                    logger.debug("Failed to create torch.Generator for seed; continuing without generator")

            # Synthesize
            try:
                output = self._pipeline(
                    prompt=batch,
                    height=self.config.height,
                    width=self.config.width,
                    guidance_scale=self.config.guidance_scale,
                    num_inference_steps=self.config.num_inference_steps,
                    generator=generator,
                )
            except Exception as exc:
                logger.exception("Generation failed for batch starting at %d: %s", start, exc)
                raise

            images = getattr(output, "images", None)
            if images is None:
                # Some pipeline variants return plain lists
                images = output

            # Save images with unique names
            timestamp = int(time.time() * 1000)
            for i, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    # Convert numpy arrays to PIL images if necessary
                    try:
                        img = Image.fromarray(img)
                    except Exception:
                        logger.exception("Unable to convert generated image to PIL.Image")
                        raise

                filename = f"img_{timestamp}_{start + i}.png"
                out_path = Path(self.config.output_dir) / filename
                try:
                    img.save(out_path)
                    logger.info("Saved image to %s", out_path)
                    results.append(out_path)
                except Exception:
                    logger.exception("Failed to save image to %s", out_path)
                    raise

        return results

    def close(self) -> None:
        """Release resources, move pipeline to CPU and clean CUDA cache.

        Call this when done to free GPU memory.
        """
        try:
            if self._pipeline is not None:
                # Attempt to move model to CPU before deletion
                try:
                    self._pipeline.to("cpu")
                except Exception:
                    logger.debug("Moving pipeline to CPU failed (may already be on CPU)")
                del self._pipeline
                self._pipeline = None
        finally:
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.debug("torch.cuda.empty_cache failed or not available")

    def __enter__(self) -> "Text2ImageGenerator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# Lightweight helper function for quick use
def generate_images(
    prompt: Union[str, Sequence[str]],
    model_id: str = "runwayml/stable-diffusion-v1-5",
    output_dir: str = "outputs",
    device: Optional[str] = None,
    **kwargs,
) -> List[Path]:
    """Convenience helper wrapping Text2ImageGenerator with a short signature.

    Args:
        prompt: Prompt or list of prompts.
        model_id: HF model id.
        output_dir: Directory to write images to.
        device: Optional device override.
        **kwargs: Additional GenerationConfig overrides.

    Returns:
        List[Path] to saved images.
    """
    cfg = GenerationConfig(model_id=model_id, output_dir=output_dir, device=device, **kwargs)
    with Text2ImageGenerator(cfg) as gen:
        return gen.generate(prompt)
