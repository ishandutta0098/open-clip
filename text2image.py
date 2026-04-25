import os
import logging
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    import torch
    from diffusers import DiffusionPipeline, LMSDiscreteScheduler, DPMSolverMultistepScheduler
    from PIL import Image
except Exception as e:  # pragma: no cover - explicit runtime dependency checks
    raise ImportError(
        "Missing runtime dependencies. Install required packages from requirements.txt. "
        "Import error: {}".format(e)
    )


logger = logging.getLogger("text2image")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(_handler)


@dataclass
class GenerationConfig:
    """Configuration for a single generation request.

    Attributes:
        prompt: The text prompt to turn into an image.
        height: Output image height in pixels. Must be a multiple of 8 (model constraint).
        width: Output image width in pixels. Must be a multiple of 8 (model constraint).
        num_inference_steps: Number of denoising steps. Tradeoff between quality and speed.
        guidance_scale: Classifier-free guidance scale. >1 for more faithful to prompt.
        seed: Optional integer seed. If None, uses non-deterministic generator.
    """

    prompt: str
    height: int = 512
    width: int = 512
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: Optional[int] = None


class Text2ImageGenerator:
    """A helper wrapper around Hugging Face diffusers for text-to-image generation.

    Design goals:
    - Simple programmatic API and CLI-friendly behavior.
    - Safe defaults (reasonable steps, resolution) and input validation.
    - Device-aware execution: uses CUDA if available and requested.
    - Clear logging and robust error handling.

    Usage example:
        gen = Text2ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")
        config = GenerationConfig(prompt="A colorful bird, digital art", height=512, width=512)
        images = gen.generate([config])
        images[0].save("out.png")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_safetensors: bool = True,
        revision: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        """Initialize the generator but do not download the model until load_pipeline is called.

        Args:
            model_id: Hugging Face model id or path for a compatible diffusers pipeline.
            device: Device string, e.g. 'cuda', 'cpu', 'mps'. If None, auto-selects.
            torch_dtype: Torch dtype to load model in (e.g., torch.float16 for GPU). If None, auto-selects.
            use_safetensors: Prefer safetensors weights if available.
            revision: Specific model revision to load (optional).
            hf_token: Hugging Face token for private models (optional). Will also check HUGGINGFACE_HUB_TOKEN env var.
        """
        self.model_id = model_id
        self.device = device or self._auto_select_device()
        self.torch_dtype = torch_dtype or self._suggest_dtype(self.device)
        self.use_safetensors = use_safetensors
        self.revision = revision
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.pipeline: Optional[DiffusionPipeline] = None
        logger.info("Text2ImageGenerator created with model=%s device=%s dtype=%s", model_id, self.device, self.torch_dtype)

    @staticmethod
    def _auto_select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        # Apple silicon
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _suggest_dtype(device: str) -> torch.dtype:
        # Use float16 on CUDA for memory/perf. Use float32 for CPU/MPS by default to avoid unsupported ops.
        if device == "cuda":
            return torch.float16
        return torch.float32

    def load_pipeline(self, scheduler: Optional[str] = "lms", use_auth_token: Optional[str] = None) -> None:
        """Load the diffusers pipeline. Idempotent: subsequent calls will be no-ops.

        Args:
            scheduler: Which scheduler to use. Options: 'lms', 'dpm'.
            use_auth_token: Legacy token param; if provided overrides hf_token.
        """
        if self.pipeline is not None:
            logger.debug("Pipeline already loaded")
            return

        token = use_auth_token or self.hf_token

        logger.info("Loading pipeline %s (scheduler=%s) on %s with dtype=%s", self.model_id, scheduler, self.device, self.torch_dtype)

        scheduler_obj = None
        if scheduler == "lms":
            scheduler_obj = LMSDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        elif scheduler == "dpm":
            scheduler_obj = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        else:
            logger.warning("Unknown scheduler '%s', falling back to LMSDiscreteScheduler", scheduler)
            scheduler_obj = LMSDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        # Safe loading arguments
        from_pretrained_kwargs = {
            "torch_dtype": self.torch_dtype,
        }
        if self.revision:
            from_pretrained_kwargs["revision"] = self.revision
        if self.use_safetensors:
            from_pretrained_kwargs["use_safetensors"] = True

        try:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                scheduler=scheduler_obj,
                safety_checker=None,  # Consumers should integrate their own safety checks where required
                requires_safety_checker=False,
                use_auth_token=token,
                **from_pretrained_kwargs,
            )
        except TypeError:
            # Older/newer diffusers versions may differ on args
            self.pipeline = DiffusionPipeline.from_pretrained(self.model_id, scheduler=scheduler_obj, **from_pretrained_kwargs)

        # Move to device
        try:
            self.pipeline = self.pipeline.to(self.device)
        except Exception as e:
            logger.warning("Pipeline.to(%s) failed: %s. Attempting CPU fallback.", self.device, e)
            self.pipeline = self.pipeline.to("cpu")
            self.device = "cpu"

        # Performance tweaks
        if self.device == "cuda" and self.torch_dtype == torch.float16:
            # Enable memory-efficient attention if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.debug("xFormers not available or failed to enable")

        logger.info("Pipeline loaded and moved to device: %s", self.device)

    @staticmethod
    def _validate_resolution(width: int, height: int) -> None:
        # Most current SD models require multiples of 8
        for name, v in (("width", width), ("height", height)):
            if v % 8 != 0:
                raise ValueError(f"{name} must be a multiple of 8, got {v}")
            if not (64 <= v <= 2048):
                # allow reasonable ranges; very large values may OOM
                logger.warning("%s is outside the typical recommended range (64-2048): %s", name, v)

    def generate(
        self,
        configs: List[GenerationConfig],
        batch_size: int = 1,
        scheduler: Optional[str] = "lms",
    ) -> List[Image.Image]:
        """Generate images from a list of GenerationConfig objects.

        Args:
            configs: A list of GenerationConfig instances. Each will produce a single image.
            batch_size: How many prompts to run per model call. Lower memory than single big batch.
            scheduler: scheduler to use when loading pipeline (if not already loaded)

        Returns:
            List of PIL.Image objects corresponding to each config in order.
        """
        if not configs:
            raise ValueError("configs must contain at least one GenerationConfig")

        # Ensure pipeline loaded
        self.load_pipeline(scheduler=scheduler)
        assert self.pipeline is not None, "Failed to initialize pipeline"

        results: List[Image.Image] = []

        # Validate each config
        for cfg in configs:
            if not cfg.prompt or not isinstance(cfg.prompt, str):
                raise ValueError("Each GenerationConfig.prompt must be a non-empty string")
            self._validate_resolution(cfg.width, cfg.height)

        # Process in batches
        for i in range(0, len(configs), batch_size):
            batch = configs[i : i + batch_size]
            prompts = [c.prompt for c in batch]
            # Use the first config of batch for common settings, but allow per-item seeds
            first = batch[0]

            # Create generator(s) with seeds for reproducibility when provided
            generators = []
            for c in batch:
                if c.seed is not None:
                    g = torch.Generator(device=self.device).manual_seed(c.seed)
                else:
                    # let diffusers create non-deterministic generators
                    g = None
                generators.append(g)

            # Using autocast for speed on cuda float16
            use_autocast = (self.device == "cuda" and self.torch_dtype == torch.float16)

            try:
                if use_autocast:
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=self.torch_dtype)
                else:
                    # dummy context manager
                    class _NullCtx:
                        def __enter__(self):
                            return None

                        def __exit__(self, exc_type, exc, tb):
                            return False

                    autocast_ctx = _NullCtx()

                with autocast_ctx:
                    # Build kwargs to pass to pipeline. Because pipelines expect either a single generator or none, we pass a list
                    pipeline_kwargs = {
                        "height": first.height,
                        "width": first.width,
                        "num_inference_steps": first.num_inference_steps,
                        "guidance_scale": first.guidance_scale,
                        # 'generator' can be a list of generators corresponding to each prompt (supported in many diffusers versions)
                        "generator": generators if any(g is not None for g in generators) else None,
                    }

                    images = self.pipeline(prompts=prompts, **{k: v for k, v in pipeline_kwargs.items() if v is not None}).images

                if not isinstance(images, list):
                    # Some pipeline implementations return a PIL.Image directly for single prompt
                    images = [images]

                # Validate outputs
                for img in images:
                    if not isinstance(img, Image.Image):
                        raise RuntimeError("Pipeline returned a non-image result: %r" % type(img))
                    results.append(img)

            except Exception as e:
                logger.exception("Failed to generate images for batch starting at index %s: %s", i, e)
                raise

        return results

    def save_images(self, images: List[Image.Image], output_dir: str, names: Optional[List[str]] = None, format: str = "PNG") -> List[str]:
        """Save a list of PIL.Image objects to disk.

        Args:
            images: Images to save.
            output_dir: Directory to save into. Will be created if missing.
            names: Optional list of filenames (without extension). If omitted, numeric names will be used.
            format: Image format for saving.

        Returns:
            List of absolute paths to saved images.
        """
        if not images:
            raise ValueError("No images to save")
        os.makedirs(output_dir, exist_ok=True)

        paths: List[str] = []
        for idx, img in enumerate(images):
            name = (names[idx] if names and idx < len(names) else f"image_{idx}")
            # Basic sanitization
            safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or f"image_{idx}"
            out_path = os.path.join(output_dir, f"{safe_name}.{format.lower()}")
            img.save(out_path, format=format)
            paths.append(os.path.abspath(out_path))
            logger.info("Saved image to %s", out_path)
        return paths


def _example_cli() -> None:
    """A minimal CLI for local testing. Keep dependencies light for production libraries to manage."""
    import argparse

    parser = argparse.ArgumentParser(description="Text2Image generator using Hugging Face diffusers")
    parser.add_argument("prompt", type=str, help="Prompt to generate")
    parser.add_argument("--out", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="HF model id")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--width", type=int, default=512, help="Image width (multiple of 8)")
    parser.add_argument("--height", type=int, default=512, help="Image height (multiple of 8)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic outputs")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cuda, cpu, mps")
    parser.add_argument("--kf", type=str, default=None, dest="hf_token", help="Hugging Face token (or set HUGGINGFACE_HUB_TOKEN)")

    args = parser.parse_args()

    g = Text2ImageGenerator(model_id=args.model, device=args.device, hf_token=args.hf_token)
    cfg = GenerationConfig(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    imgs = g.generate([cfg])
    saved = g.save_images(imgs, args.out, names=["result"]) if imgs else []
    print("Saved:", saved)


if __name__ == "__main__":
    _example_cli()
