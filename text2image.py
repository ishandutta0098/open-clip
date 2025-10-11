'''text2image.py

CLI utility to generate images from text prompts using Hugging Face Diffusers.

Design goals:
- Safe defaults and input validation
- Clear CLI with reproducible options (seed, scheduler, guidance, steps)
- Performance: automatic use of GPU when available and optional mixed precision
- Security: encourages using HF token via env var, avoids embedding secrets
- Maintainability: modular functions, type hints, and comprehensive logging

Usage examples:
  python text2image.py --prompt "A serene beach at sunset" --out ./out.png --model runwayml/stable-diffusion-v1-5

Environment:
  Set HF_ACCESS_TOKEN env var or pass --huggingface-token to authenticate private models.
'''

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

try:
    import torch
    from diffusers import (
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        StableDiffusionPipeline,
    )
except Exception as exc:  # pragma: no cover - import/runtime environment dependent
    raise RuntimeError(
        "Required packages are not installed. Run `pip install -r requirements.txt`."\
    ) from exc


LOG = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: Optional[str]
    model_id: str
    out_path: Path
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    scheduler: str = "dpm"  # 'dpm' or 'euler'
    device: str = "auto"
    use_fp16: bool = True
    huggingface_token: Optional[str] = None


def setup_logging(level: int = logging.INFO) -> None:
    """Configure module-level logging.

    Args:
        level: Logging level (defaults to INFO).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


def validate_and_normalize_cfg(cfg: GenerationConfig) -> GenerationConfig:
    """Validate inputs and normalize values.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    if cfg.num_inference_steps <= 0 or cfg.num_inference_steps > 250:
        raise ValueError("num_inference_steps must be between 1 and 250.")

    if cfg.guidance_scale < 1.0 or cfg.guidance_scale > 30.0:
        raise ValueError("guidance_scale should be between 1.0 and 30.0.")

    # Width/height should be multiples of 8 for many models
    if cfg.width % 8 != 0 or cfg.height % 8 != 0:
        raise ValueError("width and height must be multiples of 8.")

    if cfg.width <= 0 or cfg.height <= 0 or cfg.width > 2048 or cfg.height > 2048:
        raise ValueError("width and height must be positive and <= 2048 to avoid OOM.")

    if cfg.seed is None:
        cfg.seed = int.from_bytes(os.urandom(2), "big")
        LOG.debug("No seed provided. Using random seed=%d", cfg.seed)
    elif cfg.seed < 0:
        raise ValueError("seed must be non-negative.")

    # Device selection
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")

    # If user requests fp16 but device is CPU, turn it off
    if cfg.device == "cpu" and cfg.use_fp16:
        LOG.warning("FP16 requested but running on CPU. Disabling fp16 for compatibility.")
        cfg.use_fp16 = False

    # Model id default
    if not cfg.model_id:
        cfg.model_id = "runwayml/stable-diffusion-v1-5"

    return cfg


def get_scheduler(name: str, model):
    """Return scheduler instance by name attached to the model's config.

    Supported: 'dpm' (DPMSolverMultistepScheduler) and 'euler' (EulerDiscreteScheduler)
    """
    name_lower = name.lower()
    if name_lower == "dpm":
        return DPMSolverMultistepScheduler.from_config(model.scheduler.config)
    if name_lower == "euler":
        return EulerDiscreteScheduler.from_config(model.scheduler.config)
    raise ValueError("Unsupported scheduler. Choose from 'dpm' or 'euler'.")


def build_pipeline(cfg: GenerationConfig) -> StableDiffusionPipeline:
    """Load and return a configured StableDiffusionPipeline.

    This function handles authentication, scheduler selection, device placement,
    precision management, and safety checker config.

    Args:
        cfg: GenerationConfig with settings.

    Returns:
        Configured StableDiffusionPipeline instance.
    """
    token = cfg.huggingface_token or os.getenv("HF_ACCESS_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        os.environ["HF_HOME"] = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache/huggingface"))

    # Load pipeline with offload/low-memory options if needed
    use_auth = token is not None
    pipe_kwargs = {
        "torch_dtype": (torch.float16 if cfg.use_fp16 and cfg.device == "cuda" else torch.float32),
    }

    LOG.info("Loading model %s... (this may take a while)", cfg.model_id)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            use_auth_token=token if use_auth else None,
            safety_checker=None,  # we will rely on upstream safety checker if needed; disabling for flexibility
            **pipe_kwargs,
        )
    except Exception as exc:
        LOG.exception("Failed to load pipeline for %s", cfg.model_id)
        raise

    # Attach chosen scheduler
    try:
        scheduler = get_scheduler(cfg.scheduler, pipe)
        pipe.scheduler = scheduler
    except Exception:
        LOG.debug("Using default scheduler provided by the model.")

    # Move to device
    device = torch.device(cfg.device)
    pipe.to(device)

    # Performance tuning flags
    if cfg.device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        except Exception:
            LOG.debug("Couldn't set CUDA performance flags.")

    return pipe


def seed_generator(seed: int) -> torch.Generator:
    """Return a torch.Generator seeded deterministically for reproducibility."""
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen.manual_seed(seed)
    return gen


def sanitize_filename(text: str) -> str:
    """Create a short filesystem-safe filename from prompt and timestamp."""
    # Hash the prompt and include a timestamp to avoid collisions.
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    ts = int(time.time())
    filename = f"img_{ts}_{h}.png"
    return filename


def generate_image(cfg: GenerationConfig) -> Path:
    """Generate an image given the configuration and return the output path.

    This function handles resource-managed generation, including optional autocast
    for fp16 on CUDA-enabled devices.
    """
    cfg = validate_and_normalize_cfg(cfg)

    LOG.info("Generating image with model=%s on device=%s", cfg.model_id, cfg.device)

    pipe = build_pipeline(cfg)

    generator = seed_generator(cfg.seed)

    # We'll optionally use torch.autocast to speed up fp16 inference on CUDA
    context = torch.autocast if (cfg.use_fp16 and cfg.device == "cuda") else (lambda *a, **k: torch.device("cpu"))

    # Use the pipeline to generate an image
    try:
        with torch.autocast(device_type=cfg.device, dtype=(torch.float16 if cfg.use_fp16 and cfg.device == "cuda" else torch.float32)) if (cfg.use_fp16 and cfg.device == "cuda") else torch.no_grad():
            output = pipe(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=cfg.num_inference_steps,
                height=cfg.height,
                width=cfg.width,
                generator=generator,
            )
    except Exception as exc:
        LOG.exception("Image generation failed.")
        raise

    if not output or not hasattr(output, "images") or len(output.images) == 0:
        raise RuntimeError("Pipeline returned no images.")

    pil_img: Image.Image = output.images[0]

    out_path = cfg.out_path
    if out_path.is_dir():
        out_file = out_path / sanitize_filename(cfg.prompt)
    else:
        out_file = out_path

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save image with reasonable defaults
    pil_img.save(out_file, format="PNG")
    LOG.info("Saved generated image to %s", out_file)

    return out_file


def parse_args(argv: Optional[list] = None) -> GenerationConfig:
    """Parse CLI args and return GenerationConfig.

    Args:
        argv: Optional list of arguments for testing. If None, uses sys.argv.

    Returns:
        GenerationConfig
    """
    parser = argparse.ArgumentParser(description="Text-to-Image generation using Hugging Face Diffusers")

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate an image from.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt to discourage elements.")
    parser.add_argument("--out", dest="out_path", type=Path, default=Path("./outputs"), help="Output file or directory path.")
    parser.add_argument("--model", dest="model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model repo id.")
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--scale", dest="guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8).")
    parser.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument("--scheduler", type=str, default="dpm", choices=["dpm", "euler"], help="Scheduler to use.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    parser.add_argument("--no-fp16", dest="use_fp16", action="store_false", help="Disable fp16 even if available.")
    parser.add_argument("--huggingface-token", dest="huggingface_token", type=str, default=None, help="HF token for private models (optional). Prefer using HF_ACCESS_TOKEN env var.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    cfg = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model_id=args.model_id,
        out_path=args.out_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        scheduler=args.scheduler,
        device=args.device,
        use_fp16=args.use_fp16,
        huggingface_token=args.huggingface_token,
    )

    return cfg


def main(argv: Optional[list] = None) -> int:
    """Entrypoint for the CLI. Returns exit code.

    Args:
        argv: Optional argv list for testing.

    Returns:
        int: exit code (0 success, non-zero error)
    """
    try:
        cfg = parse_args(argv)
        out = generate_image(cfg)
        LOG.info("Done. Image available at %s", out)
        return 0
    except Exception as exc:
        LOG.exception("Generation failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
