import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from huggingface_hub import login as hf_login


logger = logging.getLogger("text2image")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging to STDOUT and to a rotating file.

    Args:
        log_level: Logging level (e.g., "DEBUG", "INFO").
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (basic: append) to the repo's logs directory
    try:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler("logs/text2image.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        logger.warning("Could not create file logger: %s", e)


@dataclass
class GenerationConfig:
    prompt: str
    output: str
    model_id: str = "runwayml/stable-diffusion-v1-5"
    seed: Optional[int] = None
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    device: Optional[str] = None
    negative_prompt: Optional[str] = None
    hf_token: Optional[str] = None


def validate_config(cfg: GenerationConfig) -> None:
    """Validate user-supplied configuration.

    This performs lightweight validation and normalizes values.
    Throws ValueError on invalid configuration.
    """
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    if cfg.num_inference_steps <= 0 or cfg.num_inference_steps > 150:
        raise ValueError("num_inference_steps must be between 1 and 150")

    if not (1.0 <= cfg.guidance_scale <= 30.0):
        raise ValueError("guidance_scale must be between 1.0 and 30.0")

    if cfg.width % 8 != 0 or cfg.height % 8 != 0:
        raise ValueError("width and height must be multiples of 8 for many Stable Diffusion models")

    if cfg.seed is not None and cfg.seed < 0:
        raise ValueError("seed must be non-negative")

    if not cfg.output:
        raise ValueError("output path must be specified")

    outdir = os.path.dirname(cfg.output) or "."
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Could not create output directory {outdir}: {e}")


def choose_device(preferred: Optional[str] = None) -> str:
    """Select compute device: prefer GPU when available.

    Args:
        preferred: explicit device string (e.g., 'cpu', 'cuda'). If provided, attempt to use it.

    Returns:
        Device string usable by PyTorch: 'cuda' or 'cpu'.
    """
    if preferred:
        p = preferred.lower()
        if p == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested device 'cuda' but CUDA is not available; falling back to CPU")
            return "cpu"
        if p == "cpu":
            return "cpu"
        return p

    return "cuda" if torch.cuda.is_available() else "cpu"


def _maybe_login(hf_token: Optional[str]) -> None:
    """Login to Hugging Face hub if token provided.

    Args:
        hf_token: token string or None
    """
    if hf_token:
        try:
            hf_login(hf_token)
            logger.debug("Logged into Hugging Face Hub using provided token")
        except Exception as e:
            logger.warning("Failed to login to Hugging Face Hub: %s", e)


def load_pipeline(model_id: str, device: str, hf_token: Optional[str]) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline with recommended performance optimizations.

    Args:
        model_id: HF model id
        device: 'cuda' or 'cpu'
        hf_token: optional token to access gated models

    Returns:
        Instantiated and moved pipeline
    """
    dtype = torch.float32
    if device == "cuda":
        # prefer float16 on GPU to save VRAM/perf
        dtype = torch.float16

    _maybe_login(hf_token)

    logger.info("Loading model %s on %s (dtype=%s)", model_id, device, dtype)

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # intentionally disabled here; you may add a safety checker if required
            use_safetensors=True,
            revision="fp16" if device == "cuda" else None,
            use_auth_token=hf_token,
        )
    except TypeError:
        # some diffusers versions changed arguments; try a compatible fallback
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, use_auth_token=hf_token)

    # Move to device
    pipe = pipe.to(device)

    # Enable memory-efficient attention when available
    try:
        pipe.enable_attention_slicing()
        logger.debug("Enabled attention slicing for memory efficiency")
    except Exception:
        logger.debug("Attention slicing not available for this pipeline/version")

    # If GPU and supported, enable VAE tiling to reduce OOM risk
    try:
        pipe.enable_vae_tiling()
        logger.debug("Enabled VAE tiling")
    except Exception:
        logger.debug("VAE tiling not available for this pipeline/version")

    return pipe


def generate_image(cfg: GenerationConfig) -> Tuple[Image.Image, dict]:
    """Generate an image for a single prompt and return it plus generation metadata.

    This is synchronous and blocking. For batch/multi-image generation, call multiple times or extend.

    Args:
        cfg: GenerationConfig

    Returns:
        (PIL.Image, metadata dict)
    """
    device = cfg.device or choose_device(None)

    pipe = load_pipeline(cfg.model_id, device, cfg.hf_token)

    # Setup generator/seed for reproducibility
    generator = None
    if cfg.seed is not None:
        # Generator must be on the appropriate device
        gen_device = device if device == "cpu" else "cuda"
        try:
            generator = torch.Generator(device=gen_device).manual_seed(int(cfg.seed))
        except Exception:
            # fallback to cpu generator if device-specific generator fails
            generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    # Use autocast on CUDA to speed up and reduce memory if dtype is float16
    use_fp16 = (device == "cuda")

    prompt_args = dict(
        prompt=cfg.prompt,
        height=cfg.height,
        width=cfg.width,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
    )
    if cfg.negative_prompt:
        prompt_args["negative_prompt"] = cfg.negative_prompt

    logger.info("Generating image with prompt: %s", cfg.prompt)

    with torch.autocast("cuda") if (use_fp16 and device == "cuda") else torch.no_grad():
        result = pipe(**prompt_args)

    image = result.images[0]
    metadata = {
        "seed": cfg.seed,
        "model_id": cfg.model_id,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "width": cfg.width,
        "height": cfg.height,
    }

    return image, metadata


def save_image(image: Image.Image, path: str, metadata: Optional[dict] = None) -> None:
    """Save a PIL image to disk with basic atomic save semantics.

    Args:
        image: PIL.Image instance
        path: output file path (preferred extension .png or .jpg)
        metadata: optional dict to store as text metadata (not embedded in image)
    """
    tmp_path = f"{path}.tmp"
    try:
        image.save(tmp_path)
        os.replace(tmp_path, path)
        logger.info("Saved image to %s", path)

        # Write metadata next to the image (JSON-like text) for traceability
        if metadata is not None:
            meta_path = f"{path}.meta.txt"
            with open(meta_path, "w", encoding="utf-8") as f:
                for k, v in metadata.items():
                    f.write(f"{k}: {v}\n")
            logger.debug("Saved metadata to %s", meta_path)
    except Exception as e:
        logger.exception("Failed to save image: %s", e)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def parse_args(argv: Optional[list] = None) -> GenerationConfig:
    """Parse command-line arguments into a GenerationConfig.

    Returns:
        GenerationConfig instance populated from CLI.
    """
    ap = argparse.ArgumentParser(description="Text-to-image generator using Hugging Face Diffusers")

    ap.add_argument("--prompt", required=True, help="Text prompt describing the desired image")
    ap.add_argument("--output", required=True, help="Output image path (e.g., ./out.png)")
    ap.add_argument("--model-id", default=os.getenv("HF_MODEL_ID", "runwayml/stable-diffusion-v1-5"), help="Hugging Face model id to use")
    ap.add_argument("--seed", type=int, help="Deterministic seed (integer)")
    ap.add_argument("--steps", type=int, default=28, help="Number of inference steps (tradeoff quality/time)")
    ap.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    ap.add_argument("--width", type=int, default=512, help="Output image width (multiple of 8)")
    ap.add_argument("--height", type=int, default=512, help="Output image height (multiple of 8)")
    ap.add_argument("--device", default=None, help="Device to run on: 'cpu' or 'cuda' (auto-detect if omitted)")
    ap.add_argument("--negative", default=None, help="Negative prompt to guide away undesired content")
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token for gated models; can also be provided via HF_TOKEN env var")
    ap.add_argument("--log-level", default="INFO", help="Logging level")

    args = ap.parse_args(argv)

    cfg = GenerationConfig(
        prompt=args.prompt,
        output=args.output,
        model_id=args.model_id,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        device=args.device,
        negative_prompt=args.negative,
        hf_token=args.hf_token,
    )

    setup_logging(args.log_level)
    validate_config(cfg)

    # normalize device selection
    cfg.device = choose_device(cfg.device)

    logger.debug("Parsed config: %s", cfg)
    return cfg


def main(argv: Optional[list] = None) -> int:
    """Entrypoint for CLI use. Returns exit code (0 success).

    Args:
        argv: optional list of CLI args (for testing)
    """
    try:
        cfg = parse_args(argv)
        image, metadata = generate_image(cfg)
        save_image(image, cfg.output, metadata)
        logger.info("Generation complete: %s", cfg.output)
        return 0
    except Exception as exc:
        logger.exception("Command failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
