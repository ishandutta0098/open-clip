#!/usr/bin/env python3
"""
Text-to-video generation script using Hugging Face Diffusers.

This script attempts to use a Diffusers TextToVideoPipeline when available.
If a TextToVideoPipeline is not present in the installed diffusers version,
it gracefully falls back to per-frame generation using a Stable Diffusion
pipeline with different seeds to produce a sequence of frames that can be
assembled into a video.

Features
- Lightweight, CLI-driven text-to-video generation
- Prefer modern TextToVideoPipeline when present; otherwise, per-frame SD-based fallback
- Outputs an MP4 video using imageio-ffmpeg backend
- Type hints, Google-style docstrings, error handling, and logging
- Configurable height/width, frames, fps, seeds, guidance scale, and steps

Notes
- Real-time quality and continuity depend on the underlying model. For best results,
  use a model explicitly designed for text-to-video (e.g., a Diffusers TextToVideo model).
- On CPU, generation will be slow; enable CUDA if available.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Iterable, List

import numpy as np
import torch
from PIL import Image

# Optional imports guarded for environments without diffusers installed
HAS_T2V = False
TextToVideoPipeline = None
StableDiffusionPipeline = None
try:
    from diffusers import TextToVideoPipeline  # type: ignore
    HAS_T2V = True
except Exception as exc:
    logging.getLogger("text2video").debug("TextToVideoPipeline not available: %s", exc)

try:
    from diffusers import StableDiffusionPipeline  # type: ignore
except Exception as exc:
    StableDiffusionPipeline = None
    logging.getLogger("text2video").debug("StableDiffusionPipeline import failed: %s", exc)

import imageio  # type: ignore

# Type aliases for clarity
Device = torch.device
ImageLike = Image.Image  # PIL Image


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _to_numpy(frame: Any) -> np.ndarray:
    """Convert a frame to a NumPy array with shape (H, W, 3) and dtype uint8."""
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    if isinstance(frame, np.ndarray):
        if frame.dtype != np.uint8:
            # Assure correct dtype
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        if frame.ndim == 2:
            # Grayscale to RGB fallback
            frame = np.stack([frame] * 3, axis=-1)
        return frame
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (3, 1):
            # LikelyCHW format; convert to HWC
            arr = np.transpose(arr, (1, 2, 0))
        return arr
    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _write_frames_to_video(frames: List[Any], fps: int, output_path: str) -> None:
    """Encode frames to an MP4 video using imageio.

    frames can be PIL Images, NumPy arrays (H, W, 3) or torch.Tensors.
    The function will attempt to convert all frames to uint8 RGB NumPy arrays.
    """
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", format="mp4")
    try:
        for idx, f in enumerate(frames):
            arr = _to_numpy(f)
            writer.append_data(arr)
            if idx % 10 == 0:
                logging.info("Appended frame %d", idx)
    finally:
        writer.close()


def _load_pipeline(model_id: str, device: Device) -> Any:
    """Load a suitable diffusion pipeline based on availability.

    Preference order:
    1) TextToVideoPipeline from diffusers, if available
    2) Fallback to StableDiffusionPipeline for per-frame generation
    """
    if HAS_T2V and TextToVideoPipeline is not None:
        # Use half-precision on CUDA when possible to save memory
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipeline = TextToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipeline = pipeline.to(device)
        logging.info("Loaded TextToVideoPipeline from %s on %s", model_id, device)
        return pipeline

    if StableDiffusionPipeline is None:
        raise RuntimeError("No suitable diffusion pipeline found. Install diffusers with a supported model.")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    logging.info("Loaded StableDiffusionPipeline from %s on %s (per-frame fallback)", model_id, device)
    return pipeline


def _generate_frames_with_pipeline(
    pipeline: Any,
    prompt: str,
    num_frames: int,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    device: Device,
    seed: int,
) -> List[Any]:
    frames: List[Any] = []
    # If the pipeline supports a dedicated text-to-video call, use it directly
    if HAS_T2V and getattr(pipeline, "__class__", object).__name__ == "TextToVideoPipeline":
        try:
            result = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            # The API may return an object with .frames or a dict with 'frames'
            frames = getattr(result, "frames", None)
            if frames is None:
                frames = getattr(result, "images", None)
            if frames is None and isinstance(result, dict):
                frames = result.get("frames") or result.get("images")
            if frames is None:
                raise RuntimeError("TextToVideoPipeline did not return frames as expected.")
            logging.info("Generated %d frames using TextToVideoPipeline", len(frames))
            return frames
        except TypeError:
            # Fallback to per-frame generation if the API signature differs
            logging.warning("TextToVideoPipeline invocation signature mismatch; using per-frame fallback.")

    # Fall back to per-frame generation with a Stable Diffusion-like model
    generator_base = torch.Generator(device=device).manual_seed(seed)
    for i in range(num_frames):
        gen = torch.Generator(device=device).manual_seed(seed + i)
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                generator=gen,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images[0]
        frames.append(image)
        logging.debug("Generated frame %d", i + 1)
    return frames


def _ensure_positive(value: int | float, name: str) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be a positive number.")


def _validate_dims(height: int, width: int) -> None:
    # Most diffusion models require height/width to be multiples of 8
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("Height and width must be multiples of 8 for most diffusion models.")


def parse_args() -> argparse Namespace:  # type: ignore[return-type]
    parser = argparse.ArgumentParser(
        prog="text2video",
        description=(
            "Generate a short video from a text prompt using Hugging Face Diffusers. "
            "Preferred: TextToVideoPipeline when available; otherwise per-frame SD-based fallback."
        ),
    )

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the desired video content.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help=(
            "HuggingFace model_id to load for diffusion. e.g., 'diffusers/xxx-text2video' or a SD model."
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames in the output video (default: 16).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Video frame height in pixels (must be multiple of 8).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video frame width in pixels (must be multiple of 8).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output video frames per second (fps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for frame generation (useful for reproducibility).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale controlling the prompt strength (typically 7.0-9.0).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps per frame (or per batch for TextToVideoPipeline).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="text2video_output.mp4",
        help="Path to save the output video (MP4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device to use (default: auto-detect).",
    )

    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    # Resolve device
    device: Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info("Using device: %s", device)

    # Validate inputs
    _ensure_positive(args.num_frames, "num_frames")
    _ensure_positive(args.height, "height")
    _ensure_positive(args.width, "width")
    _validate_dims(args.height, args.width)

    # Load model/pipeline
    if not os.path.exists(os.path.dirname(os.path.abspath(args.output)) or "."):
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    try:
        pipeline = _load_pipeline(args.model_id, device)
    except Exception as exc:
        logging.error("Failed to load model '%s': %s", args.model_id, exc)
        return 1

    # Generate frames
    logging.info(
        "Starting generation: prompt_len=%d frames=%d size=%dx%d fps=%d seed=%d",
        len(args.prompt), args.num_frames, args.height, args.width, args.fps, args.seed,
    )

    try:
        frames = _generate_frames_with_pipeline(
            pipeline=pipeline,
            prompt=args.prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            device=device,
            seed=args.seed,
        )
    except Exception as exc:
        logging.error("Frame generation failed: %s", exc)
        return 1

    # Write video
    try:
        _write_frames_to_video(frames, args.fps, args.output)
        logging.info("Video saved to %s", args.output)
    except Exception as exc:
        logging.error("Failed to write video: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
