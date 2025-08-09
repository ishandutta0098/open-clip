#!/usr/bin/env python3
"""text2video.py
A production-ready text-to-video generator using HuggingFace Diffusers.

This script provides a robust, dependency-friendly approach to turning a text prompt
into a short video. It supports two modes:

1) TextToVideoPipeline (if available in the installed diffusers version): end-to-end
   text-to-video generation.
2) Fallback to per-frame image generation using StableDiffusionPipeline and stitching
   the frames into a video with FFmpeg-compatible libraries (imageio).

Key features:
- CLI-driven configuration with sensible defaults
- Device detection (CUDA if available, otherwise CPU)
- Graceful degradation if text-to-video models are not available
- Proper error handling, logging, and input validation
- Video encoding with libx264 and fallback to frame export if encoding fails
- Type hints and Google-style docstrings for maintainability

Note: For best results, provide a text-to-video capable model_id if you choose mode 1.
Otherwise, the script will fall back to per-frame diffusion and video assembly.
"""

from __future__ import annotations

import argparse
import errno
import os
import sys
import time
import logging
from typing import List, Optional

import torch
from PIL import Image
import numpy as np

# Optional imports are performed lazily to keep the CLI usable in minimal environments

LOG = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _try_import_text_to_video_pipeline():
    """Attempt to import HuggingFace's TextToVideoPipeline, if available.

    Returns a tuple of (available: bool, PipelineClass or None).
    """
    try:
        from diffusers import TextToVideoPipeline  # type: ignore

        return True, TextToVideoPipeline  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return False, None


def _try_import_stable_diffusion_pipeline():
    """Attempt to import Diffusers' StableDiffusionPipeline for image generation."""
    try:
        from diffusers import StableDiffusionPipeline  # type: ignore
        return StableDiffusionPipeline  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None


def _load_model_pipeline(model_id: Optional[str], use_text_to_video: bool,
                       device: str):
    """Load the appropriate diffusion pipeline based on user preference.

    - If use_text_to_video is True and TextToVideoPipeline is available, load it.
    - Otherwise, fall back to StableDiffusionPipeline for image frames.

    Args:
        model_id: Diffusion model identifier on HuggingFace Hub (optional for fallback).
        use_text_to_video: User preference to use TextToVideoPipeline when available.
        device: Target device, e.g., "cuda" or "cpu".

    Returns:
        A tuple (pipeline, mode) where mode is a string: "ttv" or "image".
    """
    available_ttv, TextToVideoPipeline = _try_import_text_to_video_pipeline()

    if use_text_to_video and available_ttv:
        if not model_id:
            raise ValueError(
                "model_id must be provided when using TextToVideoPipeline."
            )
        import torch as _torch  # local alias to avoid top-level import if unused
        # Lazy import guard; do not hard fail if CUDA isn't configured yet
        try:
            pipeline = TextToVideoPipeline.from_pretrained(model_id)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load TextToVideoPipeline model '{model_id}': {e}"
            )
        if _torch.cuda.is_available() and device == "cuda":
            pipeline = pipeline.to("cuda")
        else:
            pipeline = pipeline.to("cpu")
        return pipeline, "ttv"

    # Fallback: image-based StableDiffusionPipeline
    StableDiffusionPipeline = _try_import_stable_diffusion_pipeline()
    if StableDiffusionPipeline is None:
        raise RuntimeError(
            "No viable diffusion pipelines found. Install diffusers and models."
        )
    if not model_id:
        # Provide a safe default; users should supply a meaningful model for best results
        model_id = "stabilityai/stable-diffusion-2-1"
    # Use half-precision on CUDA when available to reduce memory usage
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"Failed to load StableDiffusionPipeline model '{model_id}': {e}"
        )
    pipeline = pipeline.to(device)
    return pipeline, "image"


def _generate_frames_with_ttv(pipeline, prompt: str, num_frames: int,
                              width: int, height: int, guidance_scale: float,
                              steps: int) -> List[Image.Image]:
    """Generate frames using TextToVideoPipeline.

    The exact output shape depends on the underlying pipeline implementation.
    We attempt to extract frames from common attributes.
    """
    # Call the pipeline. Different pipelines may expose different argument names.
    try:
        output = pipeline(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        )
    except TypeError:
        # Some pipelines may require slightly different param names
        output = pipeline(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
        )

    frames: List[Image.Image] = []
    if hasattr(output, "frames"):
        frames = list(output.frames)  # type: ignore
    elif hasattr(output, "images"):
        frames = list(output.images)  # type: ignore
    elif isinstance(output, dict) and "frames" in output:
        frames = list(output["frames"])  # type: ignore
    else:
        # Last resort: assume the output is a list/tuple of PIL Images
        if isinstance(output, (list, tuple)) and len(output) > 0:
            frames = list(output)  # type: ignore
        else:
            raise RuntimeError("Unexpected output from TextToVideoPipeline.")

    # Normalize to PIL Images
    normalized: List[Image.Image] = []
    for f in frames:
        if isinstance(f, Image.Image):
            normalized.append(f)
        else:
            # If it's a numpy array or other type, try to convert
            normalized.append(Image.fromarray(np.asarray(f)))
    return normalized


def _generate_frames_with_image_pipeline(pipeline, prompt: str, num_frames: int,
                                        width: int, height: int,
                                        guidance_scale: float,
                                        steps: int, seed: int) -> List[Image.Image]:
    """Generate frames using StableDiffusionPipeline, one frame per call with a seed."""
    frames: List[Image.Image] = []
    device = next(pipeline.parameters()).device if hasattr(pipeline, 'parameters') else torch.device("cpu")
    for i in range(num_frames):
        generator = torch.Generator(device).manual_seed(seed + i)
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
        frames.append(image)
    return frames


def _frames_to_video(frames: List[Image.Image], video_path: str, fps: int) -> None:
    """Encode a list of PIL Images into a video file using imageio (FFmpeg backend).

    Falls back to saving individual frames if video encoding fails.
    """
    import imageio  # local import to keep dependencies optional at module import time

    _ensure_dir(os.path.dirname(video_path) or '.')

    # Convert frames to RGB numpy arrays
    arrays: List[np.ndarray] = []
    for img in frames:
        if isinstance(img, Image.Image):
            arrays.append(np.asarray(img.convert("RGB")))
        else:
            arrays.append(np.asarray(img))

    try:
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264", ffmpeg_log_level="error")
        for arr in arrays:
            writer.append_data(arr)
        writer.close()
        LOG.info("Video saved to %s", video_path)
    except Exception as e:  # pragma: no cover
        LOG.error("Failed to encode video: %s", e)
        # Fallback: export frames as PNGs in a frames/ directory
        frames_dir = os.path.splitext(video_path)[0] + "_frames"
        _ensure_dir(frames_dir)
        for idx, arr in enumerate(arrays):
            frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
            Image.fromarray(arr).save(frame_path)
        LOG.info("Frames saved to %s as fallback.", frames_dir)


def _load_and_prepare_model(args) -> tuple:
    device = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    pipeline, mode = _load_model_pipeline(args.model_id, args.use_t2v, device)
    return pipeline, mode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text2video",
        description=(
            "Generate a video from a textual prompt using HuggingFace Diffusers. "
            "Supports TextToVideoPipeline when available; otherwise falls back to per-frame image diffusion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt describing the desired video content."
    )
    parser.add_argument(
        "--num_frames", type=int, default=60, help="Number of frames to generate for the video."
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Frames per second for the output video."
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Width of generated frames in pixels."
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of generated frames in pixels."
    )
    parser.add_argument(
        "--model_id", type=str, default=None, help="Diffusion model identifier on HuggingFace Hub."
    )
    parser.add_argument(
        "--use_t2v", action="store_true", help="Attempt to use TextToVideoPipeline if available."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Directory to save the video and artifacts."
    )
    parser.add_argument(
        "--output_name", type=str, default="text2video_output", help="Base name for the output video file."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed for frame generation (when not using TTV pipeline)."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Guidance scale for the diffusion model."
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps per frame."
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Forces CPU usage even if CUDA is available."
    )
    parser.set_defaults(use_t2v=False)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _setup_logging()

    parser = _build_parser()
    args = parser.parse_args(argv)

    _ensure_dir(args.output_dir)
    video_path = os.path.join(args.output_dir, f"{args.output_name}.mp4")

    device = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    LOG.info("Using device: %s", device)

    # Load model/pipeline depending on availability and preference
    try:
        pipeline, mode = _load_and_prepare_model(args)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to load model/pipeline: %s", exc)
        return 1

    # Generate frames
    try:
        if mode == "ttv" and args.use_t2v:
            LOG.info("Generating frames using TextToVideoPipeline...")
            frames = _generate_frames_with_ttv(
                pipeline,
                prompt=args.prompt,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance_scale,
                steps=args.steps,
            )
        else:
            LOG.info("Generating frames using StableDiffusionPipeline (image-based fallback)...")
            frames = _generate_frames_with_image_pipeline(
                pipeline,
                prompt=args.prompt,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance_scale,
                steps=args.steps,
                seed=args.seed,
            )
    except Exception as exc:  # pragma: no cover
        LOG.error("Frame generation failed: %s", exc)
        return 1

    # Save video or fall back to frames if encoding fails
    LOG.info("Encoding video to: %s", video_path)
    try:
        _frames_to_video(frames, video_path, args.fps)
    except Exception as exc:  # pragma: no cover
        LOG.error("Video encoding failed: %s", exc)
        return 1

    LOG.info("Done. Video available at: %s", video_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
