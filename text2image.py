import argparse
import logging
import os
import sys
from typing import List, Optional

import torch
from PIL import Image

try:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
except Exception as e:
    raise SystemExit(
        "Failed to import diffusers. Ensure you have: 'pip install diffusers[torch]'. Error: {}".format(e)
    )


def load_pipeline(model_id: str, device: str) -> StableDiffusionPipeline:
    """
    Load the Stable Diffusion pipeline for a given model_id on the specified device.

    This function selects an appropriate torch dtype based on the device, attempts to
    enable an efficient scheduler, and moves the model to the requested device.

    Args:
        model_id: The HuggingFace model identifier to load (e.g., "runwayml/stable-diffusion-v1-5").
        device: Target device to run on. Can be 'cpu' or 'cuda' or 'auto' (resolved by caller).

    Returns:
        A prepared StableDiffusionPipeline instance.
    """
    # Choose appropriate tensor precision
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

    if device != "cpu":
        pipe.to(device)

    # Prefer a faster scheduler if available
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.config)
        logging.info("DPMSolverMultistepScheduler loaded for faster sampling.")
    except Exception:
        # If not available, fall back to default scheduler
        logging.info("DPMSolverMultistepScheduler not available; using default scheduler.")
        pass

    return pipe


def disable_safety(pipe: StableDiffusionPipeline) -> None:
    """
    Disable the safety checker of the diffusion pipeline.

    Note: Disabling safety checks can yield unsafe or inappropriate outputs. Use with caution
    and only in trusted environments.

    Args:
        pipe: Instantiated StableDiffusionPipeline.
    """
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = lambda images, clip_input: (images, [False for _ in images])
    else:
        logging.warning("The loaded pipeline does not expose a safety_checker; nothing to disable.")


def generate_images(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    output_dir: str,
    negative_prompt: Optional[str] = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seeds: Optional[List[int]] = None,
    device: str = "cpu",
) -> List[str]:
    """
    Generate images for a list of prompts and save them to disk.

    Args:
        pipe: The Diffusers StableDiffusionPipeline instance to use.
        prompts: List of text prompts to generate images for.
        output_dir: Directory where images will be saved.
        negative_prompt: Optional negative prompt to steer generation away from.
        height: Desired image height in pixels.
        width: Desired image width in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Guidance scale for prompt adherence.
        seeds: Optional list of integer seeds for reproducibility per prompt.
        device: Target device ('cpu' or 'cuda') for generator initialization.

    Returns:
        List of absolute file paths to the generated images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Created output directory: %s", output_dir)

    results: List[str] = []

    for idx, prompt in enumerate(prompts):
        seed = seeds[idx] if seeds is not None and idx < len(seeds) else None
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device if device else "cpu").manual_seed(seed)

        image = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        out_path = os.path.join(output_dir, f"image_{idx + 1}.png")
        image.save(out_path)
        results.append(out_path)
        logging.info("Generated image for prompt %d saved to %s", idx + 1, out_path)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-image generation using Hugging Face Diffusers (Stable Diffusion).")
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model identifier to load (default: runwayml/stable-diffusion-v1-5).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Primary text prompt to guide image generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt to reduce undesired features.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps per image.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (higher values encourage prompt adherence).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height in pixels.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width in pixels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible results. If omitted, a random seed will be used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'cpu', 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--disable-safety",
        action="store_true",
        help="Disable the safety checker (not recommended for production).",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Resolve device preference
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logging.info("Initializing model '%s' on device '%s'", args.model_id, device)
    try:
        pipe = load_pipeline(args.model_id, device)
    except Exception as exc:
        logging.error("Failed to load model: %s", exc)
        sys.exit(1)

    if args.disable_safety:
        disable_safety(pipe)

    # Prepare prompts; replicate the same prompt for the requested number of outputs
    prompts = [args.prompt] * max(1, args.num_outputs)
    seeds = [args.seed] * max(1, args.num_outputs) if args.seed is not None else None

    try:
        paths = generate_images(
            pipe=pipe,
            prompts=prompts,
            output_dir=args.output_dir,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seeds=seeds,
            device=device,
        )
        logging.info("Generated %d image(s).", len(paths))
        print("Images saved:", ", ".join(paths))
    except Exception as exc:
        logging.error("Image generation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
