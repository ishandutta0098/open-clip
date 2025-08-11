#!/usr/bin/env python3
'''Text-to-image CLI using HuggingFace Diffusers'''
import argparse
import logging
import os
import re
import sys
import time
from typing import List, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


def slugify(text: str) -> str:
    '''Return a filesystem-safe slug from the given text.'''
    return re.sub(r'[^A-Za-z0-9]+', '-', text).strip('-').lower()


def load_pipeline(model_name: str, device: str, use_fp16: bool, token: Optional[str] = None, safety_checker: bool = True):
    '''Load a Stable Diffusion pipeline from pretrained model.

    Args:
        model_name: Identifier of the pretrained model to load from Hugging Face Hub.
        device: Target device, e.g., 'cpu' or 'cuda:0'.
        use_fp16: If True, cast model to FP16 for memory efficiency on CUDA.
        token: Optional auth token for private models.
        safety_checker: Whether to enable the safety checker.

    Returns:
        An initialized StableDiffusionPipeline.
    '''
    kwargs = {}
    if token:
        kwargs['use_auth_token'] = token
    dtype = torch.float16 if use_fp16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    pipe = pipe.to(device)
    if not safety_checker:
        pipe.safety_checker = lambda images, clip_input: (images, [False]*len(images))
    return pipe


def generate_images(pipe: StableDiffusionPipeline, prompt: str, negative_prompt: Optional[str], width: int, height: int,
                    steps: int, guidance_scale: float, seed: Optional[int], count: int) -> List[Image.Image]:
    '''Generate a list of images from the given prompt using the provided pipeline.

    Args:
        pipe: Initialized StableDiffusionPipeline.
        prompt: Text prompt for image generation.
        negative_prompt: Optional negative prompt to reduce unwanted aspects.
        width: Image width (pixels, multiples of 8 recommended).
        height: Image height (pixels, multiples of 8 recommended).
        steps: Number of diffusion inference steps.
        guidance_scale: Guidance scale (higher values encourage prompt adherence).
        seed: Optional random seed for reproducibility.
        count: Number of images to generate.

    Returns:
        List of PIL.Image.Image objects.
    '''
    images: List[Image.Image] = []
    if seed is None:
        seed = int(time.time())
    for i in range(count):
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed) + i)
        with torch.no_grad():
            image = pipe(
                prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        images.append(image)
    return images


def save_images(images: List[Image.Image], out_dir: str, prompt: str) -> List[str]:
    '''Save generated images to the output directory with deterministic names.

    Args:
        images: List of PIL images to save.
        out_dir: Directory to save images.
        prompt: Original prompt to derive a slug for filenames.

    Returns:
        List of saved file paths.
    '''
    os.makedirs(out_dir, exist_ok=True)
    slug = slugify(prompt)[:50]
    paths = []
    for idx, img in enumerate(images, start=1):
        filename = f'{slug}-{idx:02d}.png'
        path = os.path.join(out_dir, filename)
        img.save(path)
        paths.append(path)
    return paths


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='Text-to-image generation using HuggingFace Diffusers')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5', help='Pretrained model to load from Hugging Face Hub')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Directory to save generated images')
    parser.add_argument('--width', type=int, default=512, help='Image width in pixels (multiples of 8 recommended)')
    parser.add_argument('--height', type=int, default=512, help='Image height in pixels (multiples of 8 recommended)')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale for the generation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--neg', type=str, default=None, help='Negative prompt to reduce undesired aspects')
    parser.add_argument('--device', type=str, default=None, help='Computation device, e.g., "cuda:0" or "cpu"')
    parser.add_argument('--token', type=str, default=None, help='Hugging Face access token if required by the model')
    parser.add_argument('--safe', dest='safety', action='store_true', help='Enable safety processing (default: enabled)')
    parser.add_argument('--no-safe', dest='safety', action='store_false', help='Disable safety processing')
    parser.set_defaults(safety=True)

    args = parser.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_fp16 = device.startswith('cuda') and torch.cuda.is_available()

    logging.info('Loading model %s on device %s (FP16=%s, safety=%s)', args.model, device, use_fp16, args.safety)
    try:
        pipe = load_pipeline(args.model, device, use_fp16, token=args.token, safety_checker=args.safety)
    except Exception as exc:
        logging.critical('Failed to load pipeline: %s', exc)
        sys.exit(1)

    width = max(8, (args.width // 8) * 8)
    height = max(8, (args.height // 8) * 8)

    logging.info('Generating %d image(s) with prompt: %s', args.num_images, args.prompt)
    try:
        images = generate_images(
            pipe=pipe,
            prompt=args.prompt,
            negative_prompt=args.neg,
            width=width,
            height=height,
            steps=args.steps,
            guidance_scale=float(args.guidance),
            seed=args.seed,
            count=args.num_images
        )
    except Exception as exc:
        logging.critical('Image generation failed: %s', exc)
        sys.exit(1)

    saved_paths = save_images(images, args.out_dir, args.prompt)
    logging.info('Saved %d image(s) to %s', len(saved_paths), args.out_dir)
    for p in saved_paths:
        print(p)


if __name__ == '__main__':
    main()
