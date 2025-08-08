# -*- coding: utf-8 -*-
'''
text2image.py

A production-ready command-line utility to generate images from text prompts using Hugging Face Diffusers.

Features:
- Device selection (cuda, mps, cpu) with automatic fallback
- Mixed precision when available for performance and memory savings
- Configurable scheduler/model/steps/seed/guidance and batch size
- Input validation and safe defaults
- Outputs images and a JSON metadata file with generation parameters
- Requires HF_TOKEN for private model access if needed (via env var or CLI)

Usage example:
python text2image.py --prompt 'a steampunk robot reading a book' --output_dir outputs --num_images 3 --seed 42

Security: Do not commit HF_TOKEN to source control. Use environment variables or secure secret stores.
'''

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def setup_logging(verbose: bool = False) -> None:
    '''Configure root logger.'''
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    '''Parse CLI arguments.'''
    parser = argparse.ArgumentParser(description='Text-to-Image using Hugging Face Diffusers')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt to generate images from')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs'), help='Directory to write generated images')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='Hugging Face model repo id')
    parser.add_argument('--hf_token', type=str, default=os.environ.get('HF_TOKEN'), help='Hugging Face token (or set HF_TOKEN env var)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu', 'auto'], default='auto', help='Device to run on')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Classifier-free guidance scale')
    parser.add_argument('--height', type=int, default=512, help='Image height (must be divisible by 8)')
    parser.add_argument('--width', type=int, default=512, help='Image width (must be divisible by 8)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--scheduler', type=str, choices=['dpm', 'default'], default='dpm', help='Scheduler to use')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'auto'], default='auto', help='Computation precision')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for generation (kept small to reduce memory)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug logging')
    return parser.parse_args(argv)


def choose_device(requested: str) -> str:
    '''Decide which device string to use given requested preference.'''
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        try:
            # MPS is available on macOS with newer PyTorch builds
            if getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass
        return 'cpu'
    return requested


def validate_params(args: argparse.Namespace) -> None:
    '''Validate CLI parameters and raise ValueError for invalid ones.'''
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError('height and width must be divisible by 8')
    if args.num_images < 1:
        raise ValueError('num_images must be >= 1')
    if args.batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    if args.num_images < args.batch_size:
        logging.warning('num_images < batch_size; batch_size will be set to num_images')
        args.batch_size = args.num_images


def _seed_from_int(seed: int) -> torch.Generator:
    '''Return a torch.Generator seeded deterministically.'''
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def load_pipeline(model_id: str, device: str, precision: str = 'auto', hf_token: Optional[str] = None) -> StableDiffusionPipeline:
    '''Load and return a StableDiffusionPipeline with the requested device and precision.'''
    logging.info('Loading model %s onto %s (precision=%s)', model_id, device, precision)

    # Use DPM solver multistep scheduler by default for quality/speed tradeoff
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')

    # When possible, enable safe model offloading or memory-efficient attention via diffusers flags
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16 if precision == 'fp16' or (precision == 'auto' and device == 'cuda') else torch.float32,
            use_auth_token=hf_token,
        )
    except TypeError:
        # some versions use token= instead of use_auth_token
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16 if precision == 'fp16' or (precision == 'auto' and device == 'cuda') else torch.float32,
            token=hf_token,
        )

    # Move pipeline to device
    pipe = pipe.to(device)

    # Enable attention slicing to reduce peak memory usage (trades off a bit of speed)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        logging.debug('enable_attention_slicing not supported by pipeline version')

    # If using CUDA, enable memory efficient attention (if available)
    if device == 'cuda':
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.debug('Enabled xFormers memory efficient attention')
        except Exception:
            logging.debug('xFormers not available or failed to enable')

    return pipe


def prompt_to_filename(prompt: str, seed: Optional[int], index: int) -> str:
    '''Create a deterministic filename from prompt, seed and index to avoid invalid chars.'''
    digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:10]
    seed_part = f'{seed}' if seed is not None else 'nosd'
    return f'img_{digest}_s{seed_part}_i{index}.png'


def generate_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    output_dir: Path,
    num_images: int = 1,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    '''Generate images and write them to disk. Returns metadata per image.'''
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    device = next(pipe.parameters()).device if hasattr(pipe, 'parameters') else torch.device('cpu')
    device_str = str(device)
    logging.info('Generating %d image(s) on %s (batch_size=%d)', num_images, device_str, batch_size)

    # Base RNG
    base_seed = seed if seed is not None else int.from_bytes(os.urandom(2), 'big')

    generated = 0
    batch_index = 0
    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)

        # Create per-batch generator seeded deterministically for reproducibility
        batch_seed = base_seed + batch_index
        gen = _seed_from_int(batch_seed)

        logging.debug('Generating batch %d (size=%d) seed=%d', batch_index, current_batch, batch_seed)

        # Use autocast for mixed precision on CUDA
        try:
            context_manager = torch.autocast(device.type if hasattr(device, 'type') else str(device))
        except Exception:
            # Fallback if device not supported for autocast
            from contextlib import nullcontext

            context_manager = nullcontext()

        with context_manager:
            output = pipe(
                [prompt] * current_batch,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            )

        images = output.images

        for i, img in enumerate(images):
            idx = generated + i
            fname = prompt_to_filename(prompt, base_seed, idx)
            out_path = output_dir / fname
            try:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img.save(out_path)
            except Exception as e:
                logging.exception('Failed to save image %s: %s', out_path, e)
                continue

            meta = {
                'prompt': prompt,
                'model_id': getattr(pipe, 'model_id', 'unknown'),
                'seed': base_seed,
                'batch_seed': batch_seed,
                'index': idx,
                'file': str(out_path),
                'height': height,
                'width': width,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
            }
            results.append(meta)

        generated += current_batch
        batch_index += 1

    # Write metadata file
    meta_path = output_dir / 'generation_metadata.json'
    try:
        with meta_path.open('w', encoding='utf-8') as fh:
            json.dump({'results': results}, fh, indent=2)
        logging.info('Wrote metadata to %s', meta_path)
    except Exception:
        logging.exception('Failed to write metadata file')

    return results


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        validate_params(args)
    except Exception as e:
        logging.error('Invalid arguments: %s', e)
        return 2

    device = choose_device(args.device)

    # Determine precision
    precision = args.precision
    if precision == 'auto':
        precision = 'fp16' if device == 'cuda' else 'fp32'

    # Security: Ensure token is provided for private models
    if args.hf_token is None:
        logging.info('No HF token provided; attempting to load public models only')

    try:
        pipe = load_pipeline(args.model_id, device, precision=precision, hf_token=args.hf_token)
    except Exception:
        logging.exception('Failed to load pipeline; ensure model_id is correct and HF token is set for gated models')
        return 3

    try:
        meta = generate_images(
            pipe=pipe,
            prompt=args.prompt,
            output_dir=args.output_dir,
            num_images=args.num_images,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )
        logging.info('Generation completed; %d items produced', len(meta))
    except Exception:
        logging.exception('Generation failed')
        return 4

    return 0


if __name__ == '__main__':
    sys.exit(main())
