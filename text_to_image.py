import torch
from diffusers import StableDiffusionPipeline
import argparse

def generate_image(prompt, model_id="CompVis/stable-diffusion-v1-4", output_path="output.png"):
    """
    Generate an image from a text prompt using the Stable Diffusion model.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str): The model ID for the Stable Diffusion model.
        output_path (str): The path where the generated image will be saved.
    """
    try:
        # Load the Stable Diffusion pipeline
        print("Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Move the pipeline to GPU if available

        # Generate the image
        print(f"Generating image for prompt: '{prompt}'")
        image = pipe(prompt).images[0]

        # Save the generated image
        image.save(output_path)
        print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using Stable Diffusion.")
    parser.add_argument("prompt", type=str, help="The text prompt for image generation.")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4", help="Model ID for Stable Diffusion.")
    parser.add_argument("--output", type=str, default="output.png", help="Output path for the generated image.")

    args = parser.parse_args()

    # Generate the image based on the provided prompt
    generate_image(args.prompt, model_id=args.model_id, output_path=args.output)

if __name__ == "__main__":
    main()