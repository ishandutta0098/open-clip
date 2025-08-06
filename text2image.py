import torch
from diffusers import StableDiffusionPipeline
import argparse
import os

def generate_image(prompt, output_dir, model_name="CompVis/stable-diffusion-v1-4"):
    """
    Generate an image from a text prompt using a diffusion model.

    Args:
        prompt (str): The text prompt to generate the image from.
        output_dir (str): The directory to save the generated image.
        model_name (str): The name of the pre-trained model to use.

    Returns:
        None
    """
    # Check if the output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    # Load the pre-trained model
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Move the model to GPU if available
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Generate the image
    try:
        image = pipe(prompt).images[0]
    except Exception as e:
        print(f"Error generating image: {e}")
        return

    # Save the generated image
    image_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
    try:
        image.save(image_path)
        print(f"Image saved at: {image_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using a diffusion model.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate the image from.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the generated image.")
    args = parser.parse_args()

    # Generate the image
    generate_image(args.prompt, args.output_dir)

if __name__ == "__main__":
    main()