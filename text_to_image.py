import torch
from diffusers import DiffusionPipeline
import sys

def generate_image(prompt, model_name="CompVis/stable-diffusion-v1-4", output_path="output.png"):
    """
    Generate an image from a text prompt using the DiffusionPipeline.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_name (str): The name of the model to use for generation.
        output_path (str): The path where the generated image will be saved.
    """
    try:
        # Load the diffusion pipeline
        print(f"Loading model: {model_name}...")
        pipe = DiffusionPipeline.from_pretrained(model_name)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate the image
        print(f"Generating image for prompt: '{prompt}'...")
        image = pipe(prompt).images[0]

        # Save the generated image
        image.save(output_path)
        print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python text_to_image.py '<your prompt>' [model_name] [output_path]")
        sys.exit(1)

    prompt = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "CompVis/stable-diffusion-v1-4"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output.png"

    generate_image(prompt, model_name, output_path)