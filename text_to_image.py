import torch
from diffusers import StableDiffusionPipeline
import sys

def generate_image(prompt, output_path, model_id="CompVis/stable-diffusion-v1-4"):
    """
    Generate an image from a text prompt using the Stable Diffusion model.

    Args:
        prompt (str): The text prompt to generate the image from.
        output_path (str): The file path to save the generated image.
        model_id (str): The model ID for the Stable Diffusion model.

    Raises:
        ValueError: If the prompt is empty.
        RuntimeError: If the image generation fails.
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    try:
        # Load the Stable Diffusion model
        print("Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Move the model to GPU if available

        # Generate the image
        print(f"Generating image for prompt: '{prompt}'...")
        image = pipe(prompt).images[0]

        # Save the image
        image.save(output_path)
        print(f"Image saved to {output_path}")

    except Exception as e:
        raise RuntimeError(f"An error occurred during image generation: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python text_to_image.py '<prompt>' '<output_path>'")
        sys.exit(1)

    prompt = sys.argv[1]
    output_path = sys.argv[2]

    try:
        generate_image(prompt, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)