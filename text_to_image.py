import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def generate_image(prompt: str, output_path: str, model_id: str = "CompVis/stable-diffusion-v1-4", num_inference_steps: int = 50, guidance_scale: float = 7.5):
    """
    Generate an image from a text prompt using the Stable Diffusion model.

    Args:
        prompt (str): The text prompt to generate the image from.
        output_path (str): The path where the generated image will be saved.
        model_id (str): The model ID for the Stable Diffusion model. Default is "CompVis/stable-diffusion-v1-4".
        num_inference_steps (int): The number of inference steps for image generation. Default is 50.
        guidance_scale (float): The scale for classifier-free guidance. Default is 7.5.

    Raises:
        ValueError: If the output path is not valid or if the prompt is empty.
    """
    # Validate inputs
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if not output_path:
        raise ValueError("Output path cannot be empty.")

    # Load the Stable Diffusion model
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Move the model to GPU if available
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Generate the image
    try:
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")

    # Save the image
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create output directory if it doesn't exist
        image.save(output_path)
        print(f"Image saved to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save image: {e}")

if __name__ == "__main__":
    # Example usage
    prompt = "A fantasy landscape with mountains and a river"
    output_path = "output/generated_image.png"
    
    try:
        generate_image(prompt, output_path)
    except Exception as e:
        print(f"Error: {e}")