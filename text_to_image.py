import torch
from diffusers import DiffusionPipeline
import argparse

def generate_image(prompt: str, model_name: str = "CompVis/stable-diffusion-v1-4", num_inference_steps: int = 50, guidance_scale: float = 7.5):
    """
    Generate an image from a text prompt using the DiffusionPipeline.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_name (str): The model name to use for generation.
        num_inference_steps (int): The number of inference steps for the diffusion process.
        guidance_scale (float): The scale for classifier-free guidance.

    Returns:
        PIL.Image: The generated image.
    """
    try:
        # Load the diffusion pipeline
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Move the pipeline to GPU if available

        # Generate the image
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        return image

    except Exception as e:
        print(f"An error occurred while generating the image: {e}")
        return None

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using DiffusionPipeline.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate the image from.")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="The model name to use for generation.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="The number of inference steps for the diffusion process.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="The scale for classifier-free guidance.")
    
    args = parser.parse_args()

    # Generate the image
    image = generate_image(args.prompt, args.model_name, args.num_inference_steps, args.guidance_scale)

    if image:
        # Save the image
        image.save("generated_image.png")
        print("Image generated and saved as 'generated_image.png'.")

if __name__ == "__main__":
    main()