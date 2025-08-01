import torch
from diffusers import DiffusionPipeline
import sys

def generate_image(prompt: str, model_id: str = "CompVis/stable-diffusion-v1-4", num_inference_steps: int = 50, guidance_scale: float = 7.5):
    """
    Generate an image from a text prompt using the DiffusionPipeline.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str): The model ID from Hugging Face Hub. Default is "CompVis/stable-diffusion-v1-4".
        num_inference_steps (int): The number of inference steps for the diffusion process. Default is 50.
        guidance_scale (float): The scale for classifier-free guidance. Default is 7.5.

    Returns:
        PIL.Image: The generated image.
    """
    try:
        # Load the diffusion pipeline
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Move the pipeline to GPU if available

        # Generate the image
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        return image

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python text_to_image.py '<your prompt>'")
        sys.exit(1)

    prompt = sys.argv[1]
    image = generate_image(prompt)

    # Save the generated image
    image.save("generated_image.png")
    print("Image saved as 'generated_image.png'")

if __name__ == "__main__":
    main()