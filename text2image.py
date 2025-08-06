import torch
from diffusers import StableDiffusionPipeline
import argparse

def generate_image(prompt: str, output_path: str, model_id: str = "CompVis/stable-diffusion-v1-4", num_inference_steps: int = 50):
    """
    Generate an image from a text prompt using the Stable Diffusion model.

    Args:
        prompt (str): The text prompt to generate an image from.
        output_path (str): The path where the generated image will be saved.
        model_id (str): The model ID from Hugging Face Hub. Default is "CompVis/stable-diffusion-v1-4".
        num_inference_steps (int): The number of inference steps for image generation. Default is 50.
    """
    try:
        # Load the Stable Diffusion model
        print("Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Move the model to GPU if available

        # Generate the image
        print(f"Generating image for prompt: '{prompt}'")
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

        # Save the generated image
        image.save(output_path)
        print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using Stable Diffusion.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate an image from.")
    parser.add_argument("output_path", type=str, help="The path where the generated image will be saved.")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4", help="Model ID from Hugging Face Hub.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for image generation.")

    args = parser.parse_args()

    # Generate the image
    generate_image(args.prompt, args.output_path, args.model_id, args.num_inference_steps)

if __name__ == "__main__":
    main()