import os
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
import torch

def load_images_from_folder(folder):
    """
    Load images from a specified folder.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        list: List of images loaded as numpy arrays.
    """
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: {img_path} is not a valid image.")
    return images

def create_video_from_images(images, output_path, fps=30):
    """
    Create a video from a list of images.

    Args:
        images (list): List of images (numpy arrays).
        output_path (str): Path where the video will be saved.
        fps (int): Frames per second for the video.
    """
    if not images:
        print("No images to create video.")
        return

    height, width, layers = images[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved at {output_path}")

def generate_images_with_diffusers(prompt, num_images=5):
    """
    Generate images using Hugging Face Diffusers.

    Args:
        prompt (str): Text prompt for image generation.
        num_images (int): Number of images to generate.

    Returns:
        list: List of generated images as numpy arrays.
    """
    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    images = []
    for _ in range(num_images):
        image = pipe(prompt).images[0]
        images.append(np.array(image))
    
    return images

def main():
    # Example usage
    prompt = "A beautiful landscape"
    num_images = 5
    output_video_path = "output_video.mp4"

    # Generate images
    print("Generating images...")
    images = generate_images_with_diffusers(prompt, num_images)

    # Create video from generated images
    print("Creating video...")
    create_video_from_images(images, output_video_path)

if __name__ == "__main__":
    main()