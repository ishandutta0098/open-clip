import os
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
import torch

def load_images_from_folder(folder):
    """
    Load images from a specified folder.

    Args:
        folder (str): The path to the folder containing images.

    Returns:
        list: A list of loaded images.
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: {img_path} is not a valid image.")
    return images

def generate_video_from_images(images, output_file, fps=30):
    """
    Generate a video from a list of images.

    Args:
        images (list): List of images to include in the video.
        output_file (str): The path to save the output video.
        fps (int): Frames per second for the output video.
    """
    if not images:
        print("No images to create video.")
        return

    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for img in images:
        video.write(img)

    video.release()
    print(f"Video saved as {output_file}")

def main(image_folder, output_video, model_name="CompVis/stable-diffusion-v1-4"):
    """
    Main function to convert images to video using Hugging Face Diffusers.

    Args:
        image_folder (str): Path to the folder containing images.
        output_video (str): Path to save the output video.
        model_name (str): Name of the model to use from Hugging Face.
    """
    # Load images
    images = load_images_from_folder(image_folder)

    # Load the diffusion model
    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Move model to GPU if available

    # Process images with the model
    processed_images = []
    for img in images:
        # Here we would typically process the image with the model
        # For demonstration, we will just append the original image
        # You can replace this with actual model inference
        processed_images.append(img)

    # Generate video from processed images
    generate_video_from_images(processed_images, output_video)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert images to video using Hugging Face Diffusers.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("output_video", type=str, help="Path to save the output video.")
    args = parser.parse_args()

    main(args.image_folder, args.output_video)