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
        list: List of loaded images.
    """
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and img_path.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Unable to read image {img_path}")
    return images

def create_video_from_images(images, output_path, fps=30):
    """
    Create a video from a list of images.

    Args:
        images (list): List of images to include in the video.
        output_path (str): Path to save the output video.
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

def process_images_with_diffusers(images):
    """
    Process images using Hugging Face Diffusers.

    Args:
        images (list): List of images to process.

    Returns:
        list: List of processed images.
    """
    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    processed_images = []
    for img in images:
        # Convert the image to a format suitable for the model
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to("cuda")
        with torch.no_grad():
            processed_img = pipe(img_tensor).images[0]
        processed_images.append(np.array(processed_img.permute(1, 2, 0).cpu()))

    return processed_images

def main(input_folder, output_video_path):
    """
    Main function to convert images to video.

    Args:
        input_folder (str): Path to the folder containing images.
        output_video_path (str): Path to save the output video.
    """
    images = load_images_from_folder(input_folder)
    processed_images = process_images_with_diffusers(images)
    create_video_from_images(processed_images, output_video_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert images to video using Hugging Face Diffusers.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("output_video", type=str, help="Path to save the output video.")
    args = parser.parse_args()

    main(args.input_folder, args.output_video)