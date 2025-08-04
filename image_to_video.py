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
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: {img_path} is not a valid image file.")
    return images

def create_video_from_images(images, output_path, fps=30):
    """
    Create a video from a list of images.

    Args:
        images (list): List of images as numpy arrays.
        output_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video.
    """
    if not images:
        print("Error: No images to create video.")
        return

    height, width, layers = images[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved at {output_path}")

def main(image_folder, output_video_path):
    """
    Main function to convert images to video.

    Args:
        image_folder (str): Path to the folder containing images.
        output_video_path (str): Path where the output video will be saved.
    """
    images = load_images_from_folder(image_folder)
    create_video_from_images(images, output_video_path)

if __name__ == "__main__":
    # Example usage
    image_folder = "path/to/your/image/folder"  # Change this to your image folder path
    output_video_path = "output/video.mp4"      # Change this to your desired output path

    main(image_folder, output_video_path)