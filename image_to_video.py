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
        if os.path.isfile(img_path):
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
        images (list): List of images to include in the video.
        output_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    if not images:
        print("No images to create video.")
        return

    height, width, layers = images[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in images:
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved to {output_path}")

def main(image_folder, output_video_path, fps=30):
    """
    Main function to convert images to video.

    Args:
        image_folder (str): Path to the folder containing images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    images = load_images_from_folder(image_folder)
    create_video_from_images(images, output_video_path, fps)

if __name__ == "__main__":
    # Example usage
    image_folder = "path/to/your/image/folder"  # Change this to your image folder path
    output_video_path = "output_video.mp4"  # Change this to your desired output video path
    fps = 30  # You can change the frames per second if needed

    main(image_folder, output_video_path, fps)