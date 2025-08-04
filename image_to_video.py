import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

def load_images(image_folder):
    """
    Load images from a specified folder.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        list: List of loaded images as PIL Image objects.
    """
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            images.append(Image.open(img_path))
    return images

def generate_video(images, output_file, fps=30):
    """
    Generate a video from a list of images.

    Args:
        images (list): List of images to include in the video.
        output_file (str): Path to save the output video file.
        fps (int): Frames per second for the video.
    """
    if not images:
        raise ValueError("No images to create video.")

    # Get dimensions from the first image
    width, height = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for img in images:
        # Convert PIL Image to numpy array
        frame = np.array(img)
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_file}")

def main(image_folder, output_file):
    """
    Main function to convert images to video.

    Args:
        image_folder (str): Path to the folder containing images.
        output_file (str): Path to save the output video file.
    """
    try:
        images = load_images(image_folder)
        generate_video(images, output_file)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    image_folder = "path/to/your/image/folder"  # Change this to your image folder path
    output_file = "output_video.mp4"  # Change this to your desired output video file name
    main(image_folder, output_file)