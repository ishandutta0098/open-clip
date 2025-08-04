import os
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
import argparse

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Unable to read image {img_path}")
    return images

def create_video_from_images(images, output_path, fps=30):
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
    # Initialize the Hugging Face Diffusers pipeline
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4")
    
    processed_images = []
    for img in images:
        # Convert the image to a format suitable for the pipeline
        img_tensor = np.array(img) / 255.0  # Normalize to [0, 1]
        img_tensor = img_tensor.astype(np.float32)

        # Process the image (this is a placeholder for actual processing)
        # Here you would typically call the pipeline with the image tensor
        # processed_img = pipe(img_tensor)  # Example call, adjust as necessary

        # For demonstration, we will just append the original image
        processed_images.append(img)

    return processed_images

def main(input_folder, output_video, fps):
    images = load_images_from_folder(input_folder)
    processed_images = process_images_with_diffusers(images)
    create_video_from_images(processed_images, output_video, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to video using Hugging Face Diffusers.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("output_video", type=str, help="Path to save the output video.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    
    args = parser.parse_args()
    
    main(args.input_folder, args.output_video, args.fps)