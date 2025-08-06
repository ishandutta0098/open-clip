import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os

class Text2VideoPipeline:
    def __init__(self, model_name: str, output_dir: str = "output_videos"):
        """
        Initializes the Text2VideoPipeline with the specified model.

        Args:
            model_name (str): The name of the model to use for video generation.
            output_dir (str): Directory to save generated videos.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.pipeline = self.load_model()
        self.create_output_dir()

    def load_model(self):
        """
        Loads the Stable Diffusion model for video generation.

        Returns:
            StableDiffusionPipeline: The loaded model pipeline.
        """
        try:
            print(f"Loading model: {self.model_name}")
            pipeline = StableDiffusionPipeline.from_pretrained(self.model_name)
            pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
            return pipeline
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def create_output_dir(self):
        """
        Creates the output directory if it does not exist.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def generate_video(self, text_prompt: str, num_frames: int = 30, frame_rate: int = 24):
        """
        Generates a video from a text prompt.

        Args:
            text_prompt (str): The text prompt to generate the video.
            num_frames (int): Number of frames in the video.
            frame_rate (int): Frame rate of the generated video.
        """
        try:
            print(f"Generating video for prompt: '{text_prompt}' with {num_frames} frames.")
            frames = []
            for i in range(num_frames):
                # Generate an image for each frame
                image = self.pipeline(text_prompt).images[0]
                frames.append(image)

            self.save_video(frames, frame_rate)
        except Exception as e:
            print(f"Error generating video: {e}")
            raise

    def save_video(self, frames: list, frame_rate: int):
        """
        Saves the generated frames as a video file.

        Args:
            frames (list): List of frames to save as a video.
            frame_rate (int): Frame rate of the video.
        """
        try:
            from moviepy.editor import ImageSequenceClip

            video_path = os.path.join(self.output_dir, "output_video.mp4")
            print(f"Saving video to: {video_path}")
            clip = ImageSequenceClip([np.array(frame) for frame in frames], fps=frame_rate)
            clip.write_videofile(video_path, codec='libx264')
            print("Video saved successfully.")
        except Exception as e:
            print(f"Error saving video: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    model_name = "CompVis/stable-diffusion-v-1-4"
    text_prompt = "A beautiful sunset over a mountain range"
    
    pipeline = Text2VideoPipeline(model_name)
    pipeline.generate_video(text_prompt)