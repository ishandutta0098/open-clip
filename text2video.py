import os
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageSequenceClip

class TextToVideo:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", output_dir="output_videos", fps=30):
        """
        Initializes the TextToVideo pipeline.

        Args:
            model_name (str): The name of the model to use for generating frames.
            output_dir (str): Directory to save the output video.
            fps (int): Frames per second for the output video.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.fps = fps
        self.pipeline = self.load_model()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        """
        Loads the Stable Diffusion model.

        Returns:
            StableDiffusionPipeline: The loaded model pipeline.
        """
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(self.model_name)
            pipeline.to("cuda")  # Move to GPU if available
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def generate_frames(self, text_prompt, num_frames=30):
        """
        Generates video frames from a text prompt.

        Args:
            text_prompt (str): The text prompt to generate frames.
            num_frames (int): The number of frames to generate.

        Returns:
            list: A list of generated frames as numpy arrays.
        """
        frames = []
        for i in range(num_frames):
            try:
                image = self.pipeline(text_prompt).images[0]
                frame = np.array(image)
                frames.append(frame)
            except Exception as e:
                print(f"Error generating frame {i}: {e}")
                continue
        return frames

    def save_video(self, frames, video_name="output_video.mp4"):
        """
        Saves the generated frames as a video.

        Args:
            frames (list): List of frames to save.
            video_name (str): The name of the output video file.
        """
        if not frames:
            raise ValueError("No frames to save.")

        video_path = os.path.join(self.output_dir, video_name)
        try:
            clip = ImageSequenceClip(frames, fps=self.fps)
            clip.write_videofile(video_path, codec='libx264')
            print(f"Video saved at: {video_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save video: {e}")

    def create_video(self, text_prompt, num_frames=30, video_name="output_video.mp4"):
        """
        Creates a video from a text prompt.

        Args:
            text_prompt (str): The text prompt to generate frames.
            num_frames (int): The number of frames to generate.
            video_name (str): The name of the output video file.
        """
        frames = self.generate_frames(text_prompt, num_frames)
        self.save_video(frames, video_name)

if __name__ == "__main__":
    # Example usage
    text_prompt = "A beautiful sunset over a mountain range"
    video_pipeline = TextToVideo()
    video_pipeline.create_video(text_prompt, num_frames=30, video_name="sunset_video.mp4")