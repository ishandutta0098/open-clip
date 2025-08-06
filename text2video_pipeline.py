import torch
from diffusers import VideoPipeline
from transformers import CLIPTextModel, CLIPTokenizer

class Text2VideoPipeline:
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initializes the Text2VideoPipeline with the specified model.

        Args:
            model_name (str): The name of the model to use for video generation.
            device (str): The device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.pipeline = VideoPipeline.from_pretrained(model_name).to(self.device)

    def generate_video(self, text_prompt: str, num_frames: int = 30, fps: int = 30):
        """
        Generates a video from the given text prompt.

        Args:
            text_prompt (str): The text prompt to generate the video from.
            num_frames (int): The number of frames in the generated video.
            fps (int): The frames per second for the generated video.

        Returns:
            video: The generated video.
        """
        try:
            # Tokenize the input text
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
            # Generate video
            video = self.pipeline(inputs, num_frames=num_frames, fps=fps).videos
            return video
        except Exception as e:
            print(f"An error occurred while generating video: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    model_name = "your_model_name_here"  # Replace with the actual model name
    text_prompt = "A beautiful sunset over the mountains"
    
    pipeline = Text2VideoPipeline(model_name=model_name)
    video = pipeline.generate_video(text_prompt=text_prompt)

    if video is not None:
        print("Video generated successfully!")
        # Save or display the video as needed
    else:
        print("Video generation failed.")