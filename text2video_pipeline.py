import torch
from diffusers import VideoPipeline
from transformers import CLIPTextModel, CLIPTokenizer

class TextToVideoPipeline:
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initializes the TextToVideoPipeline with the specified model.

        Args:
            model_name (str): The name of the model to use for video generation.
            device (str): The device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.pipeline = VideoPipeline.from_pretrained(model_name).to(self.device)

    def generate_video(self, prompt: str, num_frames: int = 30, fps: int = 30):
        """
        Generates a video from a text prompt.

        Args:
            prompt (str): The text prompt to generate the video from.
            num_frames (int): The number of frames in the generated video.
            fps (int): The frames per second for the generated video.

        Returns:
            video: The generated video.
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate video
            video = self.pipeline(inputs.input_ids, num_frames=num_frames, fps=fps).videos

            return video
        except Exception as e:
            print(f"Error generating video: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    model_name = "your_model_name_here"  # Replace with your model name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_to_video = TextToVideoPipeline(model_name=model_name, device=device)
    prompt = "A beautiful sunset over the mountains"
    video = text_to_video.generate_video(prompt)

    if video is not None:
        print("Video generated successfully!")
        # Save or display the video as needed
    else:
        print("Failed to generate video.")