import torch
from diffusers import VideoPipeline
from transformers import AutoTokenizer
import os

def generate_video_from_text(text, output_path, model_name="google/videogen-1.0"):
    """
    Generate a video from the given text using the specified model.

    Args:
        text (str): The input text to generate a video for.
        output_path (str): The path where the generated video will be saved.
        model_name (str): The name of the model to use for video generation.

    Raises:
        ValueError: If the output path is not valid.
        RuntimeError: If video generation fails.
    """
    # Check if output path is valid
    if not os.path.isdir(os.path.dirname(output_path)):
        raise ValueError(f"Invalid output path: {output_path}")

    # Load the tokenizer and video pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    video_pipeline = VideoPipeline.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate video
    try:
        video = video_pipeline(**inputs).videos
        video[0].save(output_path)  # Save the first generated video
        print(f"Video saved to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Video generation failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    text_input = "A beautiful sunset over the mountains."
    output_file = "output_video.mp4"

    try:
        generate_video_from_text(text_input, output_file)
    except Exception as e:
        print(f"An error occurred: {str(e)}")