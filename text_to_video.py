import torch
from diffusers import VideoPipeline
from transformers import AutoTokenizer
import os

def generate_video_from_text(text, output_path, model_name="huggingface/VideoDiffusion"):
    """
    Generate a video from a given text prompt using the Hugging Face Diffusers library.

    Args:
        text (str): The text prompt to generate the video.
        output_path (str): The path where the generated video will be saved.
        model_name (str): The name of the model to use for video generation.

    Raises:
        ValueError: If the output path is not valid or if the text is empty.
    """
    # Validate inputs
    if not text:
        raise ValueError("The text prompt cannot be empty.")
    if not os.path.isdir(os.path.dirname(output_path)):
        raise ValueError(f"The output directory does not exist: {os.path.dirname(output_path)}")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    video_pipeline = VideoPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Generate the video
    print("Generating video...")
    video = video_pipeline(inputs["input_ids"], num_inference_steps=50).videos

    # Save the video
    video[0].save(output_path)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    text_prompt = "A beautiful sunset over the mountains"
    output_file = "output_video.mp4"

    try:
        generate_video_from_text(text_prompt, output_file)
    except Exception as e:
        print(f"An error occurred: {e}")