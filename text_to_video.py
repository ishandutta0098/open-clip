import torch
from diffusers import VideoPipeline
from transformers import AutoTokenizer
import os

def load_model(model_name: str):
    """
    Load the video generation model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        VideoPipeline: The loaded video generation model.
        AutoTokenizer: The tokenizer for the model.
    """
    try:
        print(f"Loading model: {model_name}")
        model = VideoPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to("cuda")  # Move model to GPU if available
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def generate_video(model, tokenizer, prompt: str, output_path: str, num_frames: int = 30):
    """
    Generate a video from a text prompt.

    Args:
        model (VideoPipeline): The video generation model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompt (str): The text prompt to generate the video from.
        output_path (str): The path to save the generated video.
        num_frames (int): The number of frames in the generated video.
    """
    try:
        print(f"Generating video for prompt: '{prompt}'")
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate video
        video = model(inputs.input_ids, num_frames=num_frames).videos
        
        # Save the video
        video[0].save(output_path)
        print(f"Video saved to: {output_path}")
    except Exception as e:
        print(f"Error generating video: {e}")
        raise

def main():
    model_name = "your-model-name-here"  # Replace with the actual model name
    prompt = "A beautiful sunset over the mountains"  # Example prompt
    output_path = "output_video.mp4"  # Output video file path

    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)

    # Generate the video
    generate_video(model, tokenizer, prompt, output_path)

if __name__ == "__main__":
    main()