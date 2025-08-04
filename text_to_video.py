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
        AutoTokenizer: The tokenizer associated with the model.
    """
    try:
        # Load the video generation model
        model = VideoPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        model = model.to("cuda")  # Move model to GPU if available
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def generate_video(model, tokenizer, text_prompt: str, output_path: str):
    """
    Generate a video from a text prompt using the provided model.

    Args:
        model (VideoPipeline): The video generation model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        text_prompt (str): The text prompt to generate the video from.
        output_path (str): The path where the generated video will be saved.
    """
    try:
        # Tokenize the input text prompt
        inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
        
        # Generate video
        video = model(**inputs).videos
        
        # Save the video
        video[0].save(output_path)
        print(f"Video saved to {output_path}")
    except Exception as e:
        print(f"Error generating video: {e}")
        raise

def main():
    model_name = "your_model_name_here"  # Replace with the actual model name
    text_prompt = "A beautiful sunset over the mountains."  # Example prompt
    output_path = "output_video.mp4"  # Output video file path

    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)

    # Generate the video
    generate_video(model, tokenizer, text_prompt, output_path)

if __name__ == "__main__":
    main()