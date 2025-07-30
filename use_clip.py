import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import os

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path or URL.

    Args:
        image_path (str): The path to the image file or a URL.

    Returns:
        Image.Image: The loaded image.
    """
    if os.path.isfile(image_path):
        return Image.open(image_path)
    elif image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        response.raise_for_status()  # Raise an error for bad responses
        return Image.open(BytesIO(response.content))
    else:
        raise ValueError("Invalid image path or URL provided.")

def compute_similarity(image: Image.Image, text: str) -> float:
    """
    Compute the similarity score between an image and a text input using CLIP.

    Args:
        image (Image.Image): The image to compare.
        text (str): The text to compare against the image.

    Returns:
        float: The similarity score between the image and the text.
    """
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Preprocess the image and text
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    # Move inputs to the same device as the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    return probs[0][0].item()  # Return the similarity score

if __name__ == "__main__":
    # Example usage
    image_path = input("Enter the image path or URL: ")
    text_input = input("Enter the text to compare with the image: ")

    try:
        image = load_image(image_path)
        similarity_score = compute_similarity(image, text_input)
        print(f"Similarity score between the image and text: {similarity_score:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")