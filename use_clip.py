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
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        response.raise_for_status()  # Raise an error for bad responses
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"The image file at {image_path} does not exist.")
        image = Image.open(image_path)
    
    return image

def compute_similarity(image: Image.Image, text: str) -> float:
    """
    Compute the similarity score between an image and a text using CLIP model.
    
    Args:
        image (Image.Image): The input image.
        text (str): The input text.
    
    Returns:
        float: The similarity score.
    """
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Preprocess the image and text
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    # Move tensors to the appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Compute cosine similarity
    logits_per_image = outputs.logits_per_image  # This is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    return probs[0][0].item()  # Return the similarity score

def main(image_path: str, text: str):
    """
    Main function to load an image and compute its similarity with the given text.
    
    Args:
        image_path (str): The path to the image file or a URL.
        text (str): The input text to compare with the image.
    """
    try:
        image = load_image(image_path)
        similarity_score = compute_similarity(image, text)
        print(f"Similarity score between the image and the text: {similarity_score:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"  # Replace with your image path or URL
    text = "A description of the image"  # Replace with your text
    main(image_path, text)