import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import os

def load_image(image_url):
    """
    Load an image from a URL.
    
    Args:
        image_url (str): URL of the image to load.
    
    Returns:
        PIL.Image: Loaded image.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses
        image = Image.open(requests.get(image_url, stream=True).raw)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def compute_similarity(image, text):
    """
    Compute the similarity score between an image and a text using CLIP model.
    
    Args:
        image (PIL.Image): The image to compare.
        text (str): The text to compare.
    
    Returns:
        float: Similarity score between the image and text.
    """
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Process the image and text
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    # Move inputs to the same device as the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    return probs[0][0].item()  # Return the similarity score

def main():
    """
    Main function to run the CLIP model for a given image and text.
    """
    # Example image URL and text
    image_url = "https://example.com/image.jpg"  # Replace with a valid image URL
    text = "A description of the image"  # Replace with your text

    # Load the image
    image = load_image(image_url)
    if image is None:
        print("Failed to load image. Exiting.")
        return

    # Compute similarity
    similarity_score = compute_similarity(image, text)
    print(f"Similarity score between the image and text: {similarity_score:.4f}")

if __name__ == "__main__":
    main()