import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import os

def load_model():
    """
    Load the CLIP model and processor from the transformers library.
    
    Returns:
        model: The loaded CLIP model.
        processor: The loaded CLIP processor.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor

def load_image(image_path):
    """
    Load an image from a file path or URL.
    
    Args:
        image_path (str): The path to the image file or a URL.
    
    Returns:
        image: The loaded image.
    """
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image file at {image_path} does not exist.")
        image = Image.open(image_path)
    return image

def compute_similarity(model, processor, image, text):
    """
    Compute the similarity between an image and a text description using the CLIP model.
    
    Args:
        model: The loaded CLIP model.
        processor: The loaded CLIP processor.
        image: The image to compare.
        text: The text description to compare.
    
    Returns:
        similarity: The cosine similarity score between the image and text.
    """
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # convert to probabilities
    return probs[0][0].item()  # return the similarity score

def main(image_path, text):
    """
    Main function to run the CLIP model inference.
    
    Args:
        image_path (str): The path to the image file or a URL.
        text (str): The text description to compare.
    """
    model, processor = load_model()
    image = load_image(image_path)
    similarity = compute_similarity(model, processor, image, text)
    print(f"Similarity score between the image and text: {similarity:.4f}")

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"  # Replace with your image path or URL
    text = "A description of the image"  # Replace with your text description
    main(image_path, text)