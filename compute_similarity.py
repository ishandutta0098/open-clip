import torch
import open_clip
from PIL import Image
import torchvision.transforms as transforms
import os

def load_model(model_name='ViT-B-32', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load the OpenCLIP model.

    Args:
        model_name (str): The name of the model to load.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        model: The loaded OpenCLIP model.
        preprocess: The preprocessing function for images.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    model.to(device)
    return model, preprocess

def preprocess_image(image_path, preprocess):
    """
    Preprocess the image for the model.

    Args:
        image_path (str): The path to the image file.
        preprocess: The preprocessing function for images.

    Returns:
        tensor: The preprocessed image tensor.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)

def compute_similarity(model, image_tensor, text, device):
    """
    Compute the similarity between an image and a text.

    Args:
        model: The OpenCLIP model.
        image_tensor: The preprocessed image tensor.
        text (str): The text to compare with the image.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        float: The similarity score.
    """
    with torch.no_grad():
        image_features = model.encode_image(image_tensor.to(device))
        text_features = model.encode_text(open_clip.tokenize([text]).to(device))
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = (image_features @ text_features.T).item()
    return similarity

def main(image_path, text):
    """
    Main function to compute image-text similarity.

    Args:
        image_path (str): The path to the image file.
        text (str): The text to compare with the image.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = load_model(device=device)
    
    image_tensor = preprocess_image(image_path, preprocess)
    similarity = compute_similarity(model, image_tensor, text, device)
    
    print(f"Similarity between image and text: {similarity:.4f}")

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    text = "A description of the image"  # Replace with your text
    main(image_path, text)