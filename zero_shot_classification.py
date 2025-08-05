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

def zero_shot_classification(model, image_tensor, class_names, device):
    """
    Perform zero-shot classification on an image.

    Args:
        model: The OpenCLIP model.
        image_tensor: The preprocessed image tensor.
        class_names (list): List of class names for classification.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        dict: A dictionary with class names and their corresponding similarity scores.
    """
    with torch.no_grad():
        image_features = model.encode_image(image_tensor.to(device))
        text_features = model.encode_text(open_clip.tokenize(class_names).to(device))
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
    
    return {class_name: similarity for class_name, similarity in zip(class_names, similarities)}

def main(image_path, class_names):
    """
    Main function for zero-shot classification.

    Args:
        image_path (str): The path to the image file.
        class_names (list): List of class names for classification.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = load_model(device=device)
    
    image_tensor = preprocess_image(image_path, preprocess)
    results = zero_shot_classification(model, image_tensor, class_names, device)
    
    # Sort results by similarity score
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    
    print("Zero-shot classification results:")
    for class_name, score in sorted_results:
        print(f"{class_name}: {score:.4f}")

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    class_names = ["cat", "dog", "car", "tree"]  # Replace with your class names
    main(image_path, class_names)