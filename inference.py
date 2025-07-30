import torch
import clip
from PIL import Image
import requests
from io import BytesIO

def load_model(model_name='ViT-B/32'):
    """
    Load the CLIP model.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model: The loaded CLIP model.
        preprocess: The preprocessing function for input images.
    """
    try:
        model, preprocess = clip.load(model_name)
        return model, preprocess
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_image(image_url):
    """
    Preprocess an image from a URL.

    Args:
        image_url (str): The URL of the image to preprocess.

    Returns:
        image: The preprocessed image.
    """
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error fetching or processing image: {e}")
        raise

def perform_inference(model, preprocess, image, text_prompts):
    """
    Perform inference using the CLIP model.

    Args:
        model: The loaded CLIP model.
        preprocess: The preprocessing function for input images.
        image: The preprocessed image.
        text_prompts (list): A list of text prompts for inference.

    Returns:
        scores: The similarity scores between the image and text prompts.
    """
    try:
        # Preprocess the image
        image_input = preprocess(image).unsqueeze(0)

        # Tokenize the text prompts
        text_inputs = clip.tokenize(text_prompts)

        # Perform inference
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            # Calculate similarity scores
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return scores
    except Exception as e:
        print(f"Error during inference: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    model, preprocess = load_model()
    image_url = "https://example.com/sample_image.jpg"  # Replace with a valid image URL
    text_prompts = ["a cat", "a dog", "a car"]

    try:
        image = preprocess_image(image_url)
        scores = perform_inference(model, preprocess, image, text_prompts)
        print("Inference scores:", scores)
    except Exception as e:
        print(f"An error occurred: {e}")