import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name):
    """
    Load the model and tokenizer from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise

def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generate text using the provided model and tokenizer.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        prompt (str): The input text to generate from.
        max_length (int): The maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error during text generation: {e}")
        raise

def main():
    model_name = "tencent/Hunyuan-1.8B-Instruct"
    prompt = "Once upon a time in a land far away,"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt)
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()