import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import bitsandbytes as bnb

def load_model(model_name: str):
    """
    Load the model and tokenizer with 4-bit quantization.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb.QuantizationConfig(4),
            device_map="auto"  # Automatically map model to available devices
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def chat_with_llama(model, tokenizer):
    """
    Interact with the Llama-3 model for chat completion.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    print("Chat with Llama-3! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat.")
            break
        
        # Tokenize the input
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        # Generate a response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Llama-3: {response}")

def main():
    model_name = "meta-llama/Llama-3"  # Replace with the actual model name if different
    model, tokenizer = load_model(model_name)
    chat_with_llama(model, tokenizer)

if __name__ == "__main__":
    main()