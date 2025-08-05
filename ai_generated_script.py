import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_model_and_tokenizer(model_name: str):
    """Load the model and tokenizer from Hugging Face."""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise

def train_dpo(model_name: str, preference_pairs: list, epochs: int = 3):
    """Train the model using DPO on the provided preference pairs."""
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Initialize the DPO Trainer
    trainer = DPOTrainer(model=model, tokenizer=tokenizer)

    # Train the model
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        for pair in preference_pairs:
            try:
                trainer.train(pair['better'], pair['worse'])
            except Exception as e:
                logging.error(f"Error during training on pair {pair}: {e}")

    # Save the trained model
    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")
    logging.info("Training complete and model saved.")

if __name__ == "__main__":
    # Example usage
    preference_pairs = [
        {"better": "This is a great response.", "worse": "This response is not as good."},
        {"better": "I love how you explained that!", "worse": "That explanation was okay."}
    ]
    train_dpo("gpt2", preference_pairs)