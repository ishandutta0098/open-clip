import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
from dataset import PreferenceDataset

def main():
    # Load model and tokenizer
    model_name = "gpt2"  # Replace with your chat model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load preference pairs dataset
    dataset = PreferenceDataset("path/to/preference_pairs.json")  # Update with your dataset path

    # Initialize DPO Trainer
    trainer = DPOTrainer(model=model, tokenizer=tokenizer, dataset=dataset)

    # Training parameters
    num_epochs = 3
    batch_size = 8
    learning_rate = 5e-5

    # Start training
    for epoch in range(num_epochs):
        trainer.train(batch_size=batch_size, learning_rate=learning_rate)
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")

if __name__ == "__main__":
    main()