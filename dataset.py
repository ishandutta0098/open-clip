import pandas as pd
from datasets import Dataset

def load_instruction_dataset(file_path: str):
    """Load instruction dataset from a CSV file."""
    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        # Assuming the CSV has 'input' and 'output' columns
        dataset = Dataset.from_pandas(df[['input', 'output']])
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise