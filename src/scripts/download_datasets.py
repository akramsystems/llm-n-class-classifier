import os
import json
from datasets import load_dataset
from src.logger_config import logger
from src.classifier import llm_classifier
# Directory to save the datasets
DATASETS_DIR = "datasets"

def download_dataset_as_csv(name):
    # Load the dataset using the datasets library
    dataset = load_dataset(name)
    
    dataset_name = name.split("/")[-1]
    
    for _, data in dataset.items():
        # Convert the dataset to a pandas DataFrame
        df = data.to_pandas()
        
        # Rename the last column to 'label'
        df.columns = [*df.columns[:-1], 'label']

        label_values = df['label'].unique().tolist()
        
        # Create a unique CSV file for each split (e.g., train, test, validation)
        csv_file = os.path.join(DATASETS_DIR, f"{dataset_name}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Downloaded {dataset_name} dataset to {csv_file}")

        # Get column labels
        column_labels = {label: llm_classifier.generate_description(label) for label in label_values}
        labels_file = os.path.join(DATASETS_DIR, f"{dataset_name}_labels.json")
        with open(labels_file, 'w') as f:
            json.dump(column_labels, f, indent=4)
        
        logger.info(f"Saved column labels to {labels_file}")
        

def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    # List of dataset names to download
    dataset_names = ["scikit-learn/iris", "scikit-learn/imdb", "Mireu-Lab/NSL-KDD"]
    for name in dataset_names:
        download_dataset_as_csv(name)

if __name__ == "__main__":
    main()
