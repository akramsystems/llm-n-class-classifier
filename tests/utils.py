from typing import List
from src.schemas import CustomLabel
import json
from datetime import datetime

def load_custom_labels_from_json(file_path) -> List[CustomLabel]:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return [CustomLabel(label=label, description=description) for label, description in data.items()]

def load_input_schema_definition(file_path) -> str:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return str(data)

def write_results_to_file(dataset_name, accuracy, error):
    with open('results.txt', 'a') as file:
        file.write(f"----------------------------------\n")
        file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Dataset: {dataset_name}\n")
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Error: {error:.2f}\n\n")
