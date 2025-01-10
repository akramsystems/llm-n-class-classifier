import json
from typing import List

from src.classifier import classify_input
from src.utils import load_dataset
from src.logger_config import logger
from tests.utils import load_custom_labels_from_json, load_input_schema_definition, write_results_to_file

SAMPLE_SIZE = 100

# MULTI-CLASS CLASSIFICATION
def test_iris_classification_error():
    dataset_name = 'iris'
    
    df = load_dataset(dataset_name).sample(n=SAMPLE_SIZE)
    
    total_examples = len(df)
    incorrect_predictions = 0

    # Load custom labels from JSON
    custom_labels = load_custom_labels_from_json('datasets/iris_labels.json')

    for index, row in df.iterrows():
        input_data = str(row.drop('label').to_dict())
        true_label = row['label']

        try:
            # Simulate classification
            y_hat = classify_input(
                dataset_name=dataset_name,
                input_data=input_data,
                few_shot_examples=None,
                custom_labels=custom_labels
            )
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"429 error encountered at index {index}, skipping this instance.")
                continue
            else:
                raise e

        # Check if the prediction matches the true label
        if y_hat.lower() != true_label.lower():
            incorrect_predictions += 1

        accuracy = (index+1 - incorrect_predictions) / (index+1)
        
        # log the percentage of correct predictions
        logger.info(f"Percentage of correct predictions: {accuracy}")

    # Calculate error
    error = 1 - accuracy
    logger.info(f"Classification Error for {dataset_name}: {error}")

    # Write results to file
    write_results_to_file(dataset_name, accuracy, error)

    # Assert that error is minimized (for example, less than a threshold)
    assert error < 0.1  # Example threshold 

# BINARY CLASSIFICATION
def test_imdb_classification_error():
    dataset_name = 'imdb'
    
    df = load_dataset(dataset_name).sample(n=SAMPLE_SIZE)
    
    total_examples = len(df)
    incorrect_predictions = 0

    # Load custom labels from JSON
    custom_labels = load_custom_labels_from_json('datasets/imdb_labels.json')

    for index, row in df.iterrows():
        input_data = str(row.drop('label').to_dict())
        true_label = row['label']

        try:
            # Simulate classification
            y_hat = classify_input(
                dataset_name=dataset_name,
                input_data=input_data,
                few_shot_examples=None,
                custom_labels=custom_labels
            )
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"429 error encountered at index {index}, skipping this instance.")
                continue
            else:
                raise e

        # Check if the prediction matches the true label
        if y_hat != true_label:
            incorrect_predictions += 1
        
        accuracy = (index + 1 - incorrect_predictions) / (index + 1)

    # Calculate error
    error = incorrect_predictions / total_examples
    logger.info(f"Classification Error for {dataset_name}: {error}")

    # Write results to file
    write_results_to_file(dataset_name, accuracy, error)

    # Assert that error is minimized (for example, less than a threshold
    assert error < 0.1, f"Classification Error for {dataset_name} is too high: {error}"

# BINARY CLASSIFICATION (CYBER-SECURITY related)
def test_nslkdd_classification_error():
    dataset_name = 'NSL-KDD'
    
    df = load_dataset(dataset_name).sample(n=SAMPLE_SIZE)
    
    total_examples = len(df)
    incorrect_predictions = 0

    # Load custom labels from JSON
    custom_labels = load_custom_labels_from_json('datasets/NSL-KDD_labels.json')

    for index, row in df.iterrows():
        input_data = str(row.drop('label').to_dict())
        true_label = row['label']

        input_schema_definition = load_input_schema_definition('datasets/NSL-KDD_schema.json')

        try:
            # Simulate classification
            y_hat = classify_input(
                dataset_name=dataset_name,
                input_data=input_data,
                custom_labels=custom_labels,
                input_schema_definition=input_schema_definition
            )
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"429 error encountered at index {index}, skipping this instance.")
                continue
            else:
                raise e

        # Check if the prediction matches the true label
        if y_hat != true_label:
            incorrect_predictions += 1
        
        # log the percentage of correct predictions
        logger.info(f"Percentage of correct predictions: {(index+1 - incorrect_predictions) / (index+1)}")

    # Calculate error
    error = incorrect_predictions / total_examples
    logger.info(f"Classification Error for {dataset_name}: {error}")

    # Write results to file
    write_results_to_file(dataset_name, 1 - error, error)

    # Assert that error is minimized (for example, less than a threshold)
    assert error < 0.5, f"Classification Error for {dataset_name} is too high: {error}"

if __name__ == "__main__":
    test_iris_classification_error()
    # test_nslkdd_classification_error()
    # test_imdb_classification_error()
