# LLM Classification API

## Overview

This project is a FastAPI application designed to perform classification tasks using a language model (LLM). It supports custom labels, few-shot examples, and testing with multiple datasets.

## Project Structure

- **src/**: Main source directory containing the application code.
  - **main.py**: Contains the FastAPI application and the endpoint definitions.
  - **schemas.py**: Pydantic models for request/response validation.
  - **config.py**: Holds configuration details (e.g., environment variables, LLM API keys).
  - **classifier/**: Contains the logic for classification tasks.
    - **llm.py**: Logic for interacting with the LLM.
    - **service.py**: Domain logic for classification tasks.
  - **utils.py**: Utility functions for loading datasets and other helper functions.
- **tests/**: Contains test cases for the application.
  - **test_classification.py**: Tests for classification logic.
  - **utils.py**: Utility functions for testing, such as loading custom labels.
- **datasets/**: Directory containing dataset files and label descriptions.
- **scripts/**: Contains scripts for tasks like downloading datasets.
  - **download_datasets.py**: Script to download datasets and save them as CSV.
- **Dockerfile**: Docker configuration for building the application image.
- **docker-compose.yaml**: Docker Compose configuration for running the application.
- **requirements.txt**: Python dependencies for the project.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Usage

To classify input data, send a POST request to the `/classify` endpoint with the dataset name, input data, optional few-shot examples, and custom labels.

### Endpoint

- **POST /classify**: Classifies the input data using the LLM.
  - **Parameters**:
    - `dataset_name` (str): Name of the dataset, e.g., 'iris', 'mnist', 'imdb'.
    - `input_data` (str): Raw input to classify. Could be text or a reference to features.
    - `custom_labels` (List[CustomLabel]): List of custom labels of possible output values and their descriptions.
    - `few_shot_examples` (Optional[List[Dict]]): Optional few-shot examples to be appended to the system prompt.
    - `input_schema_definition` (Optional[str]): Optional input schema definition to be appended to the system prompt.

## Testing with Datasets

- **Iris**: Tabular data with custom labels like "setosa", "versicolor", "virginica".
- **IMDB**: Text data with sentiment labels ["positive", "negative"].
- **NSL-KDD**: Cyber-security dataset with labels ["anomaly", "normal"].

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Running the Application

1. **Build the Docker Image**:
   ```bash
   docker-compose build
   ```

2. **Start the Docker Container**:
   ```bash
   docker-compose up
   ```

3. **Access the API**:
   The API will be available at `http://localhost:8000`.

### Executing a Python Function within the Docker Container

To execute a Python function within the Docker container, follow these steps:

1. **Access the Running Container**:
   ```bash
   docker exec -it <container_id> /bin/bash
   ```

   Replace `<container_id>` with the actual container ID, which you can find using `docker ps`.

2. **Run a Python Script**:
   Once inside the container, you can execute any Python script. For example:
   ```bash
   python src/scripts/download_datasets.py
   ```

## Additional Notes

- **Authentication**: Secure your endpoints in production.
- **LLM Integration**: Replace mock functions with real LLM API calls.
- **Scalability**: Consider using background tasks or caching for high request volumes.
- **Documentation**: Provide thorough documentation and describe design decisions, tradeoffs, dataset details, and challenges. 