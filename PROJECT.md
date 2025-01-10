Project Structure

src/
    ├── main.py
    ├── schemas.py
    ├── config.py
    ├── llm_service.py
    ├── classification_service.py
    ├── examples/
    │    ├── iris_examples.json
    │    ├── mnist_examples.json
    │    └── imdb_examples.json
    └── README.md

    main.py: Contains the FastAPI application and the endpoint definitions.
    schemas.py: Pydantic models for request/response validation.
    config.py: Holds configuration details (e.g., environment variables, LLM API keys).
    llm_service.py: Contains logic to communicate with the LLM (prompt building, calling the model).
    classification_service.py: Contains domain logic for classification tasks.
    examples/: Contains JSON files or Python modules with few-shot examples for each dataset.
    README.md: Basic documentation about the project, design decisions, etc.

Below is a simplified code version that demonstrates how these files might look.
schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class LabelPrediction(BaseModel):
    label_name: str = Field(..., description="Name of the label")
    value: int = Field(..., description="Predicted value for this label (0 or 1)")

class ClassificationRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset, e.g. 'iris', 'mnist', 'imdb'")
    input_data: str = Field(..., description="Raw input to classify. Could be text or a reference to features.")
    few_shot_examples: Optional[List[Dict]] = Field(
        None,
        description="Optional few-shot examples to be appended to the system prompt."
    )
    custom_labels: Optional[List[str]] = Field(
        None,
        description="Optional list of custom labels to classify."
    )

class ClassificationResponse(BaseModel):
    predictions: List[LabelPrediction] = Field(..., description="List of label predictions from the model")

config.py

import os
from dotenv import load_dotenv

load_dotenv()

# Example: storing your LLM API key or endpoint
LLM_API_KEY = os.getenv("LLM_API_KEY", "fake-key-for-demo")
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "https://dummy-llm-endpoint")

# Additional configuration
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

llm_service.py

    Note: This file simulates LLM interactions. In a real scenario, you would integrate with OpenAI, Azure OpenAI, or another LLM provider.

from typing import List, Dict, Optional

def build_system_prompt(dataset_name: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
    """
    Build a base system prompt. Extend it with few-shot examples if provided.
    """
    base_prompt = f"""
You are a classification model. Your task is to return a JSON array of objects.
Each object has:
  - "label_name": the string name of the label
  - "value": 0 or 1
The dataset is {dataset_name}. 
"""
    if few_shot_examples:
        examples_str = "\n".join(
            [f"Example: {ex}" for ex in few_shot_examples]
        )
        base_prompt += f"\nHere are some few-shot examples:\n{examples_str}\n"

    return base_prompt.strip()

def call_llm_for_classification(system_prompt: str, user_input: str, custom_labels: List[str]) -> List[Dict]:
    """
    Simulate an LLM call that outputs a structured JSON with 'label_name' and 'value'.
    In a real scenario, you'd pass 'system_prompt' + 'user_input' to an LLM API endpoint.
    """

    # 1. Combine the system prompt + user input
    prompt = f"{system_prompt}\nUser Input: {user_input}\n"
    prompt += (
        f"Classify into these labels: {custom_labels}\n"
        if custom_labels else "Classify into default labels.\n"
    )

    # 2. Mock: We'll pretend we got a response from the LLM
    #    that returns a random 0 or 1 for each label. 
    #    Replace with real LLM API call (e.g., openai.ChatCompletion.create).
    fake_response = []
    labels_to_use = custom_labels if custom_labels else ["labelA", "labelB"]
    for lbl in labels_to_use:
        fake_response.append({"label_name": lbl, "value": 0})

    return fake_response

classification_service.py

from typing import List, Dict, Optional
from .llm_service import build_system_prompt, call_llm_for_classification

def classify_input(
    dataset_name: str,
    input_data: str,
    few_shot_examples: Optional[List[Dict]],
    custom_labels: Optional[List[str]]
) -> List[Dict]:
    """
    Domain logic to classify an input using our LLM pipeline.
    """
    # Build system prompt, including few-shot examples if provided
    system_prompt = build_system_prompt(dataset_name, few_shot_examples)

    # Pass the prompt + data to the LLM
    response = call_llm_for_classification(system_prompt, input_data, custom_labels or [])

    return response

main.py

from fastapi import FastAPI
from typing import List
from .schemas import ClassificationRequest, ClassificationResponse, LabelPrediction
from .classification_service import classify_input

app = FastAPI(title="LLM Classification API", version="1.0.0")

@app.post("/classify", response_model=ClassificationResponse)
def classify(request: ClassificationRequest):
    """
    Primary endpoint to run classification using an LLM.
    - Accepts a dataset name, input data, optional few-shot examples, and custom labels.
    - Returns a structured JSON with label predictions.
    """
    predictions_dict = classify_input(
        dataset_name=request.dataset_name,
        input_data=request.input_data,
        few_shot_examples=request.few_shot_examples,
        custom_labels=request.custom_labels
    )

    # Convert the dictionary predictions to LabelPrediction Pydantic models
    predictions = [LabelPrediction(**p) for p in predictions_dict]

    return ClassificationResponse(predictions=predictions)

Testing with Three Common Datasets

Here’s how you might use the above endpoint to test on different datasets:

    Iris (tabular data)
        Dataset Name: iris
        Input: a JSON or text representation of sepal length, sepal width, etc.
        Custom Labels: ["setosa", "versicolor", "virginica"]
        Few-Shot Examples: small examples where the input features and label are provided.

    IMDB (text data)
        Dataset Name: imdb
        Input: a movie review text snippet.
        Custom Labels: ["positive", "negative"] (binary) or more nuanced sentiment categories.
        Few-Shot Examples: short reviews with correct label classification.

Example request (using curl or a tool like Postman):

curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_name": "iris",
       "input_data": "5.1, 3.5, 1.4, 0.2",
       "few_shot_examples": [
          {"input": "5.0,3.6,1.4,0.2", "label": "setosa"},
          {"input": "7.0,3.2,4.7,1.4", "label": "versicolor"}
       ],
       "custom_labels": ["setosa", "versicolor", "virginica"]
     }'

Response (example):

{
  "predictions": [
    {
      "label_name": "setosa",
      "value": 1
    },
    {
      "label_name": "versicolor",
      "value": 0
    },
    {
      "label_name": "virginica",
      "value": 0
    }
  ]
}

Extending the System Prompt with Few-Shot

If more examples or instructions are needed, simply add them to the few_shot_examples field. The build_system_prompt function in llm_service.py automatically appends them to the model’s system prompt, improving the few-shot learning behavior.
Additional Notes

    Authentication: In production, secure your endpoints (e.g., using OAuth2 or API keys).
    LLM Integration: Replace the mock call_llm_for_classification function with real calls to your chosen LLM provider (OpenAI, Azure OpenAI, etc.).
    Scalability: Consider using background tasks or caching if you anticipate high request volume.
    Documentation: Provide a thorough README.md or auto-generate docs from docstrings, and describe:
        Design decisions (e.g., why use FastAPI, how you handle prompt engineering).
        Tradeoffs (e.g., correctness vs. cost vs. latency).
        Dataset details (how they’re preprocessed, how you feed them to the LLM).
        Challenges and approaches (e.g., large input data, model hallucinations, etc.).

With this structure in place, you have a modular, extendable FastAPI application for classification tasks using an LLM that supports custom labels, few-shot examples, and testing with multiple datasets.