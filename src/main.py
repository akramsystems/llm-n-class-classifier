from fastapi import FastAPI
from typing import List
from .schemas import ClassificationRequest, ClassificationResponse, LabelPrediction
from .classifier import classify_input
from src.logger_config import logger

app = FastAPI(title="General Classification API", version="1.0.0")

@app.post("/classify", response_model=ClassificationResponse)
def classify(request: ClassificationRequest):
    """
    Primary endpoint to run classification using an LLM.
    - Accepts a dataset name, input data, optional few-shot examples, and custom labels.
    - Returns a structured JSON with label predictions.
    """
    prediction = classify_input(
        dataset_name=request.dataset_name,
        input_data=request.input_data,
        custom_labels=request.custom_labels,
        few_shot_examples=request.few_shot_examples,
        input_schema_definition=request.input_schema_definition
    )

    logger.info(f"Predicted label: {prediction}")

    return ClassificationResponse(model_response=prediction) 
