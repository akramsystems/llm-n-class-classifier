from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class CustomLabel(BaseModel):
    label: str = Field(..., description="The name of the label")
    description: str = Field(..., description="A description of the label")

class LabelPrediction(BaseModel):
    label_name: str = Field(..., description="Name of the label")
    value: bool = Field(..., description="Predicted value for this label (True or False)")

class ClassificationRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset, e.g. 'iris', 'mnist', 'imdb'")
    input_data: str = Field(..., description="Raw input to classify. Could be text or a reference to features.")
    custom_labels: List[CustomLabel] = Field(description="List of custom labels of possible output values and their descriptions.")
    few_shot_examples: Optional[List[Dict]] = Field(None, description="Optional few-shot examples to be appended to the system prompt.")
    input_schema_definition: Optional[str] = Field(None, description="Optional input schema definition to be appended to the system prompt.")

class ClassificationResponse(BaseModel):
    predictions: List[LabelPrediction] = Field(..., description="List of label predictions from the model")