from typing import List, Dict, Optional, Type, Literal
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI
from src.config import OPENAI_API_KEY
from src.logger_config import logger  # Import the logger
from src.schemas import CustomLabel, ClassificationLLMResponse, LabelPrediction

DEFAULT_MODEL = "gpt-4o"

client = OpenAI(api_key=OPENAI_API_KEY)

class ClassificationLLM:

    SYSTEM_PROMPT = """
    # INSTRUCTIONS
    You are a classification model. Your task is to return a JSON array of objects.
    Each object has:
    - "label_name": the string name of the label
    - "value": True or False

    # CONSTRAINTS
    There can only be ONE True Value at MOST
    You will be given an input to classify.
    You will be given a set of custom labels to be used as the set of possible values for "label_name".
    You MAY be given a set of few-shot examples to help in your classification decision
    
    # GOAL
    Goal is to classify the input into one of the possible labels.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def build_user_prompt(self, dataset_name: str, custom_labels: List[CustomLabel] = [], few_shot_examples: Optional[List[Dict]] = None, input_schema_definition: Optional[str] = None) -> str:
        """
        Build a user prompt. Extend it with few-shot examples if provided.
        """
        logger.debug("Building user prompt for dataset: %s", dataset_name)

        user_prompt = f"Dataset Properties:\n"

        if custom_labels:
            logger.info("Adding custom labels and descriptions to the user prompt")
            labels_with_descriptions = "\n".join(
                [f"Label: {label.label}, Description: {label.description}" for label in custom_labels]
            )
            user_prompt += f"The possible labels are:\n{labels_with_descriptions}\n"

        if few_shot_examples:
            logger.info("Adding few-shot examples to the user prompt")
            examples_str = "\n".join(
                [f"Example: {ex}" for ex in few_shot_examples]
            )
            user_prompt += f"Here are some few-shot examples:\n{examples_str}\n"

        if input_schema_definition:
            logger.info("Adding input schema definition to the user prompt")
            user_prompt += f"Here is the input schema definition:\n{input_schema_definition}\n"

        return user_prompt.strip()

    def call_llm_for_classification(self, user_prompt: str, input_data: str) -> ClassificationLLMResponse:
        """
        Calls the LLM for classification and returns structured output.
        """
        # Use the OpenAI client to get a response
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt + "\n" + input_data},
            ],
            response_format=ClassificationLLMResponse,
        )

        # Extract and return the parsed response for the first choice
        classification_response = completion.choices[0].message.parsed
        logger.info("Classification response: %s", classification_response)
        return classification_response.predictions

    def generate_description(self, column_name: str) -> str:
        """
        Generate a description for a column.
        """

        class ColumnDescription(BaseModel):
            description: str

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Generate a description for the column {column_name} in under 30 words."},
            ],
            response_format=ColumnDescription,
        )
        return response.choices[0].message.parsed.description

llm_classifier = ClassificationLLM(client)


if __name__ == "__main__":
    pass