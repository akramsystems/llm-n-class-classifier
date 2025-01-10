from typing import List, Dict, Optional
from .llm import llm_classifier
from ..logger_config import logger
from ..schemas import CustomLabel

def classify_input(
    dataset_name: str,
    input_data: str,
    custom_labels: List[CustomLabel],
    few_shot_examples: Optional[List[Dict]] = None,
    input_schema_definition: Optional[str] = None
) -> str:
    """
    Classify an input using the LLM pipeline.

    Args:
        dataset_name (str): Name of the dataset.
        input_data (str): Raw input data to classify.
        custom_labels (List[CustomLabel]): List of custom labels.
        few_shot_examples (Optional[List[Dict]]): Few-shot examples for the model.
        input_schema_definition (Optional[str]): Schema definition for the input.

    Returns:
        str: Predicted label name.
    """
    # Build the user prompt
    user_prompt = llm_classifier.build_user_prompt(
        dataset_name, custom_labels, few_shot_examples, input_schema_definition
    )
    
    # Get classification response from the LLM
    classification_response = llm_classifier.call_llm_for_classification(user_prompt, input_data)

    # Log the classification response
    logger.info(f"Classification response: {classification_response}")

    # Find and return the predicted label
    for prediction in classification_response:
        if prediction.value:
            return prediction.label_name

    logger.warning("No class was predicted to be correct")
    return 'None'  # Return 'None' if no label is found 