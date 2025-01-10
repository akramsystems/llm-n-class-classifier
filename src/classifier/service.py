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
    Domain logic to classify an input using our LLM pipeline.
    """
    # Build system prompt, including few-shot examples if provided
    # note this can be cached in the future since this will act as a constant
    # for a given dataset (or we could optimize the system prompt for each dataset this will create faster results)
    user_prompt = llm_classifier.build_user_prompt(dataset_name, custom_labels, few_shot_examples, input_schema_definition)
    
    # Pass the prompt + data to the LLM (dynamic rpomp)
    classification_response = llm_classifier.call_llm_for_classification(user_prompt, input_data)

    # Find and return the labeled class (note there is a better way to do this in the future)
    logger.info(f"Classification response: {classification_response}")
    for prediction in classification_response:
        if prediction.value:
            return prediction.label_name

    logger.warning("No class was predicted to be correct")
    return 'None'  # Return None if no label is found 