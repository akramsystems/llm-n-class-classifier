import pandas as pd
import os
from typing import Type, Literal, TypeVar
from enum import Enum

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load dataset examples from a CSV file in the datasets directory.
    """
    file_path = os.path.join('datasets', f'{dataset_name}.csv')
    return pd.read_csv(file_path) 

T = TypeVar('T', bound=Enum)

def enum_to_literal(enum_cls: Type[T]) -> Type[Literal[str]]:
    """
    Convert the values of an Enum class to a Literal type.
    
    Args:
        enum_cls: The Enum class whose values need to be converted to a Literal.
    
    Returns:
        A Literal type containing all values of the Enum.
    """
    # Extract values and return a Literal
    values = tuple(item.value for item in enum_cls)
    return Literal[*values]  # This works if statically resolved by tools like mypy


if __name__ == "__main__":

    # Test out random utils

    
    # Example Enum
    class Animal(Enum):
        DOG = "dog"
        CAT = "cat"
        BIRD = "bird"

    # Generate the Literal type dynamically
    AnimalLiteral = Literal["dog", "cat", "bird"]

    # Example usage of the Literal
    def is_valid_animal(animal: AnimalLiteral) -> bool:
        return True

    # Validate correct values
    print(is_valid_animal("dog"))  # True
    print(is_valid_animal("cat"))  # True

    # Invalid case (static type checkers like mypy would catch this)
    try:
        print(is_valid_animal("fish"))  # Would raise an error in strict typing
    except Exception as e:
        print(e)