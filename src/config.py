import os
from dotenv import load_dotenv

load_dotenv()

# Example: storing your LLM API key or endpoint
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Additional configuration
DEBUG = os.getenv("DEBUG", "true").lower() == "true" 