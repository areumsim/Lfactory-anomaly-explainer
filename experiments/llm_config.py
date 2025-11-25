"""
OpenAI API Configuration for LLM-based Anomaly Explanation
"""
import os
from typing import Optional

# OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model configuration
DEFAULT_MODEL = "gpt-4o-mini"  # Cost-effective for development, can upgrade to gpt-4o later
DEFAULT_TEMPERATURE = 0.3  # Lower temperature for more factual explanations
DEFAULT_MAX_TOKENS = 2000

def get_openai_client():
    """
    Create and return OpenAI client
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

def test_api_connection() -> bool:
    """
    Test if OpenAI API is working
    """
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return True
    except Exception as e:
        print(f"API connection test failed: {e}")
        return False
