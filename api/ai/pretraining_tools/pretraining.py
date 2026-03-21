import asyncio
import json
from google import genai

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODELS = ["gemini-2.0-flash", "gemini-2.0-pro"]


# -----------------------------------------------------------------------------
# Response parsing (sync - no I/O)
# -----------------------------------------------------------------------------
def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response to extract column classifications as a validated dict."""
    try:
        cleaned_response = response_text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        result = json.loads(cleaned_response)
        if "protected_attributes" not in result or "target_columns" not in result:
            raise ValueError("Response missing required fields")
        return result

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse LLM response as JSON: {str(e)}\nResponse: {response_text}")
    except Exception as e:
        raise Exception(f"Error parsing LLM response: {str(e)}")


# -----------------------------------------------------------------------------
# LLM call (async - I/O)
# -----------------------------------------------------------------------------
def _llm_call_sync(prompt: str, api_key: str, model: str) -> str:
    """Synchronous Gemini API call (run in thread pool)."""
    if not api_key:
        raise ValueError("API key not provided")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


async def llm_call(prompt: str, api_key: str, model: str) -> str:
    """Call the LLM asynchronously and return the response text."""
    return await asyncio.to_thread(_llm_call_sync, prompt, api_key, model)

