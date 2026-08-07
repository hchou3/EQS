import asyncio
import json
import os
from typing import Literal

# Provider-specific imports
try:
    from google import genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROVIDER_MODELS = {
    "gemini": ["gemini-2.0-flash", "gemini-2.0-pro"],
    "groq": ["groq/llama3", "groq/mixtral-8x7b", "groq/gemma-7b-it"],
    "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
}

ProviderType = Literal["gemini", "groq", "openai"]

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
# Provider-specific LLM calls (sync - run in thread pool)
# -----------------------------------------------------------------------------
def _llm_call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Synchronous Gemini API call."""
    if not genai:
        raise ImportError("google-genai package not installed. Run: pip install google-genai")
    if not api_key:
        raise ValueError("API key not provided")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def _llm_call_groq(prompt: str, api_key: str, model: str) -> str:
    """Synchronous Groq API call (OpenAI-compatible)."""
    if not openai:
        raise ImportError("openai package not installed. Run: pip install openai")
    if not api_key:
        raise ValueError("API key not provided")

    # Groq uses OpenAI-compatible API
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    # Extract model name (remove "groq/" prefix if present)
    model_name = model.replace("groq/", "")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


def _llm_call_openai(prompt: str, api_key: str, model: str) -> str:
    """Synchronous OpenAI API call."""
    if not openai:
        raise ImportError("openai package not installed. Run: pip install openai")
    if not api_key:
        raise ValueError("API key not provided")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


def _llm_call_sync(prompt: str, api_key: str, model: str, provider: str) -> str:
    """Route to the appropriate provider-specific call."""
    provider = provider.lower()

    if provider == "gemini":
        return _llm_call_gemini(prompt, api_key, model)
    elif provider == "groq":
        return _llm_call_groq(prompt, api_key, model)
    elif provider == "openai":
        return _llm_call_openai(prompt, api_key, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: gemini, groq, openai")


async def llm_call(prompt: str, api_key: str, model: str, provider: str = "gemini") -> str:
    """Call the LLM asynchronously and return the response text."""
    return await asyncio.to_thread(_llm_call_sync, prompt, api_key, model, provider)

