"""
LLM Provider Abstraction Layer

Provides a clean interface for calling different LLM APIs (Gemini, Groq, OpenAI)
with a unified interface. Each provider handles its own client initialization
and API-specific call logic.
"""

from abc import ABC, abstractmethod
import json
from pyexpat.errors import messages
from typing import Literal
import asyncio

from fastapi import requests

ProviderType = Literal["gemini", "groq", "openai", "openrouter"]  # Added "openrouter" for OpenRouter support


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(self, prompt: str, model: str, api_key: str) -> str:
        """Synchronous call to the LLM API."""
        pass

    @staticmethod
    def get_provider(provider: str) -> "LLMProvider":
        """Factory method to get provider instance."""
        providers = {
            "gemini": GeminiProvider(),
            "groq": GroqProvider(),
            "openai": OpenAIProvider(),
            "openrouter": GroqProvider(),
        }
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
        return providers[provider]


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def call(self, prompt: str, model: str, api_key: str) -> str:
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")

        if not api_key:
            raise ValueError("API key not provided")

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text


class GroqProvider(LLMProvider):
    """Groq API provider (OpenAI-compatible)."""

    def call(self, prompt: str, model: str, api_key: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        if not api_key:
            raise ValueError("API key not provided")

        # Groq uses OpenAI-compatible API
        response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "qwen/qwen3.8-27b",
            "messages": [
                {
                "role": "user",
                "content": prompt
                }
            ],
            "reasoning": {"enabled": True}
        })
        )

        response = response.json()
        return response['choices'][0]['message']


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def call(self, prompt: str, model: str, api_key: str) -> str:
        try:
            import openai
        except ImportError:
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


# Async wrapper for backward compatibility
async def llm_call(prompt: str, api_key: str, model: str, provider: str = "gemini") -> str:
    """Async wrapper for LLM calls - runs sync call in thread pool."""
    provider_instance = LLMProvider.get_provider(provider)
    return await asyncio.to_thread(
        provider_instance.call, prompt, model, api_key
    )