"""
LLM Client for MemBuilder (Open Source Version)

This module provides a clean OpenAI-compatible interface for LLM operations.
It works with any OpenAI-compatible API endpoint.

Usage:
    # Standard OpenAI
    client = create_llm_client(base_url="https://api.openai.com/v1", api_key="sk-...")
    
    # Local proxy (e.g., vLLM, custom proxy)
    client = create_llm_client(base_url="http://localhost:8766/v1", api_key="any")
    
    # Other compatible services
    client = create_llm_client(base_url="https://your-service.com/v1", api_key="...")

Configuration:
    Set environment variables:
    - OPENAI_API_KEY: API key
    - OPENAI_BASE_URL: Base URL (optional, defaults to OpenAI)
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class LLMClient:
    """
    OpenAI-compatible LLM client.
    
    Works with any OpenAI-compatible API endpoint including:
    - OpenAI API
    - Azure OpenAI
    - vLLM servers
    - Local proxy servers
    - Any other OpenAI-compatible service
    """

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        """Normalize OpenAI message content into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _extract_json_text(content: str) -> str:
        """Extract JSON object text from mixed model output."""
        import re

        clean = (content or "").strip()
        if not clean:
            return clean

        # Remove markdown fences if present.
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', clean).strip()

        # Prefer the first JSON object block if model prepends analysis text.
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            return clean[start:end + 1].strip()
        return clean
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL (defaults to OPENAI_BASE_URL env var or OpenAI)
            model: Default model name
            embedding_model: Default embedding model name
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.embedding_model = embedding_model
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set via api_key parameter or OPENAI_API_KEY env var."
            )
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = OpenAI(**client_kwargs)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        model: str = None,
        **kwargs
    ) -> str:
        """
        Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            model: Model override (uses default if not specified)
            **kwargs: Additional OpenAI API parameters
        
        Returns:
            Response content as string
        """
        model = model or self.model
        
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if response_format:
            request_kwargs["response_format"] = response_format
        
        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception:
            # Some OpenAI-compatible servers (e.g., vLLM) may not support response_format.
            if "response_format" in request_kwargs:
                request_kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**request_kwargs)
            else:
                raise
        raw_content = response.choices[0].message.content
        content = self._normalize_message_content(raw_content)
        
        # Validate JSON if required
        if response_format and response_format.get("type") == "json_object":
            content = self._extract_json_text(content)
            if not content:
                raise ValueError("Model returned empty content for json_object response.")
            json.loads(content)  # Validate JSON
        
        return content
    
    def get_embeddings(
        self,
        texts: List[str],
        model: str = None
    ) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            model: Embedding model override
        
        Returns:
            List of embedding vectors
        """
        model = model or self.embedding_model
        response = self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]


# Alias for backward compatibility
OpenAIClient = LLMClient

# Provider registry: what this client module supports
AVAILABLE_PROVIDERS = ['openai', 'vllm']
DEFAULT_PROVIDER = 'openai'


def create_llm_client(
    api_key: str = None,
    base_url: str = None,
    model: str = "gpt-4o",
    provider: str = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM client.
    
    This public version always creates an OpenAI-compatible LLMClient.
    The ``provider`` parameter is accepted for API compatibility with the
    internal client module but does not change dispatch behavior here —
    use ``base_url`` to point at different backends (vLLM, proxies, etc.).
    
    Args:
        api_key: API key (defaults to OPENAI_API_KEY env var)
        base_url: Base URL (defaults to OPENAI_BASE_URL env var or OpenAI)
        model: Default model name
        provider: Accepted for interface compatibility; not used for dispatch
        **kwargs: Additional arguments passed to LLMClient
    
    Returns:
        LLMClient instance
    
    Examples:
        # Use OpenAI
        client = create_llm_client(api_key="sk-...")
        
        # Use local proxy
        client = create_llm_client(
            base_url="http://localhost:8766/v1",
            api_key="any",
            model="claude-sonnet-4-5-20250929"
        )
        
        # Use vLLM
        client = create_llm_client(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="Qwen/Qwen2-7B"
        )
    """
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        **kwargs
    )
