"""
llm_handler.py
--------------
Thin client for HuggingFace's hosted inference for Mistral-7B-Instruct.

We use the OpenAI-compatible chat-completions endpoint exposed by the
HuggingFace Inference Providers router. This is the current, supported
way to call hosted open-source LLMs on HF (the legacy serverless
text-generation endpoint is being phased out for many models).

Endpoint:
    https://router.huggingface.co/hf-inference/models/<model>/v1/chat/completions

Auth:
    Bearer token from the HF_TOKEN environment variable.
    Get one free at: https://huggingface.co/settings/tokens
"""

from __future__ import annotations

import os
from typing import Optional

import requests


class LLMConfigError(RuntimeError):
    """Raised when HF_TOKEN is missing or configuration is invalid."""


class LLMAPIError(RuntimeError):
    """Raised when the inference API returns an error or unparseable response."""


class HuggingFaceLLM:
    """Client for Mistral-7B-Instruct (or any compatible) HF-hosted model."""

    DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    BASE_URL = "https://router.huggingface.co/hf-inference/models"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        timeout: int = 60,
        hf_token: Optional[str] = None,
    ) -> None:
        """
        Args:
            model: HF model id.
            max_new_tokens: generation cap.
            temperature: sampling temperature (0.3 ≈ factual).
            timeout: request timeout in seconds.
            hf_token: explicit token. Falls back to HF_TOKEN / HUGGINGFACEHUB_API_TOKEN.
        """
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        if not self.hf_token:
            raise LLMConfigError(
                "HF_TOKEN not set. Add it to your environment or .env file. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str) -> str:
        """
        Send `prompt` to the model and return the generated text.

        Raises LLMAPIError on any non-recoverable API failure.
        """
        if not prompt or not prompt.strip():
            raise LLMAPIError("Empty prompt passed to LLM.")

        url = f"{self.BASE_URL}/{self.model}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.Timeout as e:
            raise LLMAPIError(f"Request to HuggingFace timed out after {self.timeout}s.") from e
        except requests.RequestException as e:
            raise LLMAPIError(f"Network error calling HuggingFace: {e}") from e

        # Handle HTTP errors with informative messages.
        if resp.status_code == 401:
            raise LLMAPIError("HuggingFace returned 401: invalid or expired HF_TOKEN.")
        if resp.status_code == 402 or resp.status_code == 429:
            raise LLMAPIError(
                "HuggingFace rate limit / quota exceeded. "
                "Wait a few minutes or upgrade your HF plan."
            )
        if resp.status_code == 503:
            # Model warming up.
            raise LLMAPIError(
                "Model is currently loading on HuggingFace (cold start). "
                "Please try again in 20-30 seconds."
            )
        if resp.status_code >= 400:
            raise LLMAPIError(
                f"HuggingFace API error {resp.status_code}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise LLMAPIError(f"Could not decode JSON from HF: {e}") from e

        return self._extract_text(data)

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_text(data: dict) -> str:
        """
        Pull the generated string out of a chat-completions response.

        Falls back to a few alternative shapes in case the provider changes.
        """
        # Standard OpenAI-compatible shape.
        try:
            choices = data.get("choices")
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                # Some providers return content as a list of parts.
                if isinstance(content, list):
                    joined = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                    if joined.strip():
                        return joined.strip()
                # Legacy "text" field
                text = choices[0].get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        except (AttributeError, IndexError, TypeError):
            pass

        # Legacy text-generation shape: [{"generated_text": "..."}]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            gen = data[0].get("generated_text")
            if isinstance(gen, str) and gen.strip():
                return gen.strip()

        raise LLMAPIError(f"Unexpected response shape from HuggingFace: {str(data)[:300]}")
