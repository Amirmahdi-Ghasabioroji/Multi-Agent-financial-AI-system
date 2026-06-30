"""Local LLM client for a self-hosted Ollama server (no API costs)."""

import json
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


class OllamaClient:
    """Synchronous wrapper around the Ollama REST API (chat + generate)."""

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "mistral",
        timeout: float = 120.0,
        temperature: float = 0.2,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def is_available(self) -> bool:
        """Return True if the Ollama server responds and the model is present."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.url}/api/tags")
                response.raise_for_status()
                models = [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as exc:
            logger.warning("Ollama not reachable at {}: {}", self.url, exc)
            return False

        present = any(m == self.model or m.startswith(f"{self.model}:") for m in models)
        if not present:
            logger.warning(
                "Ollama is up but model '{}' is not pulled. Available: {}",
                self.model,
                models,
            )
        return present

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send chat messages and return the assistant's text content."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, **(options or {})},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama chat request failed: {exc}") from exc

        return data.get("message", {}).get("content", "")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chat in JSON mode and parse the response into a dict."""
        raw = self.chat(messages, json_mode=True, options=options)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            extracted = self._extract_json(raw)
            if extracted is not None:
                return extracted
            raise OllamaError(f"Model did not return valid JSON. Got: {raw[:500]}")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Best-effort extraction of a JSON object embedded in free text."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
