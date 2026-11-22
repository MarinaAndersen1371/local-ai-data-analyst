import logging
from typing import Self

import httpx

from local_ai_data_analyst.config import settings

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when communication with the Ollama API fails."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        timeout: float = settings.ollama_timeout,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
        )

    def list_models(self) -> dict:
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as exc:
            logger.exception("Failed to retrieve models from Ollama.")

            raise OllamaConnectionError(
                f"Failed to retrieve models from Ollama: {exc}",
            ) from exc

    def generate(
        self,
        prompt: str,
        model: str = "qwen3:8b",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 256,
        format: dict | str | None = None,
    ) -> dict:
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": num_predict,
                },
            }

            if format is not None:
                payload["format"] = format

            response = self.client.post(
                "/api/generate",
                json=payload,
            )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as exc:
            logger.exception("Failed to generate a response with Ollama.")
            raise OllamaConnectionError(
                f"Failed to generate a response with Ollama: {exc}"
            ) from exc

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
