import pytest

from local_ai_data_analyst.llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
)


def test_list_models_raises_connection_error_when_ollama_is_unavailable():
    with (
        OllamaClient(
            base_url="http://localhost:1",
            timeout=1.0,
        ) as ollama,
        pytest.raises(OllamaConnectionError),
    ):
        ollama.list_models()


def test_list_models_returns_model_information():
    with OllamaClient() as ollama:
        result = ollama.list_models()

    assert "models" in result
    assert isinstance(result["models"], list)
    assert len(result["models"]) > 0


def test_generate_returns_response():
    with OllamaClient() as ollama:
        result = ollama.generate(
            prompt="Reply with exactly: Hello",
            model="qwen3:8b",
            temperature=0,
            num_predict=50,
        )

    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"].strip()) > 0


def test_generate_raises_connection_error_when_ollama_is_unavailable():
    with (
        OllamaClient(
            base_url="http://localhost:1",
            timeout=1.0,
        ) as ollama,
        pytest.raises(OllamaConnectionError),
    ):
        ollama.generate(
            prompt="Hello",
            model="qwen3:8b",
        )
