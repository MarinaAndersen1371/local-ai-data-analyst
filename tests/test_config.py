from local_ai_data_analyst.config import Settings


def test_settings_default_values():
    config = Settings(_env_file=None)

    assert config.ollama_base_url == "http://localhost:11434"
    assert config.ollama_model == "qwen3:8b"
    assert config.ollama_timeout == 600.0
    assert config.temperature == 0.2
    assert config.top_p == 0.9
    assert config.top_k == 40
    assert config.num_predict == 1024
    assert config.max_history_messages == 10


def test_settings_environment_variable_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("TEMPERATURE", "0.5")

    config = Settings(_env_file=None)

    assert config.ollama_model == "qwen2.5:3b"
    assert config.temperature == 0.5
