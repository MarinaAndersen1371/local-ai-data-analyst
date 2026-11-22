import logging
from unittest.mock import Mock

import local_ai_data_analyst.main as main_module
from local_ai_data_analyst.analysis.analysis_service import analyze_dataset
from local_ai_data_analyst.analysis.schemas import AIAnalysisResponse
from local_ai_data_analyst.llm.ollama_client import OllamaConnectionError


def test_interactive_session_continues_after_connection_error(
    sample_dataframe,
    monkeypatch,
    capsys,
):
    user_inputs = iter(
        [
            "First question",
            "Second question",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(user_inputs),
    )

    mock_ask_ai = Mock(
        side_effect=[
            OllamaConnectionError("Ollama unavailable"),
            {
                "response": "Recovered answer",
                "prompt_eval_count": 10,
                "eval_count": 5,
                "validated_response": AIAnalysisResponse(
                    question="Second question",
                    answer="Recovered answer",
                    confidence="high",
                ),
            },
        ]
    )

    monkeypatch.setattr(main_module, "ask_ai", mock_ask_ai)

    analysis = analyze_dataset(sample_dataframe)

    main_module.run_interactive_session(
        ollama=Mock(),
        analysis=analysis,
    )

    captured = capsys.readouterr()

    assert "couldn't connect to the AI model" in captured.out
    assert "Recovered answer" in captured.out
    assert mock_ask_ai.call_count == 2


def test_interactive_session_logs_response_time(
    sample_dataframe,
    monkeypatch,
    capsys,
    caplog,
):
    user_inputs = iter(["Test question", "exit"])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(user_inputs),
    )

    monkeypatch.setattr(
        main_module,
        "ask_ai",
        Mock(
            return_value={
                "response": "Test answer",
                "prompt_eval_count": 20,
                "eval_count": 8,
                "validated_response": AIAnalysisResponse(
                    question="Test question",
                    answer="Test answer",
                    confidence="high",
                ),
            }
        ),
    )

    # Simulate a request taking exactly 12.5 seconds.
    monkeypatch.setattr(
        main_module,
        "time",
        Mock(perf_counter=Mock(side_effect=[100.0, 112.5])),
    )

    analysis = analyze_dataset(sample_dataframe)

    with caplog.at_level(logging.INFO):
        main_module.run_interactive_session(
            ollama=Mock(),
            analysis=analysis,
        )

    captured = capsys.readouterr()

    assert "Response time: 12.50 seconds" in captured.out
    assert "AI response generated in 12.50 seconds." in caplog.text
    assert "Tokens: 20 prompt + 8 generated = 28 total" in captured.out
    assert "Token usage: 20 prompt, 8 generated, 28 total." in caplog.text
