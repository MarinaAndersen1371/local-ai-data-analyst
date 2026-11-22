from local_ai_data_analyst.analysis.analysis_service import analyze_dataset
from local_ai_data_analyst.prompts.analysis_prompt import build_analysis_prompt


def test_build_analysis_prompt_includes_user_question(sample_dataframe):
    analysis = analyze_dataset(sample_dataframe)

    prompt = build_analysis_prompt(
        question="What is the total revenue?",
        analysis=analysis,
    )

    assert "What is the total revenue?" in prompt


def test_build_analysis_prompt_includes_analysis_data(sample_dataframe):
    analysis = analyze_dataset(sample_dataframe)

    prompt = build_analysis_prompt(
        question="Summarize the sales.",
        analysis=analysis,
    )

    assert "550.0" in prompt
    assert "Laptop" in prompt


def test_build_analysis_prompt_includes_conversation_history(sample_dataframe):
    analysis = analyze_dataset(sample_dataframe)

    history = [
        {"role": "user", "content": "Which product sold the most?"},
        {"role": "assistant", "content": "Laptop sold 5 units."},
    ]

    prompt = build_analysis_prompt(
        question="What about its revenue?",
        analysis=analysis,
        conversation_history=history,
    )

    assert "USER: Which product sold the most?" in prompt
    assert "ASSISTANT: Laptop sold 5 units." in prompt
