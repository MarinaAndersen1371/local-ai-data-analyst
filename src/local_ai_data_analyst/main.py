import logging
import time

from local_ai_data_analyst.analysis.analysis_service import (
    analyze_dataset,
)
from local_ai_data_analyst.analysis.csv_loader import load_csv
from local_ai_data_analyst.analysis.schemas import AIAnalysisResponse, DatasetAnalysis
from local_ai_data_analyst.config import settings
from local_ai_data_analyst.llm.ollama_client import OllamaClient, OllamaConnectionError
from local_ai_data_analyst.logging_config import setup_logging
from local_ai_data_analyst.prompts.analysis_prompt import build_analysis_prompt
from local_ai_data_analyst.reporting.report_generator import (
    create_anomaly_report,
    save_anomaly_report,
)


def ask_ai(
    ollama: OllamaClient,
    question: str,
    analysis: DatasetAnalysis,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    prompt = build_analysis_prompt(
        question=question,
        analysis=analysis,
        conversation_history=conversation_history,
    )

    response = ollama.generate(
        prompt=prompt,
        model=settings.ollama_model,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        num_predict=settings.num_predict,
        format=AIAnalysisResponse.model_json_schema(),
    )

    validated_response = AIAnalysisResponse.model_validate_json(response["response"])

    response["validated_response"] = validated_response

    return response


def run_interactive_session(
    ollama: OllamaClient,
    analysis: DatasetAnalysis,
) -> None:
    logger = logging.getLogger(__name__)
    conversation_history: list[dict[str, str]] = []

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nAI is thinking...")
        start_time = time.perf_counter()

        try:
            response = ask_ai(ollama, question, analysis, conversation_history)
            validated_response = response["validated_response"]
            answer = validated_response.answer
            key_findings = validated_response.key_findings
            confidence = validated_response.confidence

        except OllamaConnectionError as exc:
            logger.error("AI request failed: %s", exc)
            print("\nSorry, I couldn't connect to the AI model. Please try again.")
            continue

        print(f"\nAI: {answer}")
        if key_findings:
            print("\nKey findings:")
            for finding in key_findings:
                print(f"- {finding}")
        print(f"\nConfidence: {confidence}")

        response_time = time.perf_counter() - start_time
        prompt_tokens = response.get("prompt_eval_count", 0)
        generated_tokens = response.get("eval_count", 0)
        total_tokens = prompt_tokens + generated_tokens

        logger.info("AI response generated in %.2f seconds.", response_time)

        print(f"Response time: {response_time:.2f} seconds")
        print(
            f"Tokens: {prompt_tokens} prompt + "
            f"{generated_tokens} generated = {total_tokens} total"
        )

        logger.info(
            "Token usage: %s prompt, %s generated, %s total.",
            prompt_tokens,
            generated_tokens,
            total_tokens,
        )

        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

        conversation_history = conversation_history[-settings.max_history_messages :]

        logger.debug(
            "Conversation history limited to %s messages.",
            settings.max_history_messages,
        )


def main():
    setup_logging()

    df = load_csv("data/sales.csv")

    logger = logging.getLogger(__name__)
    logger.info("Dataset loaded successfully.")

    analysis = analyze_dataset(df)

    logger.info("Dataset analysis completed successfully.")

    print("\n--- MONTHLY REVENUE EXTREMES ---")
    print(
        f"Highest: {analysis.monthly_revenue_extremes.highest.month} "
        f"- €{analysis.monthly_revenue_extremes.highest.revenue:,.2f}"
    )
    print(
        f"Lowest: {analysis.monthly_revenue_extremes.lowest.month} "
        f"- €{analysis.monthly_revenue_extremes.lowest.revenue:,.2f}"
    )

    analysis_path = "data/analysis_summary.json"
    with open(analysis_path, "w", encoding="utf-8") as file:
        file.write(analysis.model_dump_json(indent=4))
    print(f"\nAnalysis summary saved to: {analysis_path}")

    print("\n--- SALES SUMMARY ---")
    print(f"Total Revenue: €{analysis.sales.total_revenue:,.2f}")
    print(f"Average Order Value: €{analysis.sales.average_order_value:,.2f}")
    print(f"Average Monthly Revenue: €{analysis.sales.average_monthly_revenue:,.2f}")

    print(f"Best-selling Product: {analysis.sales.best_selling_product.product_name}")

    print("\n--- REVENUE BY PRODUCT ---")
    for product, revenue in analysis.revenue_breakdown.by_product.items():
        print(f"{product}: €{revenue:,.2f}")

    print("\n--- AVERAGE REVENUE PER UNIT ---")
    for product, average in analysis.revenue_breakdown.average_revenue_per_unit.items():
        print(f"{product}: €{average:,.2f}")

    report = create_anomaly_report(df)

    output_path = save_anomaly_report(
        report,
        "data/anomaly_report.json",
    )

    print("\n--- ANOMALY REPORT ---")
    print(report)

    print(f"\nReport saved to: {output_path}")

    print("\n--- AI DATA ANALYST ---")
    print("Ask questions about your sales data.")
    print("Type 'exit' or 'quit' to stop.")
    print(f"Conversation history limit: {settings.max_history_messages} messages.")

    with OllamaClient() as ollama:
        run_interactive_session(ollama, analysis)


if __name__ == "__main__":
    main()
