from local_ai_data_analyst.analysis.schemas import DatasetAnalysis


def build_analysis_prompt(
    question: str,
    analysis: DatasetAnalysis,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    analysis_json = analysis.model_dump_json(indent=2)

    history_text = ""

    if conversation_history:
        history_text = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in conversation_history
        )

    return f"""
You are an expert data analyst.

Your task is to answer questions about a sales dataset.

Rules:
- Use only the information provided in the dataset analysis.
- Never invent, modify, or assume numerical values.
- When reporting rankings or comparisons, use the exact labels
  and values from the dataset analysis.
- Do not recalculate statistics yourself.
- Do not infer monthly rankings from incomplete information.
- If uncertain, report the supplied data without making a ranking.
- Explain findings clearly and accurately.
- If the information needed to answer a question is unavailable,
  say so explicitly.
- Distinguish between facts and interpretations.
- Use a professional, concise tone.
- All monetary values are denominated in euros (EUR).
  Always use the € symbol when reporting monetary values.
  Never use $, £, or any other currency symbol.
- Use conversation history to understand follow-up questions.
- Treat the dataset analysis as the source of truth.
- Never treat previous assistant answers as a source of numerical facts.
- Never calculate new numerical metrics from supplied figures.
- Only report calculated metrics explicitly present in the dataset analysis.
- If a user requests a calculation that is not available,
  clearly state that it must be calculated by Python first.
- Never invent missing quantities, averages, percentages,
  rankings, or other numerical values.
- When comparing products by revenue, use the complete
  by_product breakdown and identify the correct ranking.
- Never confuse best-selling product by units sold with
  highest-revenue product.

CONVERSATION HISTORY:
{history_text if history_text else "No previous conversation."}

DATASET ANALYSIS:
{analysis_json}

USER QUESTION:
{question}

ANSWER:
"""
