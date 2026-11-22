# Local AI Data Analyst

A local AI-powered data analysis assistant built with Python, Pandas, Pydantic, and Ollama. It analyzes CSV datasets, identifies anomalies, generates structured reports, and answers natural-language questions using a locally running Large Language Model (LLM).

## Features

- **CSV data ingestion:** Load and process sales data using Pandas.
- **Data profiling:** Calculate revenue, average order value, monthly revenue, product performance, and other business metrics.
- **Data quality analysis:** Identify missing values and duplicate records.
- **Anomaly detection:** Detect unusual quantities and prices using configurable thresholds and IQR-based statistical analysis.
- **Local AI integration:** Ask natural-language questions using Ollama and Qwen.
- **Structured AI responses:** Validate AI-generated answers with Pydantic schemas.
- **Interactive conversations:** Ask follow-up questions while retaining conversation history.
- **JSON reports:** Export complete dataset analysis and anomaly reports.
- **Logging and usage tracking:** Track application activity, response time, and token usage.

## Technologies

- Python 3.12
- Pandas
- Pydantic
- Pydantic Settings
- Ollama
- Qwen 3 (8B)
- HTTPX
- Pytest
- Ruff
- uv

## Architecture

The project follows one important principle:

Python calculates. The LLM interprets.

Python and Pandas perform the actual data analysis and calculate the metrics. The validated analysis is then provided to the LLM, which interprets the results and answers questions in natural language.

This approach reduces unnecessary data transfer to the model and helps prevent the LLM from inventing numerical results.

```text
CSV Dataset
    |
    v
CSV Loader
    |
    v
Data Analysis (Pandas)
    |
    v
Pydantic Validation
    |
    +----------------------+
    |                      |
    v                      v
JSON Reports          Analysis Context
                           |
                           v
                     Prompt Builder
                           |
                           v
                     Ollama / Qwen
                           |
                           v
                  Structured AI Response
                           |
                           v
                    Pydantic Validation
                           |
                           v
                 Interactive AI Assistant
```

## Project Structure

```text
local-ai-data-analyst/
│
├── data/
│   ├── sales.csv
│   ├── analysis_summary.json
│   └── anomaly_report.json
│
├── src/
│   └── local_ai_data_analyst/
│       ├── analysis/
│       ├── llm/
│       ├── prompts/
│       ├── reporting/
│       ├── config.py
│       ├── logging_config.py
│       └── main.py
│
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md

```

Run the application:

```powershell
uv run python -m local_ai_data_analyst.main
```

Run tests:

```powershell
uv run pytest
```

Check code quality:

```powershell
uv run ruff check .
```


