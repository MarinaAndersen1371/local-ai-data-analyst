from pathlib import Path

from local_ai_data_analyst.analysis.data_profiler import generate_anomaly_report
from local_ai_data_analyst.analysis.schemas import AnomalyReport


def create_anomaly_report(df) -> AnomalyReport:
    """
    Generate and validate an anomaly report from a DataFrame.
    """
    raw_report = generate_anomaly_report(df)

    validated_report = AnomalyReport.model_validate(raw_report)

    return validated_report


def save_anomaly_report(
    report: AnomalyReport,
    output_file: str | Path,
) -> Path:
    """
    Serialize and save a validated anomaly report as JSON.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_json = report.model_dump_json(
        indent=4,
    )

    output_path.write_text(
        report_json,
        encoding="utf-8",
    )

    return output_path
