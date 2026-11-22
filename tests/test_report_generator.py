from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from local_ai_data_analyst.analysis.schemas import AnomalyRecord, AnomalyReport
from local_ai_data_analyst.reporting.report_generator import (
    create_anomaly_report,
    save_anomaly_report,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    quantities = [2, 3, 4, 5, 6, 7, 8, 45]

    return pd.DataFrame(
        [
            {
                "transaction_id": f"T{i:05d}",
                "product_name": "Notebook",
                "quantity": quantity,
                "unit_price": 4.99,
                "total_amount": round(quantity * 4.99, 2),
            }
            for i, quantity in enumerate(quantities, start=1)
        ]
    )


def test_create_anomaly_report(sample_dataframe):
    report = create_anomaly_report(sample_dataframe)

    assert isinstance(report, AnomalyReport)
    assert report.summary.total_anomalies == 1
    assert report.summary.quantity_anomalies == 1
    assert report.summary.price_anomalies == 0


def test_save_anomaly_report(sample_dataframe, tmp_path: Path):
    report = create_anomaly_report(sample_dataframe)

    output_file = tmp_path / "anomaly_report.json"

    saved_path = save_anomaly_report(report, output_file)

    assert saved_path.exists()
    assert saved_path == output_file

    saved_report = AnomalyReport.model_validate_json(
        saved_path.read_text(encoding="utf-8")
    )

    assert saved_report == report


def test_anomaly_record_rejects_both_anomaly_types():
    with pytest.raises(
        ValidationError,
        match="Exactly one of quantity or unit_price",
    ):
        AnomalyRecord(
            transaction_id="T99999",
            product_name="Laptop",
            quantity=5,
            unit_price=899.99,
            total_amount=4499.95,
        )
