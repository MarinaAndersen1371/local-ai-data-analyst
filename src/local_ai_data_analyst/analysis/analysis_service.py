import pandas as pd

from local_ai_data_analyst.analysis.data_profiler import (
    generate_anomaly_report,
    get_average_monthly_revenue,
    get_average_order_value,
    get_average_revenue_per_unit_by_product,
    get_best_selling_product,
    get_dataset_overview,
    get_duplicate_count,
    get_duplicate_transactions,
    get_iqr_bounds,
    get_missing_values,
    get_monthly_revenue,
    get_monthly_revenue_extremes,
    get_numerical_summary,
    get_revenue_by_category,
    get_revenue_by_payment_method,
    get_revenue_by_product,
    get_revenue_by_region,
    get_revenue_by_segment,
    get_total_revenue,
)
from local_ai_data_analyst.analysis.schemas import DatasetAnalysis


def analyze_dataset(df: pd.DataFrame) -> DatasetAnalysis:
    """
    Run the complete dataset analysis and return validated results.
    """
    raw_analysis = {
        "dataset_overview": get_dataset_overview(df),
        "data_quality": {
            "missing_values": get_missing_values(df),
            "duplicate_rows": get_duplicate_count(df),
            "duplicate_transactions": get_duplicate_transactions(df),
        },
        "numerical_summary": get_numerical_summary(df),
        "sales": {
            "total_revenue": get_total_revenue(df),
            "average_order_value": get_average_order_value(df),
            "average_monthly_revenue": get_average_monthly_revenue(df),
            "best_selling_product": get_best_selling_product(df),
        },
        "revenue_breakdown": {
            "by_product": get_revenue_by_product(df),
            "by_category": get_revenue_by_category(df),
            "by_month": get_monthly_revenue(df),
            "by_region": get_revenue_by_region(df),
            "by_customer_segment": get_revenue_by_segment(df),
            "by_payment_method": get_revenue_by_payment_method(df),
            "average_revenue_per_unit": get_average_revenue_per_unit_by_product(df),
        },
        "monthly_revenue_extremes": get_monthly_revenue_extremes(df),
        "anomalies": {
            "quantity_iqr_bounds": get_iqr_bounds(df, "quantity"),
            "anomaly_report": generate_anomaly_report(df),
        },
    }

    return validate_analysis(raw_analysis)


def validate_analysis(raw_analysis: dict) -> DatasetAnalysis:
    """
    Validate the complete dataset analysis using Pydantic.
    """
    return DatasetAnalysis.model_validate(raw_analysis)
