import pytest
from pydantic import ValidationError

from local_ai_data_analyst.analysis.schemas import (
    AIAnalysisResponse,
    BestSellingProduct,
    RevenueBreakdown,
)


def test_revenue_breakdown_accepts_valid_values():
    breakdown = RevenueBreakdown(
        by_product={"Laptop": 1500.0},
        by_category={"Electronics": 1500.0},
        by_month={"2025-01": 1500.0},
        by_region={"West": 1500.0},
        by_customer_segment={"Consumer": 1500.0},
        by_payment_method={"Credit Card": 1500.0},
    )

    assert breakdown.by_product["Laptop"] == 1500.0


def test_revenue_breakdown_rejects_negative_values():
    with pytest.raises(
        ValidationError,
        match="Revenue values cannot be negative",
    ):
        RevenueBreakdown(
            by_product={"Laptop": -100.0},
            by_category={},
            by_month={},
            by_region={},
            by_customer_segment={},
            by_payment_method={},
        )


def test_best_selling_product_validation():
    product = BestSellingProduct(
        product_name="Notebook",
        units_sold=271,
    )

    assert product.product_name == "Notebook"
    assert product.units_sold == 271


def test_best_selling_product_rejects_negative_units():
    with pytest.raises(ValidationError):
        BestSellingProduct(
            product_name="Notebook",
            units_sold=-1,
        )


def test_ai_analysis_response_valid_data():
    response = AIAnalysisResponse(
        question="Which product generated the most revenue?",
        answer="Laptop generated the most revenue.",
        key_findings=["Laptop generated €147,508.32."],
        confidence="high",
    )

    assert response.question == "Which product generated the most revenue?"
    assert response.confidence == "high"
    assert len(response.key_findings) == 1


def test_ai_analysis_response_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        AIAnalysisResponse(
            question="Test question",
            answer="Test answer",
            key_findings=[],
            confidence="very high",
        )


def test_ai_analysis_response_default_key_findings():
    response = AIAnalysisResponse(
        question="Test question",
        answer="Test answer",
        confidence="medium",
    )

    assert response.key_findings == []
