from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AnomalySummary(BaseModel):
    total_anomalies: int = Field(ge=0)
    quantity_anomalies: int = Field(ge=0)
    price_anomalies: int = Field(ge=0)


class AnomalyRecord(BaseModel):
    transaction_id: str
    product_name: str
    quantity: int | None = None
    unit_price: float | None = None
    total_amount: float

    @model_validator(mode="after")
    def validate_anomaly_type(self):
        if (self.quantity is None) == (self.unit_price is None):
            raise ValueError("Exactly one of quantity or unit_price must be provided.")

        return self


class AnomalyReport(BaseModel):
    summary: AnomalySummary
    quantity_anomalies: list[AnomalyRecord]
    price_anomalies: list[AnomalyRecord]


class DatasetOverview(BaseModel):
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[str]
    data_types: dict[str, str]


class DataQuality(BaseModel):
    missing_values: dict[str, int]
    duplicate_rows: int = Field(ge=0)
    duplicate_transactions: int = Field(ge=0)


class BestSellingProduct(BaseModel):
    product_name: str
    units_sold: int = Field(ge=0)


class MonthlyRevenueExtreme(BaseModel):
    month: str
    revenue: float = Field(ge=0)


class MonthlyRevenueExtremes(BaseModel):
    highest: MonthlyRevenueExtreme
    lowest: MonthlyRevenueExtreme


class SalesSummary(BaseModel):
    total_revenue: float
    average_order_value: float
    average_monthly_revenue: float
    best_selling_product: BestSellingProduct


class RevenueBreakdown(BaseModel):
    by_product: dict[str, float] = Field(default_factory=dict)
    by_category: dict[str, float] = Field(default_factory=dict)
    by_month: dict[str, float] = Field(default_factory=dict)
    by_region: dict[str, float] = Field(default_factory=dict)
    by_customer_segment: dict[str, float] = Field(default_factory=dict)
    by_payment_method: dict[str, float] = Field(default_factory=dict)
    average_revenue_per_unit: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_revenue_values(self):
        for breakdown in (
            self.by_product,
            self.by_category,
            self.by_month,
            self.by_region,
            self.by_customer_segment,
            self.by_payment_method,
            self.average_revenue_per_unit,
        ):
            if any(value < 0 for value in breakdown.values()):
                raise ValueError("Revenue values cannot be negative.")

        return self


class QuantityIQRBounds(BaseModel):
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float


class DatasetAnalysis(BaseModel):
    dataset_overview: DatasetOverview
    data_quality: DataQuality
    numerical_summary: dict[str, dict[str, float]]
    sales: SalesSummary
    revenue_breakdown: RevenueBreakdown
    monthly_revenue_extremes: MonthlyRevenueExtremes
    anomalies: dict[str, Any]


# validates the response structure, rejects invalid confidence values, and handles optional findings
class AIAnalysisResponse(BaseModel):
    question: str
    answer: str
    key_findings: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
