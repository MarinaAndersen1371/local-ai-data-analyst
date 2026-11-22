import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "transaction_id": "T001",
                "order_date": "2025-01-10",
                "product_name": "Laptop",
                "category": "Electronics",
                "quantity": 2,
                "unit_price": 100.0,
                "discount_pct": 0.0,
                "total_amount": 200.0,
                "region": "North",
                "customer_segment": "Consumer",
                "payment_method": "Credit Card",
            },
            {
                "transaction_id": "T002",
                "order_date": "2025-01-15",
                "product_name": "Notebook",
                "category": "Stationery",
                "quantity": 5,
                "unit_price": 10.0,
                "discount_pct": 0.0,
                "total_amount": 50.0,
                "region": "South",
                "customer_segment": "Corporate",
                "payment_method": "PayPal",
            },
            {
                "transaction_id": "T003",
                "order_date": "2025-02-10",
                "product_name": "Laptop",
                "category": "Electronics",
                "quantity": 3,
                "unit_price": 100.0,
                "discount_pct": 0.0,
                "total_amount": 300.0,
                "region": "North",
                "customer_segment": "Consumer",
                "payment_method": "Credit Card",
            },
        ]
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df
