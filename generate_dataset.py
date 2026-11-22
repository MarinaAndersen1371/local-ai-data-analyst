import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUTPUT_FILE = Path(__file__).parent / "data" / "sales.csv"

PRODUCTS = [
    ("P001", "Laptop", "Electronics", 899.99),
    ("P002", "Smartphone", "Electronics", 649.99),
    ("P003", "Headphones", "Electronics", 89.99),
    ("P004", "Monitor", "Electronics", 249.99),
    ("P005", "Office Chair", "Furniture", 179.99),
    ("P006", "Desk", "Furniture", 299.99),
    ("P007", "Notebook", "Stationery", 4.99),
    ("P008", "Pen Set", "Stationery", 12.99),
    ("P009", "Backpack", "Accessories", 49.99),
    ("P010", "Water Bottle", "Accessories", 19.99),
]

REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Consumer", "Corporate", "Small Business"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]

START_DATE = date(2025, 1, 1)
DATE_RANGE_DAYS = 365
NUMBER_OF_TRANSACTIONS = 500


def generate_sales_data() -> list[dict]:
    records = []

    for i in range(1, NUMBER_OF_TRANSACTIONS + 1):
        product_id, product_name, category, unit_price = random.choice(PRODUCTS)

        quantity = random.randint(1, 8)
        discount_pct = random.choice([0, 0.05, 0.10, 0.15, 0.20])

        order_date = START_DATE + timedelta(days=random.randint(0, DATE_RANGE_DAYS - 1))

        region = random.choice(REGIONS)
        customer_segment = random.choice(SEGMENTS)

        # Introduce a few missing values for data-quality exercises.
        if i in (25, 180, 350):
            region = ""

        if i in (75, 290):
            customer_segment = ""

        # Introduce unusual transactions for anomaly detection.
        if i == 120:
            quantity = 45

        if i == 320:
            unit_price = 9999.99

        total_amount = round(
            quantity * unit_price * (1 - discount_pct),
            2,
        )

        records.append(
            {
                "transaction_id": f"T{i:05d}",
                "order_date": order_date.isoformat(),
                "customer_id": f"C{random.randint(1, 120):04d}",
                "customer_segment": customer_segment,
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "total_amount": total_amount,
                "region": region,
                "payment_method": random.choice(PAYMENT_METHODS),
            }
        )

    return records


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = generate_sales_data()

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Dataset created: {OUTPUT_FILE}")
    print(f"Number of transactions: {len(records)}")


if __name__ == "__main__":
    main()
