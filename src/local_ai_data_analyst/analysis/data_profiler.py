import pandas as pd


def get_missing_values(df: pd.DataFrame) -> dict:
    missing_counts = df.isna().sum()

    return {column: int(count) for column, count in missing_counts.items() if count > 0}


def get_dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


def get_duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def get_numerical_summary(df: pd.DataFrame) -> dict:
    numerical_columns = df.select_dtypes(
        include="number"
    )  # select columns based on their data types

    numerical_summary = numerical_columns.describe()

    return numerical_summary.to_dict()


def get_duplicate_transactions(df: pd.DataFrame) -> int:
    return int(df["transaction_id"].duplicated().sum())


def get_total_revenue(df: pd.DataFrame) -> float:
    return round(float(df["total_amount"].sum()), 2)


# AOV: Average Order Value
def get_average_order_value(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    return round(float(df["total_amount"].mean()), 2)


def get_best_selling_product(df: pd.DataFrame) -> dict:
    product_sales = (
        df.groupby("product_name")["quantity"].sum().sort_values(ascending=False)
    )

    best_product = product_sales.index[0]

    return {
        "product_name": best_product,
        "units_sold": int(product_sales.iloc[0]),
    }


def get_revenue_by_product(df: pd.DataFrame) -> dict:
    revenue = (
        df.groupby("product_name")["total_amount"].sum().sort_values(ascending=False)
    )

    return {product: round(float(amount), 2) for product, amount in revenue.items()}


def get_revenue_by_category(df: pd.DataFrame) -> dict:
    revenue = df.groupby("category")["total_amount"].sum().sort_values(ascending=False)

    return {category: round(float(amount), 2) for category, amount in revenue.items()}


def get_monthly_revenue(df: pd.DataFrame) -> dict:
    monthly_revenue = (
        df.groupby(df["order_date"].dt.to_period("M"))["total_amount"]
        .sum()
        .sort_index()
    )

    return {
        str(month): round(float(revenue), 2)
        for month, revenue in monthly_revenue.items()
    }


def get_monthly_revenue_extremes(df: pd.DataFrame) -> dict:
    monthly_revenue = df.groupby(df["order_date"].dt.to_period("M"))[
        "total_amount"
    ].sum()

    highest_month = monthly_revenue.idxmax()
    lowest_month = monthly_revenue.idxmin()

    return {
        "highest": {
            "month": str(highest_month),
            "revenue": round(float(monthly_revenue.loc[highest_month]), 2),
        },
        "lowest": {
            "month": str(lowest_month),
            "revenue": round(float(monthly_revenue.loc[lowest_month]), 2),
        },
    }


def get_average_monthly_revenue(df: pd.DataFrame) -> float:
    monthly_revenue = df.groupby(df["order_date"].dt.to_period("M"))[
        "total_amount"
    ].sum()

    if monthly_revenue.empty:
        return 0.0

    return round(float(monthly_revenue.mean()), 2)


def get_revenue_by_region(df: pd.DataFrame) -> dict:
    revenue = df.groupby("region")["total_amount"].sum().sort_values(ascending=False)

    return {region: round(float(amount), 2) for region, amount in revenue.items()}


def get_revenue_by_segment(df: pd.DataFrame) -> dict:
    revenue = (
        df.groupby("customer_segment")["total_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    return {segment: round(float(amount), 2) for segment, amount in revenue.items()}


def get_revenue_by_payment_method(df: pd.DataFrame) -> dict:
    revenue = (
        df.groupby("payment_method")["total_amount"].sum().sort_values(ascending=False)
    )

    return {
        payment_method: round(float(amount), 2)
        for payment_method, amount in revenue.items()
    }


# statistical anomaly detection methods - the Interquartile Range (IQR).
def get_unusual_quantities(
    df: pd.DataFrame,
    threshold: int = 20,
) -> list[dict]:
    unusual = df[df["quantity"] > threshold]

    return unusual[
        ["transaction_id", "product_name", "quantity", "total_amount"]
    ].to_dict(orient="records")


def get_unusual_prices(
    df: pd.DataFrame,
    threshold: float = 1000.0,
) -> list[dict]:
    unusual = df[df["unit_price"] > threshold]

    return unusual[
        ["transaction_id", "product_name", "unit_price", "total_amount"]
    ].to_dict(orient="records")


def get_iqr_bounds(
    df: pd.DataFrame,
    column: str,
) -> dict:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def detect_iqr_anomalies(
    df: pd.DataFrame,
    column: str,
    group_by: str | None = None,
) -> pd.DataFrame:

    if group_by is None:
        bounds = get_iqr_bounds(df, column)

        return df[
            (df[column] < bounds["lower_bound"]) | (df[column] > bounds["upper_bound"])
        ]

    anomalies = []

    for _, group in df.groupby(group_by):
        bounds = get_iqr_bounds(group, column)

        detected = group[
            (group[column] < bounds["lower_bound"])
            | (group[column] > bounds["upper_bound"])
        ]

        if not detected.empty:
            anomalies.append(detected)

    if not anomalies:
        return df.iloc[0:0].copy()

    return pd.concat(anomalies).sort_index()


def generate_anomaly_report(df: pd.DataFrame) -> dict:
    quantity_anomalies = detect_iqr_anomalies(
        df,
        "quantity",
        group_by="product_name",
    )

    price_anomalies = detect_iqr_anomalies(
        df,
        "unit_price",
        group_by="product_name",
    )

    return {
        "summary": {
            "total_anomalies": (len(quantity_anomalies) + len(price_anomalies)),
            "quantity_anomalies": len(quantity_anomalies),
            "price_anomalies": len(price_anomalies),
        },
        "quantity_anomalies": quantity_anomalies[
            [
                "transaction_id",
                "product_name",
                "quantity",
                "total_amount",
            ]
        ].to_dict(orient="records"),
        "price_anomalies": price_anomalies[
            [
                "transaction_id",
                "product_name",
                "unit_price",
                "total_amount",
            ]
        ].to_dict(orient="records"),
    }


def get_average_revenue_per_unit_by_product(
    df: pd.DataFrame,
) -> dict[str, float]:
    product_revenue = df.groupby("product_name")["total_amount"].sum()
    product_units = df.groupby("product_name")["quantity"].sum()

    average_revenue = product_revenue / product_units

    return {
        product: round(float(revenue), 2)
        for product, revenue in average_revenue.items()
    }
