from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A Pandas DataFrame containing the CSV data.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, received: {path.suffix}")

    df = pd.read_csv(path)

    df["order_date"] = pd.to_datetime(df["order_date"])

    return df
