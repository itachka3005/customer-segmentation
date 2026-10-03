from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset as described in README."
        )
    return pd.read_csv(path)


def get_features(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    return df.drop(columns=drop_columns)