import numpy as np
import pandas as pd
import pytest

from src.pipeline import build_pipeline, build_preprocessor


@pytest.fixture
def customers() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], n),
        "Ever_Married": rng.choice(["Yes", "No"], n),
        "Age": rng.integers(18, 90, n),
        "Graduated": rng.choice(["Yes", "No"], n),
        "Profession": rng.choice(["Artist", "Doctor", "Engineer"], n),
        "Work_Experience": rng.integers(0, 15, n).astype(float),
        "Spending_Score": rng.choice(["Low", "Average", "High"], n),
        "Family_Size": rng.integers(1, 9, n).astype(float),
        "Var_1": rng.choice(["Cat_1", "Cat_4", "Cat_6"], n),
    })
    missing_cols = ["Ever_Married", "Profession", "Work_Experience", "Family_Size"]
    df.loc[:4, missing_cols] = np.nan
    return df


def test_preprocessor_removes_missing_values(customers):
    result = build_preprocessor().fit_transform(customers)
    assert result.shape[0] == len(customers)
    assert not np.isnan(result).any()


def test_pipeline_assigns_every_customer_to_a_cluster(customers):
    labels = build_pipeline(n_clusters=3).fit_predict(customers)
    assert len(labels) == len(customers)
    assert set(labels) <= {0, 1, 2}


def test_pipeline_handles_unseen_profession(customers):
    pipe = build_pipeline(n_clusters=3).fit(customers)
    new_customer = customers.head(1).copy()
    new_customer["Profession"] = "Astronaut"
    assert len(pipe.predict(new_customer)) == 1


def test_pipeline_with_pca(customers):
    labels = build_pipeline(n_clusters=3, n_components=2).fit_predict(customers)
    assert len(labels) == len(customers)