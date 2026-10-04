import pytest
from fastapi.testclient import TestClient

from src.api import app

VALID_CUSTOMER = {
    "Gender": "Female",
    "Ever_Married": "No",
    "Age": 27,
    "Graduated": "No",
    "Profession": "Healthcare",
    "Work_Experience": 1,
    "Spending_Score": "Low",
    "Family_Size": 4,
    "Var_1": "Cat_6",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_segment(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    assert response.json() == {"cluster": 3, "segment": "Young healthcare workers"}


def test_predict_accepts_missing_optional_fields(client):
    customer = {
        k: v for k, v in VALID_CUSTOMER.items()
        if k not in ("Profession", "Family_Size")
    }
    response = client.post("/predict", json=customer)
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [("Age", 10), ("Spending_Score", "Huge"), ("Gender", "Unknown")],
)
def test_predict_rejects_invalid_values(client, field, value):
    customer = {**VALID_CUSTOMER, field: value}
    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client):
    customer = {k: v for k, v in VALID_CUSTOMER.items() if k != "Age"}
    response = client.post("/predict", json=customer)
    assert response.status_code == 422