import os
from contextlib import asynccontextmanager
from typing import Literal
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import load_config

CATEGORICAL = [
    "Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1",
]

cfg = load_config()
MODEL_URI = os.getenv(
    "MODEL_URI", f"models:/{cfg['mlflow']['registered_model_name']}@champion"
)
SEGMENTS: dict[int, str] = cfg["segments"]
state = {}


class Customer(BaseModel):
    Gender: Literal["Male", "Female"]
    Ever_Married: Literal["Yes", "No"] | None = None
    Age: int = Field(ge=18, le=100)
    Graduated: Literal["Yes", "No"] | None = None
    Profession: str | None = None
    Work_Experience: float | None = Field(default=None, ge=0)
    Spending_Score: Literal["Low", "Average", "High"]
    Family_Size: float | None = Field(default=None, ge=1)
    Var_1: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class Prediction(BaseModel):
    cluster: int
    segment: str


def to_frame(customer: Customer) -> pd.DataFrame:
    data = customer.model_dump()
    row = {k: np.nan if v is None else v for k, v in data.items()}
    df = pd.DataFrame([row])
    df[CATEGORICAL] = df[CATEGORICAL].astype(object)
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)
    yield
    state.clear()


app = FastAPI(title="Customer Segmentation API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer) -> Prediction:
    cluster = int(state["model"].predict(to_frame(customer))[0])
    return Prediction(cluster=cluster, segment=SEGMENTS[cluster])