import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import get_features, load_data
from src.pipeline import build_pipeline


def load_features(cfg: dict) -> pd.DataFrame:
    df = load_data(cfg["data"]["path"])
    return get_features(df, cfg["data"]["drop_columns"])


def params_from_config(cfg: dict) -> dict:
    return {
        "n_clusters": cfg["model"]["n_clusters"],
        "n_components": cfg["model"]["n_components"],
        "rare_threshold": cfg["preprocessing"]["rare_threshold"],
        "random_state": cfg["model"]["random_state"],
    }


def evaluate(pipe: Pipeline, X: pd.DataFrame, labels) -> dict[str, float]:
    X_prep = pipe.named_steps["preprocess"].transform(X)
    return {
        "silhouette": silhouette_score(X_prep, labels),
        "davies_bouldin": davies_bouldin_score(X_prep, labels),
        "calinski_harabasz": calinski_harabasz_score(X_prep, labels),
        "inertia": pipe.named_steps["cluster"].inertia_,
    }


def train_and_log(
    X: pd.DataFrame,
    params: dict,
    run_name: str | None = None,
    log_model: bool = True,
    registered_model_name: str | None = None,
) -> dict[str, float]:
    pipe = build_pipeline(**params)
    with mlflow.start_run(run_name=run_name):
        labels = pipe.fit_predict(X)
        metrics = evaluate(pipe, X, labels)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if log_model:
            mlflow.log_artifact("configs/config.yaml")
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                skops_trusted_types=["numpy.dtype"],
                registered_model_name=registered_model_name,
            )
    return metrics


def main() -> None:
    cfg = load_config()
    X = load_features(cfg)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    train_and_log(
        X,
        params_from_config(cfg),
        registered_model_name=cfg["mlflow"]["registered_model_name"],
    )


if __name__ == "__main__":
    main()