import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.config import load_config
from src.data import get_features, load_data
from src.pipeline import build_pipeline


def main() -> None:
    cfg = load_config()
    df = load_data(cfg["data"]["path"])
    X = get_features(df, cfg["data"]["drop_columns"])

    params = {
        "n_clusters": cfg["model"]["n_clusters"],
        "n_components": cfg["model"]["n_components"],
        "rare_threshold": cfg["preprocessing"]["rare_threshold"],
        "random_state": cfg["model"]["random_state"],
    }
    pipe = build_pipeline(**params)

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run():
        labels = pipe.fit_predict(X)
        X_prep = pipe.named_steps["preprocess"].transform(X)

        mlflow.log_params(params)
        mlflow.log_metrics({
            "silhouette": silhouette_score(X_prep, labels),
            "davies_bouldin": davies_bouldin_score(X_prep, labels),
            "calinski_harabasz": calinski_harabasz_score(X_prep, labels),
            "inertia": pipe.named_steps["cluster"].inertia_,
        })
        mlflow.log_artifact("configs/config.yaml")
        mlflow.sklearn.log_model(
            pipe, name="model", skops_trusted_types=["numpy.dtype"]
        )


if __name__ == "__main__":
    main()