import mlflow

from src.config import load_config
from src.train import load_features, params_from_config, train_and_log

K_VALUES = range(2, 9)
PCA_OPTIONS = [None, 13, 2]


def main() -> None:
    cfg = load_config()
    X = load_features(cfg)
    base_params = params_from_config(cfg)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    for n_components in PCA_OPTIONS:
        for k in K_VALUES:
            params = {**base_params, "n_clusters": k, "n_components": n_components}
            run_name = f"k={k}_pca={n_components or 'none'}"
            metrics = train_and_log(X, params, run_name=run_name, log_model=False)
            print(f"{run_name}: silhouette={metrics['silhouette']:.3f}")


if __name__ == "__main__":
    main()