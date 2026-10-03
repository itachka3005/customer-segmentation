import argparse

from mlflow import MlflowClient

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the champion model version")
    parser.add_argument("version", help="Model version to promote")
    args = parser.parse_args()

    cfg = load_config()
    name = cfg["mlflow"]["registered_model_name"]
    MlflowClient().set_registered_model_alias(name, "champion", args.version)
    print(f"{name} v{args.version} is now champion")


if __name__ == "__main__":
    main()