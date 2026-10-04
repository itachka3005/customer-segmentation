import argparse
import shutil
from pathlib import Path

import mlflow.sklearn

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export champion model to a folder")
    parser.add_argument("--output", default="models/champion")
    args = parser.parse_args()

    cfg = load_config()
    uri = f"models:/{cfg['mlflow']['registered_model_name']}@champion"
    output = Path(args.output)

    model = mlflow.sklearn.load_model(uri)
    if output.exists():
        shutil.rmtree(output)
    mlflow.sklearn.save_model(
        model, path=str(output), skops_trusted_types=["numpy.dtype"]
    )
    print(f"Exported {uri} to {output}")


if __name__ == "__main__":
    main()