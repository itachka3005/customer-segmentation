# Customer Segmentation

End-to-end ML project: customer clustering with a reproducible scikit-learn pipeline, experiment tracking and model registry in MLflow.

## Problem

An automobile company plans to enter new markets with its existing products. In the current market, the sales team has split customers into 4 segments and uses a different outreach strategy for each of them.

The goal of this project is to build an unsupervised segmentation pipeline that:
- groups customers by their demographic profile without using the existing labels;
- assigns a segment to any new customer;
- produces segments that can be described and used by the marketing team.

## Data

[Customer Segmentation dataset](https://www.kaggle.com/datasets/vetrirah/customer) (Analytics Vidhya Janatahack, available on Kaggle).

- 8068 customers, 9 features: `Gender`, `Ever_Married`, `Age`, `Graduated`, `Profession`, `Work_Experience`, `Spending_Score`, `Family_Size`, `Var_1`.
- `Segmentation` (A–D) is the ground-truth segment from the sales team. It is **not** used for training, only for comparison with the obtained clusters.

Download `train.csv` and put it into `data/raw/`.

## Approach

### EDA

Key findings (`notebooks/01_eda.ipynb`):
- 6 of 9 features are categorical, so encoding is the core of preprocessing.
- 6 columns have missing values; the largest share is in `Work_Experience` (~10%).
- Missing values in `Family_Size` and `Work_Experience` are informative: customers with a missing value belong to segment D much more often.
- Numeric features are right-skewed and contain a few real outliers.
- 417 rows share identical features but have different IDs. These are different customers with the same profile, so they are kept.

### Preprocessing

Implemented as a scikit-learn `Pipeline` with `ColumnTransformer` (`src/pipeline.py`), so the same transformations are applied during training and to new customers.

| Features | Transformation |
|---|---|
| `Age`, `Work_Experience`, `Family_Size` | Median imputation + missing indicators, `RobustScaler` |
| `Gender`, `Ever_Married`, `Graduated` | Most frequent imputation, binary 0/1 encoding |
| `Spending_Score` | Ordinal encoding (Low < Average < High), `RobustScaler` |
| `Profession` | Missing values as a separate category, one-hot encoding |
| `Var_1` | Missing values as a separate category, one-hot encoding, categories below 5% grouped together |

Result: 24 numeric features.

### Experiments

KMeans with k from 2 to 8 was compared across three options: no PCA, PCA with 13 components (90% of variance) and PCA with 2 components. All metrics are computed in the same preprocessed feature space, so the options are directly comparable.

Silhouette score:

| k | No PCA | PCA 13 | PCA 2 |
|---|---|---|---|
| 2 | 0.156 | 0.156 | 0.154 |
| 3 | 0.171 | 0.171 | 0.155 |
| 4 | 0.158 | 0.158 | 0.123 |
| 5 | 0.146 | 0.145 | 0.117 |
| 6 | 0.135 | 0.135 | 0.093 |
| 7 | 0.135 | 0.136 | 0.096 |
| 8 | 0.134 | 0.137 | 0.087 |

Conclusions:
- PCA with 13 components gives the same quality as no PCA, so the simpler model without PCA is used.
- PCA with 2 components loses too much information and is used only for visualization.
- **k = 4** was chosen: best Davies–Bouldin score (1.89), elbow of the inertia curve, close silhouette to k = 3 (0.158 vs 0.171), and direct comparability with the 4 business segments.

## Tech stack

Python 3.11, pandas, scikit-learn, MLflow, matplotlib, ruff

## Project structure

```
configs/
  config.yaml                  # data paths, hyperparameters, MLflow settings
data/                          # raw and processed data (not tracked by git)
notebooks/
  01_eda.ipynb                 # exploratory data analysis
  02_preprocessing_prototype.ipynb
src/
  config.py                    # config loading
  data.py                      # data loading
  pipeline.py                  # preprocessing + clustering pipeline
  train.py                     # training with MLflow tracking and registration
  sweep.py                     # hyperparameter sweep
  promote.py                   # set the champion model version
tests/
```

## How to run

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/itachka3005/customer-segmentation.git
cd customer-segmentation
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements-dev.txt
```

2. Download the dataset into `data/raw/train.csv`.

3. Run the hyperparameter sweep (optional):

```bash
python -m src.sweep
```

4. Train the final model and register it in MLflow:

```bash
python -m src.train
```

5. Promote a model version to `champion`:

```bash
python -m src.promote 1
```

6. Explore experiments in the MLflow UI at http://127.0.0.1:5000:

```bash
mlflow ui
```

## Roadmap

- [x] Project scaffolding
- [x] Exploratory data analysis
- [x] Training pipeline
- [x] Experiment tracking with MLflow
- [x] Model registry
- [ ] Segment interpretation
- [ ] REST API (FastAPI)
- [ ] Docker
- [ ]