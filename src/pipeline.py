from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

NUMERIC = ["Age", "Work_Experience", "Family_Size"]
BINARY = ["Gender", "Ever_Married", "Graduated"]
ORDINAL = ["Spending_Score"]
SPENDING_ORDER = [["Low", "Average", "High"]]


def build_preprocessor(rare_threshold: float = 0.05) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", RobustScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="if_binary", sparse_output=False)),
    ])
    ordinal = Pipeline([
        ("encode", OrdinalEncoder(categories=SPENDING_ORDER)),
        ("scale", RobustScaler()),
    ])
    profession = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    var_1 = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(
            min_frequency=rare_threshold,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("binary", binary, BINARY),
        ("ordinal", ordinal, ORDINAL),
        ("profession", profession, ["Profession"]),
        ("var_1", var_1, ["Var_1"]),
    ])

def build_pipeline(
    n_clusters: int = 4,
    n_components: int | None = None,
    rare_threshold: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    if n_components:
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        reducer = "passthrough"

    return Pipeline([
        ("preprocess", build_preprocessor(rare_threshold)),
        ("reduce", reducer),
        ("cluster", KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state
        )),
    ])
