"""Per-family preprocessing pipelines.

- GBDTs: minimal preprocessing (label-encode categoricals)
- Deep Learning: standardize numericals, one-hot encode categoricals
- TabPFN: minimal, handles preprocessing internally
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.data.registry import DatasetInfo

logger = logging.getLogger("tabular_benchmark")


@dataclass
class PreprocessedData:
    """Holds transformed arrays and the fitted preprocessor."""

    X_train: np.ndarray
    X_val: np.ndarray | None
    X_test: np.ndarray | None
    preprocessor: ColumnTransformer | None
    cat_feature_indices: list[int] | None = None


def preprocess_for_gbdt(
    X_train: pd.DataFrame,
    info: DatasetInfo,
    X_val: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
) -> PreprocessedData:
    """GBDT preprocessing: ordinal-encode categoricals, impute NaNs."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, info.num_columns),
            ("cat", cat_pipe, info.cat_columns),
        ],
        remainder="drop",
    )

    X_tr = preprocessor.fit_transform(X_train)
    X_v = preprocessor.transform(X_val) if X_val is not None else None
    X_te = preprocessor.transform(X_test) if X_test is not None else None

    # Compute cat feature indices in the transformed array:
    # ColumnTransformer outputs num columns first, then cat columns
    n_num = len(info.num_columns)
    n_cat = len(info.cat_columns)
    cat_indices = list(range(n_num, n_num + n_cat)) if n_cat > 0 else None

    return PreprocessedData(
        X_train=X_tr, X_val=X_v, X_test=X_te,
        preprocessor=preprocessor, cat_feature_indices=cat_indices,
    )


def preprocess_for_deep_learning(
    X_train: pd.DataFrame,
    info: DatasetInfo,
    X_val: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
) -> PreprocessedData:
    """Deep learning preprocessing: standardize numericals, one-hot categoricals."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if info.num_columns:
        transformers.append(("num", num_pipe, info.num_columns))
    if info.cat_columns:
        transformers.append(("cat", cat_pipe, info.cat_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_tr = preprocessor.fit_transform(X_train).astype(np.float32)
    X_v = preprocessor.transform(X_val).astype(np.float32) if X_val is not None else None
    X_te = preprocessor.transform(X_test).astype(np.float32) if X_test is not None else None

    return PreprocessedData(X_train=X_tr, X_val=X_v, X_test=X_te, preprocessor=preprocessor)


def preprocess_for_tabpfn(
    X_train: pd.DataFrame,
    info: DatasetInfo,
    X_val: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
) -> PreprocessedData:
    """TabPFN preprocessing: minimal — ordinal encode categoricals, impute NaNs.

    TabPFN handles feature normalization internally.
    """
    return preprocess_for_gbdt(X_train, info, X_val, X_test)


def get_preprocessor(model_family: str):
    """Return the appropriate preprocessing function for a model family."""
    mapping = {
        "gbdt": preprocess_for_gbdt,
        "deep_learning": preprocess_for_deep_learning,
        "foundation_model": preprocess_for_tabpfn,
    }
    if model_family not in mapping:
        raise ValueError(f"Unknown model family: {model_family}. Choose from {list(mapping.keys())}")
    return mapping[model_family]
