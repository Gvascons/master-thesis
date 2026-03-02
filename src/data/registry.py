"""Dataset registry: loading, splitting, and cross-validation fold generation."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.utils.config import load_datasets_config, load_experiment_config

logger = logging.getLogger("tabular_benchmark")


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset."""

    name: str
    task_type: str  # "binary", "multiclass", "regression"
    n_classes: int | None
    feature_types: str  # "numerical", "mixed"
    cat_columns: list[str]
    num_columns: list[str]


@dataclass
class DataSplit:
    """A train/test or train/val split."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    info: DatasetInfo


def load_dataset(name: str, data_dir: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, DatasetInfo]:
    """Load a dataset from parquet and return (X, y, info)."""
    datasets_cfg = load_datasets_config()
    exp_cfg = load_experiment_config()

    if data_dir is None:
        data_dir = Path(exp_cfg.data_dir)

    # Find config
    ds_cfg = None
    for task_group in ("classification", "regression"):
        if name in datasets_cfg[task_group]:
            ds_cfg = datasets_cfg[task_group][name]
            break
    if ds_cfg is None:
        raise ValueError(f"Unknown dataset: {name}")

    parquet_path = data_dir / f"{name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {parquet_path}. Run 'python scripts/download_data.py' first."
        )

    df = pd.read_parquet(parquet_path)
    y = df["target"].values
    X = df.drop(columns=["target"])

    # Force nominal features (which OpenML may have stored as int64) to category dtype
    # by cross-referencing OpenML feature metadata
    if hasattr(ds_cfg, "openml_id") and ds_cfg.openml_id:
        try:
            import openml
            oml_dataset = openml.datasets.get_dataset(
                ds_cfg.openml_id,
                download_data=False,
                download_qualities=False,
                download_features_meta_data=True,
            )
            nominal_features = [
                f.name
                for f in oml_dataset.features.values()
                if f.data_type == "nominal" and f.name != oml_dataset.default_target_attribute
            ]
            for col in nominal_features:
                if col in X.columns:
                    X[col] = X[col].astype("category")
        except Exception as e:
            logger.warning(f"Could not fetch OpenML feature metadata for '{name}': {e}")

    # Identify categorical vs numerical columns
    cat_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_columns = X.select_dtypes(include=["number"]).columns.tolist()

    # Encode target for classification
    if ds_cfg.task_type in ("binary", "multiclass"):
        if y.dtype == object or isinstance(y[0], str):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float64)

    n_classes = int(ds_cfg.get("n_classes", 0)) or None

    info = DatasetInfo(
        name=name,
        task_type=ds_cfg.task_type,
        n_classes=n_classes,
        feature_types=ds_cfg.feature_types,
        cat_columns=cat_columns,
        num_columns=num_columns,
    )

    logger.info(
        f"Loaded '{name}': {X.shape[0]} samples, {X.shape[1]} features "
        f"({len(num_columns)} num, {len(cat_columns)} cat), task={ds_cfg.task_type}"
    )
    return X, y, info


def get_holdout_split(
    X: pd.DataFrame,
    y: np.ndarray,
    info: DatasetInfo,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Split into train+val pool and hold-out test set."""
    stratify = y if info.task_type in ("binary", "multiclass") else None
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify,
    )
    return X_pool, y_pool, X_test, y_test


def get_cv_folds(
    X: pd.DataFrame,
    y: np.ndarray,
    info: DatasetInfo,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate CV fold indices (train_idx, val_idx)."""
    if info.task_type in ("binary", "multiclass"):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(X, y))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(X))


def get_all_dataset_names() -> list[str]:
    """Return a flat list of all dataset names from the config."""
    datasets_cfg = load_datasets_config()
    names = []
    for task_group in ("classification", "regression"):
        names.extend(list(datasets_cfg[task_group].keys()))
    return names
