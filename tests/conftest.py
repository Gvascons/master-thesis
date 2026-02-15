"""Shared fixtures for the P0 test suite."""

# Prevent OpenMP segfaults on macOS: XGBoost/LightGBM and PyTorch ship
# different OpenMP runtimes that conflict when both are loaded.
# Setting OMP_NUM_THREADS=1 avoids the multi-threading crash.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest

from src.data.registry import DatasetInfo


@pytest.fixture
def binary_data():
    """100 samples, 5 numerical features, binary target."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 5), columns=[f"num_{i}" for i in range(5)])
    y = rng.randint(0, 2, size=100).astype(np.int64)
    info = DatasetInfo(
        name="test_binary",
        task_type="binary",
        n_classes=2,
        feature_types="numerical",
        cat_columns=[],
        num_columns=[f"num_{i}" for i in range(5)],
    )
    return X, y, info


@pytest.fixture
def multiclass_data():
    """150 samples, 5 numerical features, 3 classes."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(150, 5), columns=[f"num_{i}" for i in range(5)])
    y = rng.randint(0, 3, size=150).astype(np.int64)
    info = DatasetInfo(
        name="test_multiclass",
        task_type="multiclass",
        n_classes=3,
        feature_types="numerical",
        cat_columns=[],
        num_columns=[f"num_{i}" for i in range(5)],
    )
    return X, y, info


@pytest.fixture
def regression_data():
    """100 samples, 5 numerical features, continuous target."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 5), columns=[f"num_{i}" for i in range(5)])
    y = rng.randn(100).astype(np.float64)
    info = DatasetInfo(
        name="test_regression",
        task_type="regression",
        n_classes=None,
        feature_types="numerical",
        cat_columns=[],
        num_columns=[f"num_{i}" for i in range(5)],
    )
    return X, y, info


@pytest.fixture
def mixed_data():
    """100 samples with both numerical and categorical features."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "num_0": rng.randn(100),
        "num_1": rng.randn(100),
        "num_2": rng.randn(100),
        "cat_0": rng.choice(["A", "B", "C"], size=100),
        "cat_1": rng.choice(["X", "Y"], size=100),
    })
    # Ensure categorical dtype
    df["cat_0"] = df["cat_0"].astype("category")
    df["cat_1"] = df["cat_1"].astype("category")
    y = rng.randint(0, 2, size=100).astype(np.int64)
    info = DatasetInfo(
        name="test_mixed",
        task_type="binary",
        n_classes=2,
        feature_types="mixed",
        cat_columns=["cat_0", "cat_1"],
        num_columns=["num_0", "num_1", "num_2"],
    )
    return df, y, info
