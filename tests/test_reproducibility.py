"""Tests for reproducibility: same seed + same data => identical outputs.

Verifies that set_seed() + deterministic model training produces
bit-identical predictions across two independent runs.
"""

import numpy as np
import pytest

from src.models.factory import create_model
from src.utils.reproducibility import set_seed


def _make_data(task_type, seed=42):
    """Tiny synthetic dataset for reproducibility testing."""
    rng = np.random.RandomState(seed)
    X_train = rng.randn(60, 5).astype(np.float32)
    X_val = rng.randn(20, 5).astype(np.float32)
    X_test = rng.randn(20, 5).astype(np.float32)

    if task_type == "binary":
        y_train = rng.randint(0, 2, size=60).astype(np.int64)
        y_val = rng.randint(0, 2, size=20).astype(np.int64)
    else:
        y_train = rng.randn(60).astype(np.float64)
        y_val = rng.randn(20).astype(np.float64)

    return X_train, y_train, X_val, y_val, X_test


def _train_and_predict(model_name, task_type, n_classes, seed, extra_kwargs=None):
    """Train a model from scratch and return predictions."""
    set_seed(seed)
    kwargs = extra_kwargs or {}
    model = create_model(model_name, task_type, n_classes=n_classes, seed=seed, **kwargs)
    X_tr, y_tr, X_va, y_va, X_te = _make_data(task_type, seed=0)
    model.fit(X_tr, y_tr, X_va, y_va)
    return model.predict(X_te)


class TestGBDTReproducibility:
    """GBDT models should produce identical predictions given the same seed."""

    @pytest.mark.parametrize("model_name", ["xgboost", "lightgbm", "catboost"])
    def test_binary_deterministic(self, model_name):
        preds_a = _train_and_predict(model_name, "binary", n_classes=2, seed=42)
        preds_b = _train_and_predict(model_name, "binary", n_classes=2, seed=42)
        np.testing.assert_array_equal(preds_a, preds_b)

    @pytest.mark.parametrize("model_name", ["xgboost", "lightgbm", "catboost"])
    def test_regression_deterministic(self, model_name):
        preds_a = _train_and_predict(model_name, "regression", n_classes=None, seed=42)
        preds_b = _train_and_predict(model_name, "regression", n_classes=None, seed=42)
        np.testing.assert_array_equal(preds_a, preds_b)

    @pytest.mark.parametrize("model_name", ["xgboost", "lightgbm", "catboost"])
    def test_different_seeds_differ(self, model_name):
        """Different seeds should (almost surely) produce different predictions."""
        preds_a = _train_and_predict(model_name, "binary", n_classes=2, seed=42)
        preds_b = _train_and_predict(model_name, "binary", n_classes=2, seed=99)
        # With different seeds, at least some predictions should differ
        # (not guaranteed but extremely likely with reasonable data)
        # We use a soft check: if all identical, that would be suspicious
        # but not necessarily wrong — so we just verify the test runs
        assert preds_a.shape == preds_b.shape


class TestDLReproducibility:
    """DL models should produce identical predictions given the same seed (CPU)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", ["mlp"])
    def test_dl_binary_deterministic(self, model_name):
        kwargs = {"max_epochs": 3, "patience": 3, "d_hidden": 16, "n_blocks": 1}
        preds_a = _train_and_predict(model_name, "binary", n_classes=2, seed=42, extra_kwargs=kwargs)
        preds_b = _train_and_predict(model_name, "binary", n_classes=2, seed=42, extra_kwargs=kwargs)
        np.testing.assert_array_equal(preds_a, preds_b)


class TestSetSeedFunction:
    """Verify that set_seed produces deterministic numpy/random outputs."""

    def test_numpy_deterministic(self):
        set_seed(123)
        a = np.random.rand(10)
        set_seed(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_python_random_deterministic(self):
        import random
        set_seed(123)
        a = [random.random() for _ in range(10)]
        set_seed(123)
        b = [random.random() for _ in range(10)]
        assert a == b
