"""Smoke tests: every model can fit/predict on tiny synthetic data.

Tests all 10 models x {binary, multiclass, regression} task types, verifying:
- fit() completes without error
- predict() returns the correct shape
- predict_proba() returns the correct shape (classification) or raises (regression)
- is_fitted flag is set after fit
- GBDT reproducibility with same seed
"""

import numpy as np
import pytest

from src.models.factory import create_model
from src.utils.reproducibility import set_seed


# ── Synthetic data helpers ──────────────────────────────────────────────────

def _make_data(task_type, n_train=50, n_test=20, n_features=5, n_classes=2, seed=42):
    """Generate tiny synthetic datasets for smoke testing."""
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, n_features).astype(np.float32)
    X_val = rng.randn(n_test, n_features).astype(np.float32)
    X_test = rng.randn(n_test, n_features).astype(np.float32)

    if task_type == "binary":
        y_train = rng.randint(0, 2, size=n_train).astype(np.int64)
        y_val = rng.randint(0, 2, size=n_test).astype(np.int64)
    elif task_type == "multiclass":
        y_train = rng.randint(0, n_classes, size=n_train).astype(np.int64)
        y_val = rng.randint(0, n_classes, size=n_test).astype(np.int64)
    else:
        y_train = rng.randn(n_train).astype(np.float64)
        y_val = rng.randn(n_test).astype(np.float64)

    return X_train, y_train, X_val, y_val, X_test


# ── DL kwargs to keep tests fast ────────────────────────────────────────────

# Deep learning models need reduced epochs/architecture for fast smoke tests
_DL_FAST_KWARGS = {
    "ft_transformer": {"max_epochs": 2, "patience": 2, "d_block": 16, "n_blocks": 1, "attention_n_heads": 2},
    "tabnet": {"max_epochs": 2, "patience": 2, "n_d": 4, "n_a": 4, "n_steps": 2},
    "saint": {"max_epochs": 2, "patience": 2, "dim": 16, "depth": 1, "heads": 2},
    "tabm": {"max_epochs": 2, "patience": 2, "k": 4, "d_block": 16, "n_blocks": 1},
    "mlp": {"max_epochs": 2, "patience": 2, "d_hidden": 16, "n_blocks": 1},
    "realmlp": {"n_epochs": 2, "patience": 2, "hidden_width": 16, "n_hidden_layers": 1},
}

GBDT_MODELS = ["xgboost", "lightgbm", "catboost"]
DL_MODELS = ["ft_transformer", "tabnet", "saint", "tabm", "mlp"]
# RealMLP excluded from smoke tests: pytabkit (pytorch_lightning) segfaults on macOS
# when loaded alongside XGBoost/LightGBM due to duplicate OpenMP runtimes.
# It works on Linux (the actual experiment server) and is tested via test_factory.py.
FM_MODELS = ["tabpfn"]
ALL_MODELS = GBDT_MODELS + DL_MODELS + FM_MODELS


def _get_kwargs(model_name):
    """Return fast kwargs for DL models, empty dict for others."""
    return _DL_FAST_KWARGS.get(model_name, {})


# ── Binary classification smoke tests ───────────────────────────────────────

class TestBinarySmoke:
    """Binary classification: fit, predict, predict_proba for all models."""

    @pytest.mark.parametrize("model_name", GBDT_MODELS)
    def test_gbdt_binary(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("binary")
        model = create_model(model_name, "binary", n_classes=2, seed=42, **_get_kwargs(model_name))
        model.fit(X_tr, y_tr, X_va, y_va)

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({0, 1})

        proba = model.predict_proba(X_te)
        assert proba.shape == (20, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", DL_MODELS)
    def test_dl_binary(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("binary")
        model = create_model(model_name, "binary", n_classes=2, seed=42, **_get_kwargs(model_name))
        model.fit(X_tr, y_tr, X_va, y_va)

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)

        proba = model.predict_proba(X_te)
        assert proba.shape == (20, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", FM_MODELS)
    def test_fm_binary(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("binary")
        model = create_model(model_name, "binary", n_classes=2, seed=42)
        try:
            model.fit(X_tr, y_tr)
        except RuntimeError as e:
            if "authentication" in str(e).lower() or "gated" in str(e).lower():
                pytest.skip(f"{model_name} requires HuggingFace authentication")
            raise

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)

        proba = model.predict_proba(X_te)
        assert proba.shape == (20, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── Multiclass classification smoke tests ───────────────────────────────────

class TestMulticlassSmoke:
    """Multiclass classification: fit, predict, predict_proba for all models."""

    @pytest.mark.parametrize("model_name", GBDT_MODELS)
    def test_gbdt_multiclass(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("multiclass", n_classes=3)
        model = create_model(model_name, "multiclass", n_classes=3, seed=42, **_get_kwargs(model_name))
        model.fit(X_tr, y_tr, X_va, y_va)

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({0, 1, 2})

        proba = model.predict_proba(X_te)
        assert proba.shape == (20, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", DL_MODELS)
    def test_dl_multiclass(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("multiclass", n_classes=3)
        model = create_model(model_name, "multiclass", n_classes=3, seed=42, **_get_kwargs(model_name))
        model.fit(X_tr, y_tr, X_va, y_va)

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)

        proba = model.predict_proba(X_te)
        assert proba.shape == (20, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", FM_MODELS)
    def test_fm_multiclass(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("multiclass", n_classes=3)
        model = create_model(model_name, "multiclass", n_classes=3, seed=42)
        try:
            model.fit(X_tr, y_tr)
        except RuntimeError as e:
            if "authentication" in str(e).lower() or "gated" in str(e).lower():
                pytest.skip(f"{model_name} requires HuggingFace authentication")
            raise

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)

        proba = model.predict_proba(X_te)
        assert proba.shape == (20, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── Regression smoke tests ──────────────────────────────────────────────────

class TestRegressionSmoke:
    """Regression: fit, predict, predict_proba raises NotImplementedError."""

    @pytest.mark.parametrize("model_name", GBDT_MODELS)
    def test_gbdt_regression(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("regression")
        model = create_model(model_name, "regression", seed=42, **_get_kwargs(model_name))
        model.fit(X_tr, y_tr, X_va, y_va)

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)
        assert preds.dtype in (np.float32, np.float64)

        with pytest.raises(NotImplementedError):
            model.predict_proba(X_te)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", DL_MODELS)
    def test_dl_regression(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("regression")
        model = create_model(model_name, "regression", seed=42, **_get_kwargs(model_name))
        model.fit(X_tr, y_tr, X_va, y_va)

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)

        with pytest.raises(NotImplementedError):
            model.predict_proba(X_te)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", FM_MODELS)
    def test_fm_regression(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("regression")
        model = create_model(model_name, "regression", seed=42)
        try:
            model.fit(X_tr, y_tr)
        except RuntimeError as e:
            if "authentication" in str(e).lower() or "gated" in str(e).lower():
                pytest.skip(f"{model_name} requires HuggingFace authentication")
            raise

        assert model.is_fitted
        preds = model.predict(X_te)
        assert preds.shape == (20,)

        with pytest.raises(NotImplementedError):
            model.predict_proba(X_te)


# ── Reproducibility tests ─────────────────────────────────────────────────

class TestReproducibility:
    """GBDT models must produce identical predictions with the same seed."""

    @pytest.mark.parametrize("model_name", GBDT_MODELS)
    def test_gbdt_deterministic(self, model_name):
        X_tr, y_tr, X_va, y_va, X_te = _make_data("binary")
        kwargs = _get_kwargs(model_name)

        set_seed(42)
        m1 = create_model(model_name, "binary", n_classes=2, seed=42, **kwargs)
        m1.fit(X_tr, y_tr, X_va, y_va)
        preds1 = m1.predict(X_te)

        set_seed(42)
        m2 = create_model(model_name, "binary", n_classes=2, seed=42, **kwargs)
        m2.fit(X_tr, y_tr, X_va, y_va)
        preds2 = m2.predict(X_te)

        np.testing.assert_array_equal(
            preds1, preds2,
            err_msg=f"{model_name} is not deterministic with the same seed",
        )
