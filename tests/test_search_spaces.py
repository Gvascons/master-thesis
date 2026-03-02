"""Tests for src.tuning.search_spaces — YAML search space validation."""

import numpy as np
import pytest
import optuna

from src.models.factory import create_model, list_models
from src.tuning.search_spaces import get_search_space_names, suggest_params
from src.utils.config import load_models_config


class TestAllModelsHaveSearchSpace:
    def test_every_factory_model_has_yaml_entry(self):
        """Every model in the factory must have a YAML entry (TabPFN has null)."""
        cfg = load_models_config()
        factory_models = list_models()

        for model_name in factory_models:
            assert model_name in cfg, (
                f"Model '{model_name}' registered in factory but missing from models.yaml"
            )

    def test_tabpfn_has_no_tunable_params(self):
        """TabPFN is zero-shot and should have null/empty search space."""
        cfg = load_models_config()
        # TabPFN entry should be None/null (zero-shot)
        tabpfn_space = cfg.get("tabpfn")
        assert tabpfn_space is None or len(tabpfn_space) == 0, (
            "TabPFN should have no tunable hyperparameters"
        )


class TestSuggestParamsReturnsValidTypes:
    @pytest.mark.parametrize("model_name", [
        "xgboost", "lightgbm", "catboost",
        "ft_transformer", "tabnet", "saint", "tabm", "mlp", "realmlp",
    ])
    def test_suggest_params_valid(self, model_name):
        """suggest_params should return a dict of valid Python types."""
        # Use a fixed Optuna trial to get deterministic results
        study = optuna.create_study()
        trial = study.ask()

        params = suggest_params(trial, model_name)

        assert isinstance(params, dict)
        assert len(params) > 0, f"Expected non-empty params for {model_name}"

        for key, value in params.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float, str, bool)), (
                f"Param {key}={value} has unexpected type {type(value)}"
            )


class TestParamRangesValid:
    def test_param_ranges_low_lt_high_and_log_positive(self):
        """For all int/float params: low < high; for log params: low > 0."""
        cfg = load_models_config()

        for model_name in cfg:
            space = cfg[model_name]
            if space is None:
                continue

            for param_name, spec in space.items():
                if spec is None:
                    continue

                ptype = spec.get("type")
                if ptype in ("int", "float"):
                    low = spec["low"]
                    high = spec["high"]
                    assert low < high, (
                        f"{model_name}.{param_name}: low ({low}) >= high ({high})"
                    )

                    if spec.get("log", False):
                        assert low > 0, (
                            f"{model_name}.{param_name}: log scale requires low > 0, got {low}"
                        )


# ── TUNABLE_MODELS: all models with search spaces except zero-shot TabPFN ───

_TUNABLE_MODELS = [
    "xgboost", "lightgbm", "catboost",
    "ft_transformer", "tabnet", "saint", "tabm", "mlp",
    # realmlp excluded from end-to-end fit test (macOS OpenMP conflict)
]

# Fast kwargs so the e2e test doesn't spin for long
_DL_FAST_KWARGS = {
    "ft_transformer": {"max_epochs": 2, "patience": 2, "d_block": 16, "n_blocks": 1, "attention_n_heads": 2},
    "tabnet": {"max_epochs": 2, "patience": 2, "n_d": 4, "n_a": 4, "n_steps": 2},
    "saint": {"max_epochs": 2, "patience": 2, "dim": 16, "depth": 1, "heads": 2},
    "tabm": {"max_epochs": 2, "patience": 2, "k": 4, "d_block": 16, "n_blocks": 1},
    "mlp": {"max_epochs": 2, "patience": 2, "d_hidden": 16, "n_blocks": 1},
}


class TestSuggestParamsEndToEnd:
    """Verify suggest_params -> create_model -> fit works for every tunable model.

    This catches type mismatches (e.g. categorical param returning a numpy int64
    that a model __init__ cannot handle).
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", _TUNABLE_MODELS)
    def test_suggest_create_fit(self, model_name):
        """End-to-end: suggest params, create model, fit on tiny binary data."""
        rng = np.random.RandomState(0)
        X_tr = rng.randn(40, 4).astype(np.float32)
        y_tr = rng.randint(0, 2, size=40).astype(np.int64)
        X_va = rng.randn(10, 4).astype(np.float32)
        y_va = rng.randint(0, 2, size=10).astype(np.int64)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = suggest_params(trial, model_name)

        # Override with fast kwargs so DL models finish in seconds
        fast_overrides = _DL_FAST_KWARGS.get(model_name, {})
        params.update(fast_overrides)

        model = create_model(model_name, task_type="binary", n_classes=2,
                             seed=0, **params)
        # Should not raise — type errors in params surface here
        model.fit(X_tr, y_tr, X_va, y_va)

        assert model.is_fitted
        preds = model.predict(X_va)
        assert preds.shape == (10,)
