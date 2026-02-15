"""Tests for src.models.factory — model creation and registry."""

import pytest

from src.models.base import BaseModel
from src.models.factory import create_model, get_model_class, get_model_family, list_models


EXPECTED_MODELS = [
    "xgboost", "lightgbm", "catboost",
    "ft_transformer", "tabnet", "saint",
    "tabpfn", "tabm", "realmlp", "mlp",
]

EXPECTED_FAMILIES = {
    "xgboost": "gbdt",
    "lightgbm": "gbdt",
    "catboost": "gbdt",
    "ft_transformer": "deep_learning",
    "tabnet": "deep_learning",
    "saint": "deep_learning",
    "tabpfn": "foundation_model",
    "tabm": "deep_learning",
    "realmlp": "deep_learning",
    "mlp": "deep_learning",
}


class TestCreateAllModels:
    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_create_model(self, model_name):
        """create_model should succeed for every registered model name."""
        model = create_model(model_name, task_type="binary", n_classes=2)
        assert isinstance(model, BaseModel)
        assert model.task_type == "binary"
        assert model.n_classes == 2


class TestModelNameAttribute:
    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_model_name_and_family(self, model_name):
        """MODEL_NAME and FAMILY class attributes should be correct."""
        cls = get_model_class(model_name)
        assert cls.MODEL_NAME == model_name
        assert cls.FAMILY == EXPECTED_FAMILIES[model_name]


class TestUnknownModel:
    def test_unknown_model_raises(self):
        """Creating an unknown model should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model", task_type="binary")


class TestListModels:
    def test_list_models_returns_all(self):
        """list_models should return all 10 expected model names."""
        names = list_models()
        assert len(names) == 10
        for expected in EXPECTED_MODELS:
            assert expected in names


class TestGetModelFamily:
    @pytest.mark.parametrize("model_name,expected_family", EXPECTED_FAMILIES.items())
    def test_correct_family(self, model_name, expected_family):
        """get_model_family should return the right family for each model."""
        assert get_model_family(model_name) == expected_family
