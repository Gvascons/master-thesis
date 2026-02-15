"""Tests for src.data.preprocessing — data leakage, encoding, imputation."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    PreprocessedData,
    get_preprocessor,
    preprocess_for_deep_learning,
    preprocess_for_gbdt,
)
from src.data.registry import DatasetInfo


def _make_info(num_columns, cat_columns, task_type="binary"):
    return DatasetInfo(
        name="test",
        task_type=task_type,
        n_classes=2 if task_type == "binary" else None,
        feature_types="mixed" if cat_columns else "numerical",
        cat_columns=cat_columns,
        num_columns=num_columns,
    )


class TestGBDTPreprocessing:
    def test_gbdt_no_data_leakage(self):
        """Unseen category in val should map to -1 (unknown)."""
        info = _make_info(num_columns=[], cat_columns=["color"])
        X_train = pd.DataFrame({"color": ["A", "B", "A", "B"]})
        X_val = pd.DataFrame({"color": ["C"]})

        result = preprocess_for_gbdt(X_train, info, X_val=X_val)

        # OrdinalEncoder with handle_unknown="use_encoded_value", unknown_value=-1
        assert result.X_val[0, 0] == -1, (
            "Unseen category in validation must be encoded as -1"
        )

    def test_gbdt_imputer_median_from_train(self):
        """NaN in val should be filled with train median, not val median."""
        info = _make_info(num_columns=["x"], cat_columns=[])
        # Train median = 5.0
        X_train = pd.DataFrame({"x": [1.0, 5.0, 5.0, 9.0]})
        X_val = pd.DataFrame({"x": [np.nan]})

        result = preprocess_for_gbdt(X_train, info, X_val=X_val)

        # Imputed with train median (5.0), not val median
        assert result.X_val[0, 0] == pytest.approx(5.0), (
            "NaN in val should be imputed with train median"
        )


class TestDLPreprocessing:
    def test_dl_standardization_uses_train_stats(self):
        """Validation data should be standardized using train mean/std."""
        info = _make_info(num_columns=["x"], cat_columns=[])
        X_train = pd.DataFrame({"x": [0.0, 0.0, 0.0]})
        X_val = pd.DataFrame({"x": [100.0]})

        result = preprocess_for_deep_learning(X_train, info, X_val=X_val)

        # Train mean=0, std=0 -> StandardScaler will have std clamped or set to 1.
        # The key assertion: val is NOT 0 (which it would be if using val's own stats)
        # With train mean=0 and std~=0, sklearn StandardScaler sets std to 1 when
        # constant features are encountered, so val should be transformed to 100.0
        assert result.X_val[0, 0] != pytest.approx(0.0), (
            "Val must be standardized with train stats, not its own"
        )

    def test_dl_onehot_ignores_unseen_categories(self):
        """Unseen category in val should produce a zero vector (handle_unknown='ignore')."""
        info = _make_info(num_columns=[], cat_columns=["color"])
        X_train = pd.DataFrame({"color": ["A", "B", "A", "B"]})
        X_val = pd.DataFrame({"color": ["C"]})

        result = preprocess_for_deep_learning(X_train, info, X_val=X_val)

        # With handle_unknown="ignore", an unseen category => all zeros
        assert np.all(result.X_val[0] == 0.0), (
            "Unseen category in val must produce an all-zero one-hot vector"
        )


class TestPreprocessAllNumerical:
    def test_preprocess_all_numerical(self, binary_data):
        """Preprocessing should work when cat_columns is empty."""
        X, y, info = binary_data
        assert info.cat_columns == []

        result = preprocess_for_gbdt(X, info)
        assert isinstance(result, PreprocessedData)
        assert result.X_train.shape == (100, 5)

        result_dl = preprocess_for_deep_learning(X, info)
        assert isinstance(result_dl, PreprocessedData)
        assert result_dl.X_train.shape == (100, 5)


class TestPreprocessAllCategorical:
    def test_preprocess_all_categorical(self):
        """Preprocessing should work when num_columns is empty."""
        info = _make_info(num_columns=[], cat_columns=["a", "b"])
        X = pd.DataFrame({"a": ["X", "Y", "X"], "b": ["P", "Q", "P"]})

        result = preprocess_for_gbdt(X, info)
        assert isinstance(result, PreprocessedData)
        assert result.X_train.shape == (3, 2)

        result_dl = preprocess_for_deep_learning(X, info)
        assert isinstance(result_dl, PreprocessedData)
        # 2 categories for 'a' + 2 categories for 'b' = 4 one-hot columns
        assert result_dl.X_train.shape == (3, 4)


class TestGetPreprocessorMapping:
    def test_known_families(self):
        """Each model family should map to the correct preprocessing function."""
        gbdt_fn = get_preprocessor("gbdt")
        dl_fn = get_preprocessor("deep_learning")
        fm_fn = get_preprocessor("foundation_model")

        assert gbdt_fn is preprocess_for_gbdt
        assert dl_fn is preprocess_for_deep_learning
        # foundation_model delegates to tabpfn preprocessor
        assert callable(fm_fn)

    def test_unknown_family_raises(self):
        """An unknown family should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model family"):
            get_preprocessor("transformers_are_all_you_need")
