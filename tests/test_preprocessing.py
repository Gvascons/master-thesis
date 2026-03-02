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


class TestIntCodedCategoricalDetection:
    """Verify that int-coded categorical columns are treated as categoricals.

    OpenML often stores nominal features as int64 in the parquet file. After
    loading, the registry converts them to 'category' dtype. Preprocessing must
    then route them through the categorical pipeline, not the numeric one.
    """

    def test_int_coded_cat_is_ordinal_encoded_not_standardized(self):
        """Int column with category dtype goes through ordinal encoder, not StandardScaler."""
        # Simulate: OpenML nominal feature stored as int64, then cast to category
        cat_col = pd.Series([1, 2, 3, 1, 2, 3, 1], dtype="int64").astype("category")
        num_col = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype="float64")
        X = pd.DataFrame({"num_feat": num_col, "cat_feat": cat_col})

        info = DatasetInfo(
            name="test_int_cat",
            task_type="binary",
            n_classes=2,
            feature_types="mixed",
            cat_columns=["cat_feat"],
            num_columns=["num_feat"],
        )

        result = preprocess_for_gbdt(X, info)

        # Shape: 7 rows, 2 cols (1 num + 1 cat ordinal-encoded)
        assert result.X_train.shape == (7, 2)
        # The cat column values must be ordinal-encoded integers (0, 1, 2), not
        # the raw int codes (1, 2, 3) or standardized floats
        cat_vals = result.X_train[:, 1]  # cat col is second (after num)
        assert cat_vals.min() >= 0, "Ordinal-encoded values must be >= 0"
        assert cat_vals.max() <= 2, "Ordinal encoding of 3 unique values: max index = 2"

    def test_int_coded_cat_one_hot_in_dl_pipeline(self):
        """Int column with category dtype is one-hot encoded in the DL pipeline."""
        cat_col = pd.Series([10, 20, 30, 10, 20, 30], dtype="int64").astype("category")
        num_col = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float64")
        X = pd.DataFrame({"num_feat": num_col, "cat_feat": cat_col})

        info = DatasetInfo(
            name="test_int_cat_dl",
            task_type="binary",
            n_classes=2,
            feature_types="mixed",
            cat_columns=["cat_feat"],
            num_columns=["num_feat"],
        )

        result = preprocess_for_deep_learning(X, info)

        # 3 unique int categories → 3 one-hot columns; plus 1 standardized num col = 4 total
        assert result.X_train.shape == (6, 4), (
            f"Expected (6, 4) for 1 num + 3 one-hot cat, got {result.X_train.shape}"
        )
        # One-hot part (columns 1-3) should be binary
        one_hot_part = result.X_train[:, 1:]
        assert set(np.unique(one_hot_part)).issubset({0.0, 1.0})

    def test_int_coded_cat_excluded_from_num_columns(self):
        """Columns marked as categorical must NOT appear in num_columns."""
        # If a column appears in cat_columns, the DatasetInfo contract says
        # num_columns should not also contain it. Verify preprocessing respects this.
        X = pd.DataFrame({
            "num_a": [1.0, 2.0, 3.0],
            "cat_b": pd.Categorical([5, 10, 15]),
        })
        info = DatasetInfo(
            name="test_excl",
            task_type="binary",
            n_classes=2,
            feature_types="mixed",
            cat_columns=["cat_b"],
            num_columns=["num_a"],
        )
        result = preprocess_for_gbdt(X, info)
        # Expect 2 columns: 1 num + 1 cat ordinal
        assert result.X_train.shape == (3, 2)
        assert result.cat_feature_indices == [1]


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
