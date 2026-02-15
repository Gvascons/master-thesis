"""Tests for CV fold generation and holdout splitting in src.data.registry."""

import numpy as np
import pytest

from src.data.registry import DatasetInfo, get_cv_folds, get_holdout_split


class TestFoldSizes:
    def test_fold_sizes_balanced(self, binary_data):
        """100 samples, 5 folds -> each val fold should have ~20 samples."""
        X, y, info = binary_data
        folds = get_cv_folds(X, y, info, n_splits=5, seed=42)

        assert len(folds) == 5
        for train_idx, val_idx in folds:
            assert len(val_idx) == pytest.approx(20, abs=1)
            assert len(train_idx) == pytest.approx(80, abs=1)


class TestNoOverlap:
    def test_no_overlap_between_folds(self, binary_data):
        """Train and val indices must be disjoint within each fold."""
        X, y, info = binary_data
        folds = get_cv_folds(X, y, info, n_splits=5, seed=42)

        for train_idx, val_idx in folds:
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Train/val overlap: {overlap}"


class TestAllIndicesCovered:
    def test_all_indices_covered(self, binary_data):
        """Union of all val indices should equal {0, 1, ..., n-1}."""
        X, y, info = binary_data
        n = len(X)
        folds = get_cv_folds(X, y, info, n_splits=5, seed=42)

        all_val = set()
        for _, val_idx in folds:
            all_val.update(val_idx)

        assert all_val == set(range(n))


class TestStratification:
    def test_stratification_for_classification(self):
        """Classification folds should preserve class proportions."""
        rng = np.random.RandomState(42)
        n = 200
        # Imbalanced: 80% class 0, 20% class 1
        y = np.array([0] * 160 + [1] * 40)
        import pandas as pd
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        info = DatasetInfo(
            name="imbalanced",
            task_type="binary",
            n_classes=2,
            feature_types="numerical",
            cat_columns=[],
            num_columns=["a", "b", "c"],
        )

        folds = get_cv_folds(X, y, info, n_splits=5, seed=42)

        overall_ratio = np.mean(y == 1)  # 0.2
        for _, val_idx in folds:
            fold_ratio = np.mean(y[val_idx] == 1)
            assert fold_ratio == pytest.approx(overall_ratio, abs=0.05), (
                f"Fold class ratio {fold_ratio} deviates from overall {overall_ratio}"
            )

    def test_kfold_for_regression(self, regression_data):
        """Regression should use KFold (not StratifiedKFold) and still produce valid folds."""
        X, y, info = regression_data
        assert info.task_type == "regression"

        folds = get_cv_folds(X, y, info, n_splits=5, seed=42)
        assert len(folds) == 5

        # All indices covered
        all_val = set()
        for _, val_idx in folds:
            all_val.update(val_idx)
        assert all_val == set(range(len(X)))


class TestHoldoutSplit:
    def test_holdout_split_size(self, binary_data):
        """test_size=0.2 should put ~20% in the test set."""
        X, y, info = binary_data
        X_pool, y_pool, X_test, y_test = get_holdout_split(X, y, info, test_size=0.2)

        assert len(X_test) == pytest.approx(20, abs=2)
        assert len(X_pool) == pytest.approx(80, abs=2)
        assert len(y_pool) + len(y_test) == len(y)

    def test_holdout_split_stratified(self):
        """For classification, holdout should preserve class proportions."""
        rng = np.random.RandomState(42)
        n = 500
        y = np.array([0] * 400 + [1] * 100)  # 80/20 split
        import pandas as pd
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        info = DatasetInfo(
            name="imbalanced",
            task_type="binary",
            n_classes=2,
            feature_types="numerical",
            cat_columns=[],
            num_columns=["a", "b", "c"],
        )

        _, _, X_test, y_test = get_holdout_split(X, y, info, test_size=0.2, seed=42)

        test_ratio = np.mean(y_test == 1)
        overall_ratio = np.mean(y == 1)  # 0.2
        assert test_ratio == pytest.approx(overall_ratio, abs=0.05)
