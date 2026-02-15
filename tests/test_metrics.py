"""Tests for src.evaluation.metrics — classification and regression metrics."""

import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import roc_auc_score as sk_roc_auc

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


class TestClassificationMetricsPerfect:
    def test_perfect_binary(self):
        """Perfect binary predictions should give accuracy=1 and f1=1."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])

        metrics = compute_classification_metrics(y_true, y_pred, task_type="binary")

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_perfect_multiclass(self):
        """Perfect multiclass predictions should give accuracy=1 and f1_macro=1."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_classification_metrics(y_true, y_pred, task_type="multiclass")

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["f1_weighted"] == pytest.approx(1.0)


class TestBinaryMetricsWithProba:
    def test_roc_auc_and_log_loss(self):
        """Verify ROC-AUC and log loss against sklearn for known inputs."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8],
        ])
        y_pred = np.array([0, 0, 1, 1])

        metrics = compute_classification_metrics(
            y_true, y_pred, y_proba=y_proba, task_type="binary"
        )

        expected_auc = sk_roc_auc(y_true, y_proba[:, 1])
        expected_ll = sk_log_loss(y_true, y_proba)

        assert metrics["roc_auc"] == pytest.approx(expected_auc)
        assert metrics["log_loss"] == pytest.approx(expected_ll)
        # AUC should be 1.0 for these well-separated predictions
        assert metrics["roc_auc"] == pytest.approx(1.0)


class TestMulticlassMetrics:
    def test_multiclass_with_proba(self):
        """3-class predictions, cross-check with sklearn."""
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 1])  # one mistake
        y_proba = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.5, 0.2],
        ])

        metrics = compute_classification_metrics(
            y_true, y_pred, y_proba=y_proba, task_type="multiclass"
        )

        assert metrics["accuracy"] == pytest.approx(3.0 / 4.0)
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "log_loss" in metrics

        expected_ll = sk_log_loss(y_true, y_proba)
        assert metrics["log_loss"] == pytest.approx(expected_ll)


class TestRegressionMetrics:
    def test_known_values(self):
        """Hand-verified regression metrics."""
        y_true = np.array([3.0, 0.0, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        metrics = compute_regression_metrics(y_true, y_pred)

        # RMSE = sqrt(mean([0.25, 0, 0, 1])) = sqrt(0.3125) ~ 0.5590
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        # MAE = mean([0.5, 0, 0, 1]) = 0.375
        expected_mae = np.mean(np.abs(y_true - y_pred))
        # R2 = 1 - SS_res/SS_tot
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected_r2 = 1.0 - ss_res / ss_tot

        assert metrics["rmse"] == pytest.approx(expected_rmse)
        assert metrics["mae"] == pytest.approx(expected_mae)
        assert metrics["r2"] == pytest.approx(expected_r2)

        # Sanity checks on the hand-computed values
        assert metrics["rmse"] == pytest.approx(0.5590, abs=0.001)
        assert metrics["mae"] == pytest.approx(0.375)


class TestPrimaryMetric:
    """Test compute_primary_metric returns the right metric for each task type.

    compute_primary_metric requires a fitted model, so we use a mock.
    """

    def test_binary_returns_auc(self):
        """For binary classification, primary metric should be ROC-AUC."""
        from unittest.mock import MagicMock
        from src.evaluation.metrics import compute_primary_metric

        model = MagicMock()
        model.predict.return_value = np.array([0, 0, 1, 1])
        model.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]
        ])

        X = np.zeros((4, 2))
        y_true = np.array([0, 0, 1, 1])

        result = compute_primary_metric(model, X, y_true, "binary")

        # Should return roc_auc, which is 1.0 for these well-separated predictions
        assert result == pytest.approx(1.0)

    def test_regression_returns_rmse(self):
        """For regression, primary metric should be RMSE."""
        from unittest.mock import MagicMock
        from src.evaluation.metrics import compute_primary_metric

        model = MagicMock()
        model.predict.return_value = np.array([2.5, 0.0, 2.0, 8.0])

        X = np.zeros((4, 2))
        y_true = np.array([3.0, 0.0, 2.0, 7.0])

        result = compute_primary_metric(model, X, y_true, "regression")

        expected_rmse = np.sqrt(np.mean((y_true - model.predict.return_value) ** 2))
        assert result == pytest.approx(expected_rmse)

    def test_multiclass_returns_neg_log_loss(self):
        """For multiclass, primary metric should be negative log_loss (higher=better for tuner)."""
        from unittest.mock import MagicMock
        from src.evaluation.metrics import compute_primary_metric

        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 2, 1])
        model.predict_proba.return_value = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.5, 0.2],
        ])

        X = np.zeros((4, 2))
        y_true = np.array([0, 1, 2, 0])

        result = compute_primary_metric(model, X, y_true, "multiclass")

        expected = -sk_log_loss(y_true, model.predict_proba.return_value)
        assert result == pytest.approx(expected)
