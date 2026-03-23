"""Evaluation metrics for classification and regression."""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger("tabular_benchmark")


def ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov statistic for binary classification.

    KS measures the maximum separation between the cumulative distribution
    functions of predicted scores for the positive and negative classes.
    Widely used in credit scoring, fraud detection, and financial risk modelling
    to assess discriminative power independently of a threshold.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_score: Predicted probability of the positive class.

    Returns:
        KS statistic in [0, 1]. Higher is better (perfect separation = 1.0).
    """
    pos_scores = np.sort(y_score[y_true == 1])
    neg_scores = np.sort(y_score[y_true == 0])

    # Build empirical CDFs on a common set of thresholds
    all_scores = np.sort(np.concatenate([pos_scores, neg_scores]))
    cdf_pos = np.searchsorted(pos_scores, all_scores, side="right") / max(len(pos_scores), 1)
    cdf_neg = np.searchsorted(neg_scores, all_scores, side="right") / max(len(neg_scores), 1)

    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def compute_classification_metrics(y_true, y_pred, y_proba=None, task_type="binary"):
    """Compute classification metrics.

    Returns a dict of metric_name -> value.
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    if task_type == "binary":
        metrics["f1"] = f1_score(y_true, y_pred, average="binary")
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            metrics["log_loss"] = log_loss(y_true, y_proba)
            metrics["ks"] = ks_score(y_true, y_proba[:, 1])
    else:
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        if y_proba is not None:
            try:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted",
                )
            except ValueError as e:
                logger.warning("ROC-AUC computation failed: %s", e)
                metrics["roc_auc_ovr"] = np.nan
            metrics["log_loss"] = log_loss(y_true, y_proba)

    return metrics


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def compute_all_metrics(model, X, y_true, task_type):
    """Compute all relevant metrics given a fitted model."""
    y_pred = model.predict(X)

    if task_type in ("binary", "multiclass"):
        try:
            y_proba = model.predict_proba(X)
        except (NotImplementedError, AttributeError):
            y_proba = None
        return compute_classification_metrics(y_true, y_pred, y_proba, task_type)
    else:
        return compute_regression_metrics(y_true, y_pred)


def compute_primary_metric(model, X, y_true, task_type):
    """Compute the single primary metric for a task type.

    Returns a scalar value where higher is always better for classification
    (ROC-AUC for binary, negative log-loss for multiclass) and lower is
    better for regression (RMSE).
    """
    metrics = compute_all_metrics(model, X, y_true, task_type)

    if task_type == "binary":
        return metrics.get("roc_auc", metrics["accuracy"])
    elif task_type == "multiclass":
        return -metrics["log_loss"]  # negate so higher = better (matches tuner maximize)
    else:
        return metrics["rmse"]
