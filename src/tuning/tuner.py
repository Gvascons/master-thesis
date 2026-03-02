"""Unified Optuna tuning loop with inner cross-validation."""

from __future__ import annotations

import logging

import numpy as np
import optuna

from src.data.preprocessing import get_preprocessor
from src.data.registry import DatasetInfo, get_cv_folds
from src.evaluation.metrics import compute_primary_metric
from src.models.factory import create_model, get_model_family
from src.tuning.search_spaces import suggest_params
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed
from src.utils.timer import Timer

logger = logging.getLogger("tabular_benchmark")

# Suppress Optuna's default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(
    model_name: str,
    X_pool: np.ndarray | "pd.DataFrame",
    y_pool: np.ndarray,
    info: DatasetInfo,
    n_trials: int | None = None,
    inner_folds: int | None = None,
    seed: int | None = None,
) -> dict:
    """Run Optuna hyperparameter search with inner CV.

    Returns a dict with:
        - best_params: the best hyperparameters
        - best_score: the best inner CV metric
        - study: the Optuna study object
        - elapsed: total tuning time in seconds
    """
    import pandas as pd

    exp_cfg = load_experiment_config()
    n_trials = n_trials or exp_cfg.n_optuna_trials
    inner_folds = inner_folds or exp_cfg.inner_folds
    seed = seed if seed is not None else exp_cfg.seed

    # TabPFN is zero-shot — no tuning
    if model_name == "tabpfn":
        logger.info("TabPFN is zero-shot — skipping tuning")
        return {"best_params": {}, "best_score": None, "study": None, "elapsed": 0.0}

    model_family = get_model_family(model_name)
    preprocess_fn = get_preprocessor(model_family)

    # Convert to DataFrame if needed (for preprocessing)
    if isinstance(X_pool, np.ndarray):
        X_pool_df = pd.DataFrame(X_pool)
        # Adjust info for array input
        info_adj = DatasetInfo(
            name=info.name,
            task_type=info.task_type,
            n_classes=info.n_classes,
            feature_types="numerical",
            cat_columns=[],
            num_columns=list(X_pool_df.columns.astype(str)),
        )
    else:
        X_pool_df = X_pool
        info_adj = info

    # Generate inner CV folds
    folds = get_cv_folds(X_pool_df, y_pool, info_adj, n_splits=inner_folds, seed=seed)

    direction = "maximize"
    if info.task_type == "regression":
        direction = "minimize"  # RMSE — lower is better

    def objective(trial: optuna.Trial) -> float:
        trial_seed = seed + trial.number
        set_seed(trial_seed)
        params = suggest_params(trial, model_name)
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_tr = X_pool_df.iloc[train_idx]
            X_va = X_pool_df.iloc[val_idx]
            y_tr, y_va = y_pool[train_idx], y_pool[val_idx]

            # Preprocess
            prep = preprocess_fn(X_tr, info_adj, X_val=X_va)

            # Create and train model
            model_kwargs = dict(**params)
            if prep.cat_feature_indices is not None:
                model_kwargs["cat_feature_indices"] = prep.cat_feature_indices
            model = create_model(model_name, info.task_type, info.n_classes, seed=trial_seed, **model_kwargs)
            model.fit(prep.X_train, y_tr, prep.X_val, y_va)

            # Evaluate
            score = compute_primary_metric(model, prep.X_val, y_va, info.task_type)
            fold_scores.append(score)

            # Optuna pruning: report intermediate value
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    timer = Timer()
    with timer:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(
        f"Tuning {model_name}: best score={best.value:.4f}, "
        f"trials={len(study.trials)}, time={timer.result.elapsed:.1f}s"
    )

    return {
        "best_params": best.params,
        "best_score": best.value,
        "study": study,
        "elapsed": timer.result.elapsed,
    }
