#!/usr/bin/env python3
"""CLI: Train/tune any model+dataset combination.

Usage:
    python scripts/train.py --model xgboost --dataset adult
    python scripts/train.py --model ft_transformer --dataset adult --gpu 0
    python scripts/train.py --model all --dataset all
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import get_preprocessor
from src.data.registry import DatasetInfo, get_all_dataset_names, get_cv_folds, get_holdout_split, load_dataset
from src.evaluation.metrics import compute_all_metrics
from src.models.factory import create_model, get_model_family, list_models
from src.tuning.tuner import tune_model
from src.utils.config import load_experiment_config
from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed
from src.utils.timer import Timer


def train_single(
    model_name: str,
    dataset_name: str,
    results_dir: Path,
    gpu: int | None = None,
    seed: int = 42,
):
    """Train a single model on a single dataset with full experimental protocol."""
    logger = get_logger()
    exp_cfg = load_experiment_config()

    set_seed(seed)

    # Set GPU if specified
    if gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Load data
    X, y, info = load_dataset(dataset_name)

    # Hold-out test split
    X_pool, y_pool, X_test, y_test = get_holdout_split(
        X, y, info, seed=seed, test_size=exp_cfg.test_size,
    )

    # Phase 1: Hyperparameter tuning (inner CV)
    logger.info(f"=== Tuning {model_name} on {dataset_name} ===")
    tune_result = tune_model(model_name, X_pool, y_pool, info, seed=seed)
    best_params = tune_result["best_params"]
    logger.info(f"Best params: {best_params}")

    # Phase 2: Outer CV evaluation with best params
    logger.info(f"=== Outer CV ({exp_cfg.outer_folds}-fold) ===")
    folds = get_cv_folds(X_pool, y_pool, info, n_splits=exp_cfg.outer_folds, seed=seed)
    model_family = get_model_family(model_name)
    preprocess_fn = get_preprocessor(model_family)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_seed = seed + fold_idx
        set_seed(fold_seed)

        X_tr = X_pool.iloc[train_idx]
        X_va = X_pool.iloc[val_idx]
        y_tr, y_va = y_pool[train_idx], y_pool[val_idx]

        prep = preprocess_fn(X_tr, info, X_val=X_va)
        model_kwargs = dict(**best_params)
        if prep.cat_feature_indices is not None:
            model_kwargs["cat_feature_indices"] = prep.cat_feature_indices
        model = create_model(model_name, info.task_type, info.n_classes, seed=fold_seed, **model_kwargs)

        timer = Timer()
        with timer:
            model.fit(prep.X_train, y_tr, prep.X_val, y_va)

        metrics = compute_all_metrics(model, prep.X_val, y_va, info.task_type)
        metrics["fold"] = fold_idx
        metrics["train_time_s"] = timer.result.elapsed
        fold_results.append(metrics)

        logger.info(f"  Fold {fold_idx}: {metrics}")

    # Phase 3: Final evaluation on hold-out test set
    logger.info("=== Final test set evaluation ===")
    stratify = y_pool if info.task_type != "regression" else None
    X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
        X_pool, y_pool, test_size=0.1, random_state=seed, stratify=stratify,
    )
    prep_final = preprocess_fn(X_final_train, info, X_val=X_final_val, X_test=X_test)
    final_kwargs = dict(**best_params)
    if prep_final.cat_feature_indices is not None:
        final_kwargs["cat_feature_indices"] = prep_final.cat_feature_indices
    final_model = create_model(model_name, info.task_type, info.n_classes, seed=seed, **final_kwargs)

    timer = Timer()
    with timer:
        final_model.fit(prep_final.X_train, y_final_train, prep_final.X_val, y_final_val)

    test_metrics = compute_all_metrics(final_model, prep_final.X_test, y_test, info.task_type)
    test_metrics["train_time_s"] = timer.result.elapsed
    logger.info(f"Test metrics: {test_metrics}")

    # Save results
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "task_type": info.task_type,
        "best_params": best_params,
        "tuning_time_s": tune_result["elapsed"],
        "fold_results": fold_results,
        "test_metrics": test_metrics,
        "seed": seed,
    }

    out_dir = results_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    return result


@click.command()
@click.option("--model", "-m", required=True, help="Model name or 'all'")
@click.option("--dataset", "-d", required=True, help="Dataset name or 'all'")
@click.option("--gpu", "-g", default=None, type=int, help="GPU device ID")
@click.option("--results-dir", default="results", type=click.Path())
@click.option("--seed", default=42, type=int)
@click.option("--verbose", "-v", is_flag=True)
def main(model, dataset, gpu, results_dir, seed, verbose):
    """Train and evaluate models on tabular datasets."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    results_path = Path(results_dir)

    models = list_models() if model == "all" else [model]
    datasets = get_all_dataset_names() if dataset == "all" else [dataset]

    for ds in datasets:
        for m in models:
            try:
                train_single(m, ds, results_path, gpu=gpu, seed=seed)
            except Exception as e:
                get_logger().error(f"Failed {m} on {ds}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
