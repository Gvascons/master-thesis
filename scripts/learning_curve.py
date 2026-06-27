#!/usr/bin/env python3
"""Learning-curve / crossover experiment (RQ2).

For each (dataset, model), retrain on increasing slices of the training pool and
evaluate on the *same* fixed hold-out test set used by the main benchmark. The
question: at what training-set size does modern deep learning (TabM, RealMLP,
FT-Transformer) catch the GBDTs — and where does the foundation model (TabPFN)
sit on small vs large data?

Design choices (documented for the dissertation):
  * Hyperparameters are NOT re-tuned per size. We reuse the params Optuna already
    found on the full pool (results/raw/<model>_<dataset>.json) and hold them
    fixed, so the curve isolates the *data-size* effect at fixed model capacity.
    Early stopping on an internal val split lets effective capacity adapt
    somewhat. (Re-tuning per size is a heavier optional variant, not run here.)
  * The hold-out test split is regenerated with the same seed (42) and test_size
    as the benchmark, so curve endpoints are comparable to the main results.
  * Each (size, seed) draws a fresh stratified subsample of the pool; multiple
    seeds give error bars, which matter most at the small-data end.

Resumable like measure_latency.py: every (dataset, model, size, seed) row is
appended immediately; a restart skips rows already in the CSV.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/learning_curve.py
    # subset / override:
    uv run python scripts/learning_curve.py --models xgboost,tabm --datasets adult --seeds 1
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.preprocessing import get_preprocessor
from src.data.registry import get_holdout_split, load_dataset
from src.evaluation.metrics import compute_all_metrics
from src.models.factory import get_model_family
from src.utils.config import load_experiment_config
from src.utils.gpu import create_fit_with_retry, free_gpu_memory
from src.utils.reproducibility import set_seed

SEED = 42
TEST_SIZE = 0.2
OUT = REPO / "results" / "learning_curve" / "curve.csv"

# Curated contrast: 2 GBDTs, 3 deep nets, 1 foundation model.
DEFAULT_MODELS = ["xgboost", "catboost", "tabm", "realmlp", "ft_transformer", "tabpfn"]
# All three task types, each on a dataset large enough to sweep across sizes.
DEFAULT_DATASETS = [
    "give_me_some_credit",   # binary,    pool ~80k
    "higgs",                 # binary,    pool ~78k
    "adult",                 # binary,    pool ~39k (also the latency anchor)
    "jannis",                # multiclass pool ~67k (tests the multiclass reversal)
    "year_prediction",       # regression pool ~80k
]
# Log-ish spaced training-pool sizes; capped per dataset at the available pool.
DEFAULT_SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]

# Columns of compute_all_metrics we persist (blank when absent for the task).
METRIC_COLS = ["roc_auc", "ks", "log_loss", "accuracy", "rmse", "mae", "r2"]
HEADER = (
    ["dataset", "task_type", "model", "family", "train_size", "n_train",
     "n_val", "n_test", "seed", "fit_time_s"] + METRIC_COLS
)


def already_done():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], row["model"],
                          int(row["train_size"]), int(row["seed"])))
    return done


def append_row(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_best_params(model_name, dataset_name):
    jpath = REPO / "results" / "raw" / f"{model_name}_{dataset_name}.json"
    if not jpath.exists():
        return {}
    return json.loads(jpath.read_text()).get("best_params", {}) or {}


def prep_pool(dataset_name, exp_cfg):
    """Load, subsample to the benchmark cap, and carve the fixed hold-out test."""
    X, y, info = load_dataset(dataset_name)
    max_samples = getattr(exp_cfg, "max_dataset_samples", None)
    if max_samples and len(X) > max_samples:
        stratify = y if info.task_type in ("binary", "multiclass") else None
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, random_state=SEED, stratify=stratify)
    X_pool, y_pool, X_test, y_test = get_holdout_split(
        X, y, info, seed=SEED, test_size=TEST_SIZE)
    return X_pool, y_pool, X_test, y_test, info


def fit_eval(model_name, X_sub, y_sub, X_test, y_test, info, best_params, seed):
    """Preprocess, fit on a train/val split of the subsample, eval on test."""
    family = get_model_family(model_name)
    preprocess_fn = get_preprocessor(family)
    is_clf = info.task_type in ("binary", "multiclass")

    stratify = y_sub if is_clf else None
    Xtr, Xval, ytr, yval = train_test_split(
        X_sub, y_sub, test_size=0.1, random_state=seed, stratify=stratify)

    prep = preprocess_fn(Xtr, info, X_val=Xval, X_test=X_test)
    kwargs = dict(**best_params)
    if prep.cat_feature_indices is not None:
        kwargs["cat_feature_indices"] = prep.cat_feature_indices
    if model_name == "stab":
        kwargs["n_inference_samples"] = 64

    set_seed(seed)
    t0 = time.perf_counter()
    model = create_fit_with_retry(
        model_name, info.task_type, info.n_classes, seed, kwargs,
        prep.X_train, ytr, prep.X_val, yval,
    )
    fit_time = time.perf_counter() - t0
    metrics = compute_all_metrics(model, prep.X_test, y_test, info.task_type)

    del model
    free_gpu_memory()
    return fit_time, metrics, len(Xtr), len(Xval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--seeds", default=3, type=int, help="number of repeat seeds")
    args = ap.parse_args()

    models = args.models.split(",")
    datasets = args.datasets.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = list(range(args.seeds))

    exp_cfg = load_experiment_config()
    done = already_done()
    total = 0

    for ds in datasets:
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        pool_n = len(X_pool)
        # cap sizes at the available pool; always include the full pool as the top point
        ds_sizes = sorted({min(s, pool_n) for s in sizes} | {pool_n})
        print(f"\n=== {ds} ({info.task_type}) pool={pool_n} test={len(X_test)} "
              f"sizes={ds_sizes} ===", flush=True)

        for m in models:
            best_params = load_best_params(m, ds)
            for s in ds_sizes:
                # TabPFN targets <=50k rows; above that it subsamples to 50k
                # internally (so the point is redundant) and the >50k 4-config
                # ensemble path can deadlock the GPU. Keep TabPFN to its
                # supported regime; the curve still spans 500..50k for it.
                if m == "tabpfn" and s > 50000:
                    continue
                for seed in seeds:
                    if (ds, m, s, seed) in done:
                        continue
                    is_clf = info.task_type in ("binary", "multiclass")
                    if s < pool_n:
                        strat = y_pool if is_clf else None
                        X_sub, _, y_sub, _ = train_test_split(
                            X_pool, y_pool, train_size=s,
                            random_state=seed, stratify=strat)
                    else:
                        X_sub, y_sub = X_pool, y_pool
                    try:
                        ft, metrics, n_tr, n_va = fit_eval(
                            m, X_sub, y_sub, X_test, y_test, info, best_params, seed)
                    except Exception as e:
                        print(f"  [FAIL] {m} {ds} size={s} seed={seed}: "
                              f"{type(e).__name__}: {e}", flush=True)
                        free_gpu_memory()
                        continue
                    row = {
                        "dataset": ds, "task_type": info.task_type, "model": m,
                        "family": get_model_family(m), "train_size": s,
                        "n_train": n_tr, "n_val": n_va, "n_test": len(X_test),
                        "seed": seed, "fit_time_s": round(ft, 3),
                    }
                    for c in METRIC_COLS:
                        v = metrics.get(c)
                        row[c] = round(float(v), 6) if v is not None else ""
                    append_row(row)
                    total += 1
                    key = (metrics.get("roc_auc") if is_clf and info.task_type == "binary"
                           else metrics.get("log_loss") if info.task_type == "multiclass"
                           else metrics.get("rmse"))
                    print(f"  [{m:14s} size={s:6d} seed={seed}] "
                          f"primary={key:.4f}  fit={ft:.1f}s", flush=True)

    print(f"\nALL DONE — {total} new rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
