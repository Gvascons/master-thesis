#!/usr/bin/env python3
"""Measure inference latency (us/row) for the 14 models on multiple datasets.

Multi-dataset generalization of scripts/measure_latency.py (which produced
results/latency/latency_adult.csv for E2's cost proxy). The measurement
protocol is identical, model by model:

  * best_params loaded from results/raw/<model>_<dataset>.json (foundation
    models have empty best_params there -> defaults, same as the adult run);
  * refit on a 90/10 train/val split of the hold-out pool (seed 42,
    test_size 0.2, stratified for classification), mirroring final-eval;
  * timing = median of N_REPEATS=5 full predict() passes over the hold-out
    test set, after 1 warm-up pass, with torch.cuda.synchronize() around
    each pass; device per family is whatever the wrapper chooses (GBDT on
    CPU, DL/foundation on CUDA), unchanged;
  * STab runs with n_inference_samples=64 (full Bayesian averaging);
  * TabFM's predict_proba cache is cleared before every timed pass.

Resumable: each (model, dataset) row is appended to
results/latency/latency_multi.csv immediately after measurement; on restart,
pairs already in the CSV are skipped.

IMPORTANT: latency numbers are only meaningful on an otherwise idle machine.
Run this ALONE — no concurrent benchmark/distillation jobs on CPU or GPU —
and preferably under reduced priority, e.g.:

    PYTORCH_ALLOC_CONF=expandable_segments:True \
        nice -n 15 uv run python scripts/latency_multi.py \
        --datasets credit_g cardiovascular_disease

Usage:
    uv run python scripts/latency_multi.py --datasets <ds1> <ds2> ...
        [--models m1 m2 ...]   # subset of models (default: all 14)
        [--out PATH]           # default results/latency/latency_multi.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.preprocessing import get_preprocessor
from src.data.registry import get_holdout_split, load_dataset
from src.models.factory import create_model, get_model_family
from src.utils.reproducibility import set_seed

SEED = 42
TEST_SIZE = 0.2
N_REPEATS = 5
DEFAULT_OUT = REPO / "results" / "latency" / "latency_multi.csv"

MODELS = [
    "xgboost", "lightgbm", "catboost", "tabpfn",
    "mlp", "tabm", "realmlp", "tabnet",
    "ft_transformer", "saint", "stab",
    "kan", "tabkan", "tabfm",
]

HEADER = ["model", "dataset", "family", "n_test_rows", "us_per_row", "ms_total", "n_repeats"]


def _reset_prediction_cache(model):
    """TabFM's wrapper caches the last predict_proba (metrics-computation
    optimization); latency timing must clear it so every pass pays full
    inference cost."""
    if hasattr(model, "_proba_cache"):
        model._proba_cache = None


def already_done(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                done.add((row["model"], row["dataset"]))
    return done


def append_row(out_path, row):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)


def measure(model_name, dataset, X_pool, y_pool, X_test, y_test, info):
    family = get_model_family(model_name)
    preprocess_fn = get_preprocessor(family)

    # best params from the saved experiment JSON
    jpath = REPO / "results" / "raw" / f"{model_name}_{dataset}.json"
    best_params = json.loads(jpath.read_text()).get("best_params", {}) or {}

    # fit on a train/val split of the pool (mirrors final-eval protocol)
    stratify = y_pool if info.task_type != "regression" else None
    Xtr, Xval, ytr, yval = train_test_split(
        X_pool, y_pool, test_size=0.1, random_state=SEED, stratify=stratify,
    )
    prep = preprocess_fn(Xtr, info, X_val=Xval, X_test=X_test)

    kwargs = dict(**best_params)
    if prep.cat_feature_indices is not None:
        kwargs["cat_feature_indices"] = prep.cat_feature_indices
    if model_name == "stab":
        kwargs["n_inference_samples"] = 64  # full Bayesian averaging at inference

    set_seed(SEED)
    model = create_model(model_name, info.task_type, info.n_classes, SEED, **kwargs)
    model.fit(prep.X_train, ytr, prep.X_val, yval)

    Xte = prep.X_test
    n = Xte.shape[0]

    # warm-up (build any lazy CUDA graphs / caches)
    _reset_prediction_cache(model)
    _ = model.predict(Xte)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass

    times = []
    for _ in range(N_REPEATS):
        _reset_prediction_cache(model)
        t0 = time.perf_counter()
        _ = model.predict(Xte)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        times.append(time.perf_counter() - t0)

    med = float(np.median(times))
    row = {
        "model": model_name,
        "dataset": dataset,
        "family": family,
        "n_test_rows": n,
        "us_per_row": round(med / n * 1e6, 2),
        "ms_total": round(med * 1e3, 2),
        "n_repeats": N_REPEATS,
    }

    del model
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="dataset names from configs/datasets.yaml (parquet must exist in data/raw)")
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help=f"subset of models (default: all {len(MODELS)})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output CSV (default: results/latency/latency_multi.csv)")
    args = ap.parse_args()

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        ap.error(f"unknown model(s): {unknown}")

    done = already_done(args.out)

    for dataset in args.datasets:
        X, y, info = load_dataset(dataset)
        print(f"{dataset}: {len(X)} rows, task={info.task_type}, n_classes={info.n_classes}", flush=True)
        X_pool, y_pool, X_test, y_test = get_holdout_split(X, y, info, seed=SEED, test_size=TEST_SIZE)
        print(f"pool={len(X_pool)}  test={len(X_test)}", flush=True)

        for m in args.models:
            if (m, dataset) in done:
                print(f"[skip] {m} x {dataset} already in CSV", flush=True)
                continue
            jpath = REPO / "results" / "raw" / f"{m}_{dataset}.json"
            if not jpath.exists():
                print(f"[skip] {m} x {dataset}: no raw JSON at {jpath} (structural exclusion?)", flush=True)
                continue
            print(f"[run ] {m} x {dataset} ...", flush=True)
            t0 = time.perf_counter()
            row = measure(m, dataset, X_pool, y_pool, X_test, y_test, info)
            append_row(args.out, row)
            print(f"[done] {m} x {dataset}: {row['us_per_row']} us/row  "
                  f"({row['ms_total']} ms / {row['n_test_rows']} rows)  "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    print("ALL DONE -> " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
