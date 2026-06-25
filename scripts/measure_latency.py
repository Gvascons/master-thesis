#!/usr/bin/env python3
"""Measure inference latency (us/row) for all 11 models on the `adult` dataset.

Loads each model's tuned best_params from results/raw/<model>_adult.json, fits
on the training pool, and times predict() over the hold-out test set (median of
N_REPEATS passes after a warm-up, with CUDA sync). STab uses its full
n_inference_samples=64. Writes results incrementally to results/latency/ after
EACH model so an interrupted run never loses completed measurements; on restart
it skips models already present in the CSV.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/measure_latency.py
"""
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

DATASET = "adult"
SEED = 42
TEST_SIZE = 0.2
N_REPEATS = 5
OUT = REPO / "results" / "latency" / "latency_adult.csv"

MODELS = [
    "xgboost", "lightgbm", "catboost", "tabpfn",
    "mlp", "tabm", "realmlp", "tabnet",
    "ft_transformer", "saint", "stab",
]


def already_done():
    done = {}
    if OUT.exists():
        with open(OUT) as f:
            for row in csv.DictReader(f):
                done[row["model"]] = row
    return done


def append_row(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "family", "n_test_rows", "us_per_row", "ms_total", "n_repeats"]
    write_header = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


def measure(model_name, X_pool, y_pool, X_test, y_test, info):
    family = get_model_family(model_name)
    preprocess_fn = get_preprocessor(family)

    # best params from the saved experiment JSON
    jpath = REPO / "results" / "raw" / f"{model_name}_{DATASET}.json"
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
    _ = model.predict(Xte)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass

    times = []
    for _ in range(N_REPEATS):
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
    X, y, info = load_dataset(DATASET)
    print(f"{DATASET}: {len(X)} rows, task={info.task_type}, n_classes={info.n_classes}", flush=True)
    X_pool, y_pool, X_test, y_test = get_holdout_split(X, y, info, seed=SEED, test_size=TEST_SIZE)
    print(f"pool={len(X_pool)}  test={len(X_test)}", flush=True)

    done = already_done()
    for m in MODELS:
        if m in done:
            print(f"[skip] {m} already in CSV ({done[m]['us_per_row']} us/row)", flush=True)
            continue
        print(f"[run ] {m} ...", flush=True)
        t0 = time.perf_counter()
        row = measure(m, X_pool, y_pool, X_test, y_test, info)
        append_row(row)
        print(f"[done] {m}: {row['us_per_row']} us/row  "
              f"({row['ms_total']} ms / {row['n_test_rows']} rows)  "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    print("ALL DONE -> " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
