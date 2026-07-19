#!/usr/bin/env python3
"""Pareto frontier of FM-acceleration strategies (AI-2 phase 3, RQ-D3/H3).

For each dataset, measures ACCURACY (RMSE + CRPS where distributional) and
INFERENCE LATENCY (us/row on the benchmark hold-out) of every strategy:

  teacher_ctx    TabPFN with context subsampled to {1k, 5k, 10k, 25k, full<=50k}
                 (the "compress the context" strategy), n_estimators=8
  teacher_ens    TabPFN full context with n_estimators {1, 4} ("shrink the
                 ensemble"; 8 == the full teacher_ctx point)
  student_*      distilled students retrained from the cached OOF targets
                 (xgb_quant/soft = the H2 winner; xgb_point/hard and
                 xgb_quant/hard = controls)
  tabfm_full     TabFM point teacher at full context (accuracy + latency
                 anchor of the strongest model)

Latency protocol: warm-up chunk, then ONE timed full-hold-out pass for the
teachers (a median-of-5 would 5x hours of GPU; deviation from
measure_latency.py documented) and median-of-5 for the cheap students.

Datasets: the three with positive teacher gaps (california_housing,
diamonds, year_prediction). Rows appended incrementally to
results/distillation/pareto.csv; resumable.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/pareto_strategies.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from distill import (QUANTILES, TEACHER_MAX_ROWS, OOF_DIR, crps_from_quantiles,
                     fit_xgb_point, fit_xgb_quant_hard, fit_xgb_quant_soft,
                     free_teacher, load_xgb_params, ordinal_encode, prep_pool,
                     sort_quantiles, teacher_predict)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

OUT = REPO / "results" / "distillation" / "pareto.csv"
DATASETS = ["california_housing", "diamonds", "year_prediction"]
CTX_GRID = [1000, 5000, 10000, 25000, TEACHER_MAX_ROWS]
ENS_GRID = [1, 4]
HEADER = ["dataset", "system", "config", "rmse", "crps", "us_per_row",
          "n_test", "fit_time_s"]


def already_done():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["system"], r["config"]))
    return done


def append(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def subsample(X, y, n, seed=42):
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def timed_teacher_eval(model, Xt, y_test, teacher="tabpfn"):
    """Warm-up on a small chunk, then one timed full pass (see module doc)."""
    _ = teacher_predict(model, Xt[:200], teacher=teacher)
    t0 = time.perf_counter()
    m, Q = teacher_predict(model, Xt, teacher=teacher)
    dt = time.perf_counter() - t0
    rmse = float(np.sqrt(np.mean((y_test - m) ** 2)))
    crps = round(crps_from_quantiles(y_test, Q), 6) if Q is not None else ""
    return rmse, crps, dt / len(y_test) * 1e6


def timed_student_predict(model, Xt, is_quant):
    """Median of 5 timed passes (students are cheap)."""
    _ = model.predict(Xt)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        pred = model.predict(Xt)
        times.append(time.perf_counter() - t0)
    us = float(np.median(times)) / len(Xt) * 1e6
    if is_quant:
        Q = sort_quantiles(np.asarray(pred))
        point = Q[:, QUANTILES.index(0.50)]
        return point, Q, us
    return pred, None, us


def main():
    exp_cfg = load_experiment_config()
    done = already_done()

    for ds in DATASETS:
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        n_test = len(y_test)
        params = load_xgb_params(ds)
        print(f"\n=== {ds}: pool={len(y_pool)} test={n_test} ===", flush=True)

        # ---- estrategia 1: contexto comprimido (TabPFN, n_est=8) ----
        from tabpfn import TabPFNRegressor
        for ctx in CTX_GRID:
            ctx_eff = min(ctx, len(y_pool))
            key = (ds, "teacher_ctx", str(ctx_eff))
            if key in done or (ctx != ctx_eff and (ds, "teacher_ctx", str(ctx_eff)) in done):
                print(f"  [skip] teacher_ctx {ctx_eff}", flush=True)
                continue
            Xs, ys = subsample(Xp, y_pool, ctx_eff)
            model = TabPFNRegressor(device="auto", n_estimators=8, random_state=42)
            t0 = time.perf_counter()
            model.fit(Xs, ys)
            ft = time.perf_counter() - t0
            rmse, crps, us = timed_teacher_eval(model, Xt, y_test)
            free_teacher(model)
            append({"dataset": ds, "system": "teacher_ctx", "config": str(ctx_eff),
                    "rmse": round(rmse, 6), "crps": crps, "us_per_row": round(us, 2),
                    "n_test": n_test, "fit_time_s": round(ft, 2)})
            print(f"  [teacher_ctx {ctx_eff:6d}] rmse={rmse:.4f} crps={crps} "
                  f"lat={us:.0f}us/row", flush=True)
            if ctx_eff == len(y_pool):
                break

        # ---- estrategia 2: ensemble reduzido (contexto cheio) ----
        for ens in ENS_GRID:
            key = (ds, "teacher_ens", str(ens))
            if key in done:
                print(f"  [skip] teacher_ens {ens}", flush=True)
                continue
            Xs, ys = subsample(Xp, y_pool, TEACHER_MAX_ROWS)
            model = TabPFNRegressor(device="auto", n_estimators=ens, random_state=42)
            t0 = time.perf_counter()
            model.fit(Xs, ys)
            ft = time.perf_counter() - t0
            rmse, crps, us = timed_teacher_eval(model, Xt, y_test)
            free_teacher(model)
            append({"dataset": ds, "system": "teacher_ens", "config": str(ens),
                    "rmse": round(rmse, 6), "crps": crps, "us_per_row": round(us, 2),
                    "n_test": n_test, "fit_time_s": round(ft, 2)})
            print(f"  [teacher_ens {ens}] rmse={rmse:.4f} lat={us:.0f}us/row", flush=True)

        # ---- estrategia 3: alunos destilados (dos alvos OOF cacheados) ----
        oof = pd.read_parquet(OOF_DIR / f"tabpfn_{ds}.parquet")
        qcols = [f"q{int(t*100):02d}" for t in QUANTILES]
        students = [
            ("student_point_hard", False, lambda: fit_xgb_point(Xp, y_pool, params, 0)),
            ("student_quant_hard", True, lambda: fit_xgb_quant_hard(Xp, y_pool, params, 0)),
            ("student_quant_soft", True, lambda: fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(), params, 0)),
        ]
        for name, is_quant, fit_fn in students:
            key = (ds, name, "tuned")
            if key in done:
                print(f"  [skip] {name}", flush=True)
                continue
            set_seed(0)
            t0 = time.perf_counter()
            model = fit_fn()
            ft = time.perf_counter() - t0
            point, Q, us = timed_student_predict(model, Xt, is_quant)
            rmse = float(np.sqrt(np.mean((y_test - point) ** 2)))
            crps = round(crps_from_quantiles(y_test, Q), 6) if Q is not None else ""
            append({"dataset": ds, "system": name, "config": "tuned",
                    "rmse": round(rmse, 6), "crps": crps, "us_per_row": round(us, 2),
                    "n_test": n_test, "fit_time_s": round(ft, 2)})
            print(f"  [{name}] rmse={rmse:.4f} crps={crps} lat={us:.1f}us/row", flush=True)

        # ---- ancora TabFM (ponto, contexto cheio) ----
        key = (ds, "tabfm_full", "8")
        if key not in done:
            from distill import make_teacher
            model = make_teacher("auto", "tabfm")
            t0 = time.perf_counter()
            model.fit(Xp, y_pool)
            ft = time.perf_counter() - t0
            rmse, crps, us = timed_teacher_eval(model, Xt, y_test, teacher="tabfm")
            free_teacher(model)
            append({"dataset": ds, "system": "tabfm_full", "config": "8",
                    "rmse": round(rmse, 6), "crps": "", "us_per_row": round(us, 2),
                    "n_test": n_test, "fit_time_s": round(ft, 2)})
            print(f"  [tabfm_full] rmse={rmse:.4f} lat={us:.0f}us/row", flush=True)

    print("\nPARETO DONE", flush=True)


if __name__ == "__main__":
    main()
