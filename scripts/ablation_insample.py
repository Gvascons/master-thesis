#!/usr/bin/env python3
"""Ablation (AI-2 phase 4): OOF vs in-sample teacher labeling.

Pocket FM's methodological claim (classification): an ICL teacher scoring
its own training context yields collapsed/overconfident targets, so OOF
labeling is required. This ablation replicates the question in REGRESSION:
students trained on in-sample targets (teacher fit on the full pool,
predicting that same pool) vs the OOF targets already cached.

Datasets: the four whose pool fits the 50K context entirely (wine_quality,
california_housing, superconduct, diamonds) — year_prediction is excluded
because 30K of its 80K rows would fall outside the context, making the
"in-sample" condition only partial (documented design choice).

Output: results/distillation/ablation_insample.csv with regime={oof,insample}
rows for xgb_point/soft and xgb_quant/soft (3 seeds); the OOF rows are
copied from distill.csv for side-by-side reading. Resumable.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/ablation_insample.py
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

from distill import (QUANTILES, OOF_DIR, crps_from_quantiles, fit_teacher,
                     fit_xgb_point, fit_xgb_quant_soft, free_teacher,
                     interval_metrics, load_xgb_params, make_teacher,
                     ordinal_encode, prep_pool, sort_quantiles,
                     teacher_predict)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

OUT = REPO / "results" / "distillation" / "ablation_insample.csv"
DATASETS = ["wine_quality", "california_housing", "superconduct", "kin8nm"]
SEEDS = [0, 1, 2]
HEADER = ["dataset", "regime", "student", "seed", "rmse", "crps",
          "picp80", "fit_time_s"]


def already_done():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["regime"], r["student"], int(r["seed"])))
    return done


def append(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def insample_targets(ds, Xp, y_pool, device="auto"):
    """Teacher fit on the full pool, predicting that same pool (cached)."""
    path = OOF_DIR / f"tabpfn_{ds}_insample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    model = make_teacher(device)
    fit_teacher(model, Xp, y_pool)
    mean, Q = teacher_predict(model, Xp)
    free_teacher(model)
    df = pd.DataFrame(Q, columns=[f"q{int(t*100):02d}" for t in QUANTILES])
    df.insert(0, "mean", mean)
    df.insert(0, "y", y_pool)
    df.to_parquet(path)
    return df


def eval_student(model, Xt, y_test, is_quant):
    pred = model.predict(Xt)
    if is_quant:
        Q = sort_quantiles(np.asarray(pred))
        point = Q[:, QUANTILES.index(0.50)]
        crps = round(crps_from_quantiles(y_test, Q), 6)
        picp = round(interval_metrics(y_test, Q)["picp80"], 4)
    else:
        point, crps, picp = pred, "", ""
    rmse = round(float(np.sqrt(np.mean((y_test - point) ** 2))), 6)
    return rmse, crps, picp


def main():
    exp_cfg = load_experiment_config()
    done = already_done()
    qcols = [f"q{int(t*100):02d}" for t in QUANTILES]

    # copia as linhas OOF correspondentes do distill.csv (leitura lado a lado)
    dist = pd.read_csv(REPO / "results" / "distillation" / "distill.csv")
    dist = dist[(dist.teacher == "tabpfn") & (dist.target == "soft")]
    for _, r in dist.iterrows():
        if r.dataset not in DATASETS:
            continue
        key = (r.dataset, "oof", r.student, int(r.seed))
        if key in done:
            continue
        append({"dataset": r.dataset, "regime": "oof", "student": r.student,
                "seed": int(r.seed), "rmse": r.rmse,
                "crps": r.crps if pd.notna(r.crps) else "",
                "picp80": r.picp80 if pd.notna(r.picp80) else "",
                "fit_time_s": r.fit_time_s})
        done.add(key)
    print("linhas OOF copiadas do distill.csv", flush=True)

    for ds in DATASETS:
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        params = load_xgb_params(ds)
        ins = insample_targets(ds, Xp, y_pool)
        assert len(ins) == len(y_pool)
        print(f"=== {ds}: alvos in-sample prontos ===", flush=True)

        for seed in SEEDS:
            for student, is_quant, fit_fn in (
                ("xgb_point", False,
                 lambda: fit_xgb_point(Xp, ins["mean"].to_numpy(), params, seed)),
                ("xgb_quant", True,
                 lambda: fit_xgb_quant_soft(Xp, ins[qcols].to_numpy(), params, seed)),
            ):
                key = (ds, "insample", student, seed)
                if key in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn()
                ft = time.perf_counter() - t0
                rmse, crps, picp = eval_student(model, Xt, y_test, is_quant)
                append({"dataset": ds, "regime": "insample", "student": student,
                        "seed": seed, "rmse": rmse, "crps": crps,
                        "picp80": picp, "fit_time_s": round(ft, 2)})
                print(f"  [insample/{student} seed={seed}] rmse={rmse} "
                      f"crps={crps or '-'}", flush=True)

    print("\nABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
