#!/usr/bin/env python3
"""Log-target ablation (AI-2 phase 6): is heavy-tail the failure moderator?

The N=15 picture: three of four distillation failures (fifa,
diamonds_real, health_insurance) have heavy-tailed/price-scale targets
DESPITE a positive teacher edge. Hypothesis: the quantile-curve transfer
breaks on raw heavy-tailed scales, and taming the target recovers it.

Cell: rerun teacher OOF + anchor + quantile students (hard/soft, 3 seeds)
with y' = log1p(y), evaluating CRPS/retention IN LOG SPACE (the
within-dataset hard-vs-soft comparison is scale-consistent). Datasets:
the 3 failures + 2 price-scale CONTROLS with positive raw retention
(kings_county +0.21, miami_housing +0.02) to check log does not break
what worked.

Output: results/distillation/ablation_logtarget.csv
(cells: teacher / student_quant_hard / student_quant_soft). Resumable.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/ablation_logtarget.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from distill import (QUANTILES, OOF_DIR, crps_from_quantiles, fit_teacher,
                     fit_xgb_quant_hard, fit_xgb_quant_soft, free_teacher,
                     interval_metrics, make_teacher, ordinal_encode,
                     prep_pool, sort_quantiles, teacher_predict)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

OUT = REPO / "results" / "distillation" / "ablation_logtarget.csv"
EXT_DIR = REPO / "results" / "distillation" / "extension"
FAILURES = ["fifa", "diamonds_real", "health_insurance"]
CONTROLS = ["kings_county", "miami_housing"]
SEEDS = [0, 1, 2]
HEADER = ["dataset", "role", "cell", "seed", "rmse_log", "crps_log",
          "picp80", "time_s"]


def already_done():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["cell"], int(r["seed"])))
    return done


def append(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def load_ext_params(ds):
    import json
    return json.loads((EXT_DIR / f"xgboost_{ds}.json").read_text())["best_params"]


def log_oof(ds, Xp, ylog):
    path = OOF_DIR / f"tabpfn_{ds}_logtarget.parquet"
    if path.exists():
        return pd.read_parquet(path)
    n = len(ylog)
    mean = np.full(n, np.nan)
    Q = np.full((n, len(QUANTILES)), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr, ho in kf.split(Xp):
        model = make_teacher("auto")
        fit_teacher(model, Xp[tr], ylog[tr])
        m, q = teacher_predict(model, Xp[ho])
        mean[ho], Q[ho] = m, q
        free_teacher(model)
    assert not np.isnan(mean).any()
    df = pd.DataFrame(Q, columns=[f"q{int(t*100):02d}" for t in QUANTILES])
    df.insert(0, "mean", mean)
    df.insert(0, "y", ylog)
    df.to_parquet(path)
    return df


def main():
    exp_cfg = load_experiment_config()
    done = already_done()
    qcols = [f"q{int(t*100):02d}" for t in QUANTILES]

    for ds in FAILURES + CONTROLS:
        role = "failure" if ds in FAILURES else "control"
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        assert (y_pool >= 0).all() and (y_test >= 0).all(), f"{ds}: alvo negativo"
        Xp, Xt = ordinal_encode(X_pool, X_test)
        ylog, ylog_test = np.log1p(y_pool), np.log1p(y_test)
        params = load_ext_params(ds)
        oof = log_oof(ds, Xp, ylog)
        print(f"=== {ds} ({role}): OOF log pronto ===", flush=True)

        # ancora do teacher em espaco log
        if (ds, "teacher", 0) not in done:
            model = make_teacher("auto")
            t0 = time.perf_counter()
            fit_teacher(model, Xp, ylog)
            m, Q = teacher_predict(model, Xt)
            free_teacher(model)
            append({"dataset": ds, "role": role, "cell": "teacher", "seed": 0,
                    "rmse_log": round(float(np.sqrt(np.mean((ylog_test - m) ** 2))), 6),
                    "crps_log": round(crps_from_quantiles(ylog_test, Q), 6),
                    "picp80": round(interval_metrics(ylog_test, Q)["picp80"], 4),
                    "time_s": round(time.perf_counter() - t0, 1)})

        for seed in SEEDS:
            for cell, fit_fn in (
                ("student_quant_hard", lambda: fit_xgb_quant_hard(Xp, ylog, params, seed)),
                ("student_quant_soft", lambda: fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(), params, seed)),
            ):
                if (ds, cell, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn()
                pred = model.predict(Xt)
                Q = sort_quantiles(np.asarray(pred))
                point = Q[:, QUANTILES.index(0.50)]
                append({"dataset": ds, "role": role, "cell": cell, "seed": seed,
                        "rmse_log": round(float(np.sqrt(np.mean((ylog_test - point) ** 2))), 6),
                        "crps_log": round(crps_from_quantiles(ylog_test, Q), 6),
                        "picp80": round(interval_metrics(ylog_test, Q)["picp80"], 4),
                        "time_s": round(time.perf_counter() - t0, 2)})
                print(f"  [{cell} seed={seed}] ok", flush=True)

    print("\nLOGTARGET DONE", flush=True)


if __name__ == "__main__":
    main()
