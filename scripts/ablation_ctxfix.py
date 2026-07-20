#!/usr/bin/env python3
"""Fixed-context ablation (AI-2 phase 5): deconfound leakage vs context size.

Phase 4a compared OOF (teacher sees 80% context) with in-sample (teacher
sees 100%): the two regimes differ in BOTH leakage and context size. This
cell adds the missing condition — **in-sample labels from an 80% context**:

  fold-0 teacher (fit on train-0 = 80% of the pool) labels train-0's own
  rows; fold-1 teacher labels the remaining 20% (holdout-0 rows, which are
  in-sample for fold-1). Two fits give every row an in-sample label w.r.t.
  an 80% context — identical context size to OOF, differing only in whether
  the teacher saw the labeled row.

Three-way comparison (regime column in ablation_insample.csv):
  oof             80% context, leak-free   (cached)
  insample_ctx80  80% context, leaky       (this script)
  insample        100% context, leaky      (phase 4a)

Same 4 datasets (pool <= 50K), same students, 3 seeds. Resumable.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/ablation_ctxfix.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ablation_insample import (DATASETS, SEEDS, already_done, append,
                               eval_student)
from distill import (QUANTILES, OOF_DIR, fit_teacher, fit_xgb_point,
                     fit_xgb_quant_soft, free_teacher, load_xgb_params,
                     make_teacher, ordinal_encode, prep_pool, teacher_predict)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

SEED = 42


def insample_ctx80_targets(ds, Xp, y_pool, device="auto"):
    path = OOF_DIR / f"tabpfn_{ds}_insample80.parquet"
    if path.exists():
        return pd.read_parquet(path)
    n = len(y_pool)
    mean = np.full(n, np.nan)
    Q = np.full((n, len(QUANTILES)), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(kf.split(Xp))
    # fold-0 teacher labels its own train rows; fold-1 teacher labels the rest
    (tr0, ho0), (tr1, _) = folds[0], folds[1]
    for tr_idx, label_idx in ((tr0, tr0), (tr1, ho0)):
        model = make_teacher(device)
        fit_teacher(model, Xp[tr_idx], y_pool[tr_idx])
        m, q = teacher_predict(model, Xp[label_idx])
        mean[label_idx] = m
        Q[label_idx] = q
        free_teacher(model)
    assert not np.isnan(mean).any()
    df = pd.DataFrame(Q, columns=[f"q{int(t*100):02d}" for t in QUANTILES])
    df.insert(0, "mean", mean)
    df.insert(0, "y", y_pool)
    df.to_parquet(path)
    return df


def main():
    exp_cfg = load_experiment_config()
    done = already_done()
    qcols = [f"q{int(t*100):02d}" for t in QUANTILES]

    for ds in DATASETS:
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        params = load_xgb_params(ds)
        tgt = insample_ctx80_targets(ds, Xp, y_pool)
        print(f"=== {ds}: alvos insample_ctx80 prontos ===", flush=True)

        for seed in SEEDS:
            for student, is_quant, fit_fn in (
                ("xgb_point", False,
                 lambda: fit_xgb_point(Xp, tgt["mean"].to_numpy(), params, seed)),
                ("xgb_quant", True,
                 lambda: fit_xgb_quant_soft(Xp, tgt[qcols].to_numpy(), params, seed)),
            ):
                key = (ds, "insample_ctx80", student, seed)
                if key in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn()
                ft = time.perf_counter() - t0
                rmse, crps, picp = eval_student(model, Xt, y_test, is_quant)
                append({"dataset": ds, "regime": "insample_ctx80",
                        "student": student, "seed": seed, "rmse": rmse,
                        "crps": crps, "picp80": picp,
                        "fit_time_s": round(ft, 2)})
                print(f"  [ctx80/{student} seed={seed}] rmse={rmse} "
                      f"crps={crps or '-'}", flush=True)

    print("\nCTXFIX DONE", flush=True)


if __name__ == "__main__":
    main()
