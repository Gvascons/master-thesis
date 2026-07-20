#!/usr/bin/env python3
"""CTR23 extension pipeline (AI-2 phase 5): 10 verified regression datasets.

Per dataset (all stages resumable): download (version-pinned CTR23 upload)
-> tune XGBoost (benchmark protocol: 100 TPE trials, 3-fold inner CV) ->
teacher OOF labeling + hold-out anchor (TabPFN, quantile grid) -> students
(xgb point/quant x hard/soft, 3 seeds).

Artifacts are isolated from the 14-model benchmark:
  results/distillation/extension/xgboost_<ds>.json   tuned params
  results/distillation/oof/tabpfn_<ds>.parquet       OOF targets (shared dir)
  results/distillation/extension.csv                 all evaluation rows
    cell in {baseline, teacher, student_point_hard/soft,
             student_quant_hard/soft}

Dataset selection: OpenML-CTR23 (suite 353), verified 20/07/2026 — leakage
traps (brazilian_houses, wave_energy) and overlaps with the 5 core datasets
excluded by design. See the verification memo in the AI-2 report.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/extension_ctr23.py
"""
import csv
import json
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
                     fit_xgb_point, fit_xgb_quant_hard, fit_xgb_quant_soft,
                     free_teacher, interval_metrics, make_teacher,
                     ordinal_encode, prep_pool, sort_quantiles,
                     teacher_predict)
from src.data.download import download_dataset
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

EXT_DIR = REPO / "results" / "distillation" / "extension"
OUT = REPO / "results" / "distillation" / "extension.csv"
DATASETS = ["abalone", "cpu_activity", "diamonds_real", "grid_stability",
            "miami_housing", "fifa", "kings_county", "health_insurance",
            "physiochemical_protein", "video_transcoding"]
SEEDS = [0, 1, 2]
HEADER = ["dataset", "cell", "seed", "rmse", "crps", "picp80", "time_s"]


def already_done():
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["cell"], int(r["seed"])))
    return done


def append(row):
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    write_hdr = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def tuned_params(ds, X_pool, y_pool, info):
    """Benchmark-protocol XGBoost tuning, cached to the extension dir."""
    path = EXT_DIR / f"xgboost_{ds}.json"
    if path.exists():
        return json.loads(path.read_text())["best_params"]
    from src.tuning.tuner import tune_model
    t0 = time.perf_counter()
    res = tune_model("xgboost", X_pool, y_pool, info)
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "best_params": res["best_params"],
        "best_score": res["best_score"],
        "tuning_time_s": round(time.perf_counter() - t0, 1),
    }, indent=2))
    return res["best_params"]


def oof_targets(ds, Xp, y_pool, Xt, y_test, done):
    """Teacher OOF parquet + hold-out anchor row (cell='teacher')."""
    path = OOF_DIR / f"tabpfn_{ds}.parquet"
    if not path.exists():
        n = len(y_pool)
        mean = np.full(n, np.nan)
        Q = np.full((n, len(QUANTILES)), np.nan)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fi, (tr, ho) in enumerate(kf.split(Xp)):
            model = make_teacher("auto")
            fit_teacher(model, Xp[tr], y_pool[tr])
            m, q = teacher_predict(model, Xp[ho])
            mean[ho], Q[ho] = m, q
            free_teacher(model)
        assert not np.isnan(mean).any()
        df = pd.DataFrame(Q, columns=[f"q{int(t*100):02d}" for t in QUANTILES])
        df.insert(0, "mean", mean)
        df.insert(0, "y", y_pool)
        df.to_parquet(path)
    if (ds, "teacher", 0) not in done:
        model = make_teacher("auto")
        t0 = time.perf_counter()
        fit_teacher(model, Xp, y_pool)
        m, Q = teacher_predict(model, Xt)
        free_teacher(model)
        append({"dataset": ds, "cell": "teacher", "seed": 0,
                "rmse": round(float(np.sqrt(np.mean((y_test - m) ** 2))), 6),
                "crps": round(crps_from_quantiles(y_test, Q), 6),
                "picp80": round(interval_metrics(y_test, Q)["picp80"], 4),
                "time_s": round(time.perf_counter() - t0, 1)})
    return pd.read_parquet(path)


def eval_and_append(ds, cell, seed, model, Xt, y_test, is_quant, ft):
    pred = model.predict(Xt)
    if is_quant:
        Q = sort_quantiles(np.asarray(pred))
        point = Q[:, QUANTILES.index(0.50)]
        crps = round(crps_from_quantiles(y_test, Q), 6)
        picp = round(interval_metrics(y_test, Q)["picp80"], 4)
    else:
        point, crps, picp = pred, "", ""
    append({"dataset": ds, "cell": cell, "seed": seed,
            "rmse": round(float(np.sqrt(np.mean((y_test - point) ** 2))), 6),
            "crps": crps, "picp80": picp, "time_s": round(ft, 2)})


def main():
    exp_cfg = load_experiment_config()
    done = already_done()
    qcols = [f"q{int(t*100):02d}" for t in QUANTILES]

    for ds in DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        download_dataset(ds)
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        params = tuned_params(ds, X_pool, y_pool, info)
        print(f"  tuning OK ({len(params)} HPs)", flush=True)
        oof = oof_targets(ds, Xp, y_pool, Xt, y_test, done)
        print(f"  OOF/anchor OK (pool={len(y_pool)}, test={len(y_test)})", flush=True)

        for seed in SEEDS:
            cells = (
                ("student_point_hard", False, lambda: fit_xgb_point(Xp, y_pool, params, seed)),
                ("student_point_soft", False, lambda: fit_xgb_point(Xp, oof["mean"].to_numpy(), params, seed)),
                ("student_quant_hard", True, lambda: fit_xgb_quant_hard(Xp, y_pool, params, seed)),
                ("student_quant_soft", True, lambda: fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(), params, seed)),
            )
            for cell, is_quant, fit_fn in cells:
                if (ds, cell, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn()
                eval_and_append(ds, cell, seed, model, Xt, y_test, is_quant,
                                time.perf_counter() - t0)
                print(f"  [{cell} seed={seed}] ok", flush=True)

    print("\nEXTENSION DONE", flush=True)


if __name__ == "__main__":
    main()
