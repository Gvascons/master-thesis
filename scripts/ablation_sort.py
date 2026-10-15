#!/usr/bin/env python3
"""A-SORT (pré-registrado em docs/desenho-melhorias-finais.md,
14/10/2026): isola o papel do reparo de cruzamento de quantis por
ordenação pós-hoc, afirmado no texto como "regularizador de calibração"
mas nunca medido separadamente.

Para cada dataset×semente, treina os alunos quant hard/soft (protocolo
idêntico ao da AI-2) e computa CRPS/PICP com e sem sort, mais a fração
de linhas de teste com algum cruzamento.

H-S1: o sort melhora (ou não piora) CRPS/PICP em ambos os alvos.
Resumível. Saída: results/distillation/ablation_sort.csv
Uso: uv run python scripts/ablation_sort.py [--datasets a,b]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from decision_utility import CORE, EXT, load_params
from distill import (OOF_DIR, QUANTILES, crps_from_quantiles,
                     fit_xgb_quant_hard, fit_xgb_quant_soft,
                     interval_metrics, ordinal_encode, prep_pool,
                     sort_quantiles)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

CSV_PATH = REPO / "results" / "distillation" / "ablation_sort.csv"
SEEDS = [0, 1, 2]
FIELDS = ["dataset", "target", "seed", "crossing_frac",
          "crps_raw", "crps_sorted", "picp80_raw", "picp80_sorted",
          "picp90_raw", "picp90_sorted", "fit_time_s"]


def already_done():
    if not CSV_PATH.exists():
        return set()
    d = pd.read_csv(CSV_PATH)
    return {(r.dataset, r.target, int(r.seed)) for r in d.itertuples()}


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(CORE + EXT))
    args = ap.parse_args()
    exp_cfg = load_experiment_config()
    done = already_done()

    for ds in args.datasets.split(","):
        oof_path = OOF_DIR / f"tabpfn_{ds}.parquet"
        if not oof_path.exists():
            continue
        oof = pd.read_parquet(oof_path)
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        y_pool_a = np.asarray(y_pool, dtype=float)
        y_test_a = np.asarray(y_test, dtype=float)
        params = load_params(ds)
        qcols = [f"q{int(t*100):02d}" for t in QUANTILES]
        print(f"=== {ds} ===", flush=True)
        for seed in SEEDS:
            jobs = {
                "hard": lambda: fit_xgb_quant_hard(Xp, y_pool_a, params, seed),
                "soft": lambda: fit_xgb_quant_soft(
                    Xp, oof[qcols].to_numpy(), params, seed),
            }
            for target, fit_fn in jobs.items():
                if (ds, target, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn()
                fit_s = time.perf_counter() - t0
                Q_raw = np.asarray(model.predict(Xt))
                Q_sorted = sort_quantiles(Q_raw)
                crossing = float(np.mean(np.any(np.diff(Q_raw, axis=1) < 0,
                                                axis=1)))
                im_r = interval_metrics(y_test_a, Q_raw)
                im_s = interval_metrics(y_test_a, Q_sorted)
                row = {
                    "dataset": ds, "target": target, "seed": seed,
                    "crossing_frac": round(crossing, 6),
                    "crps_raw": round(crps_from_quantiles(y_test_a, Q_raw), 6),
                    "crps_sorted": round(
                        crps_from_quantiles(y_test_a, Q_sorted), 6),
                    "picp80_raw": round(im_r["picp80"], 6),
                    "picp80_sorted": round(im_s["picp80"], 6),
                    "picp90_raw": round(im_r["picp90"], 6),
                    "picp90_sorted": round(im_s["picp90"], 6),
                    "fit_time_s": round(fit_s, 2),
                }
                append_row(row)
                print(f"  [{target} seed={seed}] cross={crossing:.3f} "
                      f"crps {row['crps_raw']:.4f}->{row['crps_sorted']:.4f}",
                      flush=True)
    print("\nABLATION SORT DONE", flush=True)


if __name__ == "__main__":
    main()
