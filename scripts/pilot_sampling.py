#!/usr/bin/env python3
"""E6 — PILOTO (pré-registrado em docs/desenho-melhorias-finais.md,
14/10/2026 23h30): destilação por AMOSTRAGEM da distribuição do
professor, contra a destilação por curve-fitting (MSE multi-saída).

Mecanismo: K=8 pseudo-rótulos por linha do pool, amostrados da inversa
da função quantil OOF do professor (interpolação linear dos 19 nós;
u estratificado por linha: u_k=(k+U)/K; caudas clampadas em q05/q95 —
declarado). O aluno quantílico NATIVO (pinball multi-quantil) treina no
pool aumentado: regressão quantílica genuína sobre a distribuição do
professor, com perda própria.

H-X1 (exploratória): sampling >= MSE em CRPS nos conjuntos de vantagem
positiva. Piloto: 5 conjuntos, 3 sementes.
Saída: results/distillation/pilot_sampling.csv
Uso: uv run python scripts/pilot_sampling.py
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

from decision_utility import load_params
from distill import (OOF_DIR, QUANTILES, crps_from_quantiles,
                     fit_xgb_quant_hard, fit_xgb_quant_soft,
                     interval_metrics, ordinal_encode, prep_pool,
                     sort_quantiles)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

CSV_PATH = REPO / "results" / "distillation" / "pilot_sampling.csv"
DATASETS = ["california_housing", "cpu_activity", "cps88wages",
            "wine_quality", "fifa"]
SEEDS = [0, 1, 2]
K = 8
FIELDS = ["dataset", "variant", "seed", "crps", "picp80", "picp90",
          "fit_time_s"]


def sample_targets(Q_oof, seed):
    """K pseudo-rótulos por linha via inversa da função quantil
    (interp. linear dos 19 nós; u estratificado; caudas clampadas)."""
    rng = np.random.default_rng(seed)
    n = Q_oof.shape[0]
    u = (np.arange(K)[None, :] + rng.random((n, K))) / K  # (n, K)
    u = np.clip(u, QUANTILES[0], QUANTILES[-1])
    y_s = np.empty((n, K))
    for i in range(n):
        y_s[i] = np.interp(u[i], QUANTILES, Q_oof[i])
    return y_s


def already_done():
    if not CSV_PATH.exists():
        return set()
    d = pd.read_csv(CSV_PATH)
    return {(r.dataset, r.variant, int(r.seed)) for r in d.itertuples()}


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    exp_cfg = load_experiment_config()
    done = already_done()
    for ds in DATASETS:
        oof = pd.read_parquet(OOF_DIR / f"tabpfn_{ds}.parquet")
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        y_pool_a = np.asarray(y_pool, dtype=float)
        y_test_a = np.asarray(y_test, dtype=float)
        params = load_params(ds)
        qcols = [f"q{int(t*100):02d}" for t in QUANTILES]
        Q_oof = sort_quantiles(oof[qcols].to_numpy())
        print(f"=== {ds} (pool={len(y_pool_a)}) ===", flush=True)
        for seed in SEEDS:
            jobs = {
                "mse_soft": lambda: fit_xgb_quant_soft(
                    Xp, oof[qcols].to_numpy(), params, seed),
                "sampling_soft": lambda: fit_xgb_quant_hard(
                    np.repeat(Xp, K, axis=0),
                    sample_targets(Q_oof, seed).reshape(-1), params, seed),
                "hard": lambda: fit_xgb_quant_hard(Xp, y_pool_a, params,
                                                   seed),
            }
            for variant, fit_fn in jobs.items():
                if (ds, variant, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn()
                fit_s = time.perf_counter() - t0
                Q = sort_quantiles(np.asarray(model.predict(Xt)))
                im = interval_metrics(y_test_a, Q)
                row = {"dataset": ds, "variant": variant, "seed": seed,
                       "crps": round(crps_from_quantiles(y_test_a, Q), 6),
                       "picp80": round(im["picp80"], 6),
                       "picp90": round(im["picp90"], 6),
                       "fit_time_s": round(fit_s, 2)}
                append_row(row)
                print(f"  [{variant:13s} seed={seed}] crps={row['crps']:.4f} "
                      f"picp80={row['picp80']:.3f} fit={fit_s:.0f}s",
                      flush=True)
    print("\nPILOT SAMPLING DONE", flush=True)


if __name__ == "__main__":
    main()
