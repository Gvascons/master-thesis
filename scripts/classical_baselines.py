#!/usr/bin/env python3
"""E1 (fase final, pré-registrado em docs/desenho-melhorias-finais.md):
baselines distribucionais clássicos — NGBoost (Normal) e Quantile Random
Forest — nos mesmos 20 datasets/splits da AI-2.

Protocolo idêntico ao da destilação: hold-out 20% seed 42, caps do
benchmark, 3 sementes; CRPS = 2x média da pinball na MESMA grade de 19
quantis (comparabilidade interna); PICP80; RMSE (ponto = mediana prevista);
latência µs/linha (mediana de 5 passadas). Hiperparâmetros default
documentados (sem tuning — desvio declarado no pré-registro).

Resumível: pula (dataset, model, seed) já presentes no CSV.
Saída: results/distillation/classical_baselines.csv
Uso: uv run python scripts/classical_baselines.py [--datasets a,b,c]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from distill import (QUANTILES, crps_from_quantiles, interval_metrics,
                     ordinal_encode, prep_pool, sort_quantiles)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

CSV_PATH = REPO / "results" / "distillation" / "classical_baselines.csv"
CORE = ["wine_quality", "california_housing", "superconduct", "kin8nm",
        "year_prediction"]
EXT = ["abalone", "cpu_activity", "diamonds_real", "grid_stability",
       "miami_housing", "fifa", "kings_county", "health_insurance",
       "physiochemical_protein", "video_transcoding", "space_ga",
       "pumadyn32nh", "fps_benchmark", "cps88wages", "sarcos"]
SEEDS = [0, 1, 2]
FIELDS = ["dataset", "model", "seed", "n_pool", "n_test", "rmse", "crps",
          "picp80", "picp90", "fit_time_s", "us_per_row"]


def already_done():
    if not CSV_PATH.exists():
        return set()
    import pandas as pd
    d = pd.read_csv(CSV_PATH)
    return {(r.dataset, r.model, int(r.seed)) for r in d.itertuples()}


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def fit_ngboost(Xp, yp, seed):
    from ngboost import NGBRegressor
    # validacao interna de 10% do pool para early stopping (espelha o MLP
    # student); distribuicao Normal default
    from sklearn.model_selection import train_test_split
    Xtr, Xval, ytr, yval = train_test_split(Xp, yp, test_size=0.1,
                                            random_state=seed)
    m = NGBRegressor(random_state=seed, verbose=False,
                     early_stopping_rounds=50)
    m.fit(Xtr, ytr, X_val=Xval, Y_val=yval)
    return m


def quantiles_ngboost(model, Xt):
    dist = model.pred_dist(Xt)
    # Normal: quantis analiticos loc + sigma*z_tau
    from scipy.stats import norm
    loc = dist.params["loc"]
    scale = dist.params["scale"]
    Q = np.stack([loc + scale * norm.ppf(t) for t in QUANTILES], axis=1)
    return Q


def fit_qrf(Xp, yp, seed):
    from quantile_forest import RandomForestQuantileRegressor
    m = RandomForestQuantileRegressor(n_estimators=100, random_state=seed,
                                      n_jobs=-1)
    m.fit(Xp, yp)
    return m


def quantiles_qrf(model, Xt):
    return np.asarray(model.predict(Xt, quantiles=list(QUANTILES)))


def timed_quantiles(fn, model, Xt, repeats=5):
    best = []
    Q = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        Q = fn(model, Xt)
        best.append(time.perf_counter() - t0)
    return Q, float(np.median(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(CORE + EXT))
    args = ap.parse_args()
    exp_cfg = load_experiment_config()
    done = already_done()

    for ds in args.datasets.split(","):
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        y_pool_a = np.asarray(y_pool, dtype=float)
        y_test_a = np.asarray(y_test, dtype=float)
        print(f"=== {ds}: pool={len(y_pool_a)} test={len(y_test_a)} ===",
              flush=True)
        for model_name, fit_fn, q_fn in (
                ("ngboost", fit_ngboost, quantiles_ngboost),
                ("qrf", fit_qrf, quantiles_qrf)):
            for seed in SEEDS:
                if (ds, model_name, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn(Xp, y_pool_a, seed)
                fit_s = time.perf_counter() - t0
                Q, pred_s = timed_quantiles(q_fn, model, Xt)
                Q = sort_quantiles(np.asarray(Q))
                point = Q[:, QUANTILES.index(0.50)]
                im = interval_metrics(y_test_a, Q)
                row = {
                    "dataset": ds, "model": model_name, "seed": seed,
                    "n_pool": len(y_pool_a), "n_test": len(y_test_a),
                    "rmse": round(float(np.sqrt(np.mean(
                        (y_test_a - point) ** 2))), 6),
                    "crps": round(crps_from_quantiles(y_test_a, Q), 6),
                    "picp80": round(im["picp80"], 6),
                    "picp90": round(im["picp90"], 6),
                    "fit_time_s": round(fit_s, 2),
                    "us_per_row": round(pred_s / len(y_test_a) * 1e6, 3),
                }
                append_row(row)
                print(f"  [{model_name} seed={seed}] rmse={row['rmse']:.4f} "
                      f"crps={row['crps']:.4f} picp80={row['picp80']:.3f} "
                      f"fit={fit_s:.1f}s lat={row['us_per_row']}us/row",
                      flush=True)
    print("\nBASELINES DONE", flush=True)


if __name__ == "__main__":
    main()
