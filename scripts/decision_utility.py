#!/usr/bin/env python3
"""E3 (pré-registrado em docs/desenho-melhorias-finais.md, 14/10/2026):
utilidade de decisão sob custos assimétricos — custo realizado do
jornaleiro (pinball, unidades nativas) por nível τ, no hold-out.

Modelos por dataset×semente (hiperparâmetros herdados do benchmark,
idêntico ao protocolo da AI-2):
  quant_hard  XGBoost multi-quantil nos rótulos verdadeiros (controle)
  quant_soft  XGBoost multi-saída nos 19 quantis OOF do professor TabPFN
  gauss       XGBoost pontual + sigma global de resíduos (validação 10%
              do pool; refit no pool completo); quantis = ponto + sigma·z

Resumível: pula (dataset, model, seed) já presentes no CSV.
Saída: results/distillation/decision_utility.csv
Uso: uv run python scripts/decision_utility.py [--datasets a,b,c]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from distill import (OOF_DIR, QUANTILES, fit_xgb_point, fit_xgb_quant_hard,
                     fit_xgb_quant_soft, ordinal_encode, pinball, prep_pool,
                     sort_quantiles)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

CSV_PATH = REPO / "results" / "distillation" / "decision_utility.csv"
EXT_DIR = REPO / "results" / "distillation" / "extension"
CORE = ["wine_quality", "california_housing", "superconduct", "kin8nm",
        "year_prediction"]
EXT = ["abalone", "cpu_activity", "diamonds_real", "grid_stability",
       "miami_housing", "fifa", "kings_county", "health_insurance",
       "physiochemical_protein", "video_transcoding", "space_ga",
       "pumadyn32nh", "fps_benchmark", "cps88wages", "sarcos"]
SEEDS = [0, 1, 2]
FIELDS = (["dataset", "model", "seed", "n_test"]
          + [f"cost_q{int(t * 100):02d}" for t in QUANTILES]
          + ["fit_time_s"])


def load_params(ds):
    for path in (REPO / "results" / "raw" / f"xgboost_{ds}.json",
                 EXT_DIR / f"xgboost_{ds}.json"):
        if path.exists():
            p = json.loads(path.read_text())["best_params"] or {}
            p.pop("cat_feature_indices", None)
            return p
    raise FileNotFoundError(f"params do xgboost nao encontrados p/ {ds}")


def already_done():
    if not CSV_PATH.exists():
        return set()
    d = pd.read_csv(CSV_PATH)
    return {(r.dataset, r.model, int(r.seed)) for r in d.itertuples()}


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def fit_gauss(Xp, yp, params, seed):
    """Ponto + resíduo gaussiano global: sigma estimado num split interno
    de 10% do pool (sem tocar o teste), modelo final refit no pool todo."""
    from scipy.stats import norm
    from sklearn.model_selection import train_test_split
    Xtr, Xval, ytr, yval = train_test_split(Xp, yp, test_size=0.1,
                                            random_state=seed)
    m_val = fit_xgb_point(Xtr, ytr, params, seed)
    sigma = float(np.std(yval - m_val.predict(Xval)))
    m = fit_xgb_point(Xp, yp, params, seed)
    z = np.array([norm.ppf(t) for t in QUANTILES])
    return m, sigma, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(CORE + EXT))
    args = ap.parse_args()
    exp_cfg = load_experiment_config()
    done = already_done()

    for ds in args.datasets.split(","):
        oof_path = OOF_DIR / f"tabpfn_{ds}.parquet"
        if not oof_path.exists():
            print(f"[skip] sem OOF para {ds}", flush=True)
            continue
        oof = pd.read_parquet(oof_path)
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        y_pool_a = np.asarray(y_pool, dtype=float)
        y_test_a = np.asarray(y_test, dtype=float)
        assert len(oof) == len(y_pool_a), "OOF desalinhado com o pool"
        params = load_params(ds)
        qcols = [f"q{int(t * 100):02d}" for t in QUANTILES]
        print(f"=== {ds}: pool={len(y_pool_a)} test={len(y_test_a)} ===",
              flush=True)

        for seed in SEEDS:
            jobs = {
                "quant_hard": lambda: fit_xgb_quant_hard(
                    Xp, y_pool_a, params, seed),
                "quant_soft": lambda: fit_xgb_quant_soft(
                    Xp, oof[qcols].to_numpy(), params, seed),
                "gauss": lambda: fit_gauss(Xp, y_pool_a, params, seed),
            }
            for model_name, fit_fn in jobs.items():
                if (ds, model_name, seed) in done:
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                fitted = fit_fn()
                fit_s = time.perf_counter() - t0
                if model_name == "gauss":
                    m, sigma, z = fitted
                    point = m.predict(Xt)
                    Q = point[:, None] + sigma * z[None, :]
                else:
                    Q = sort_quantiles(np.asarray(fitted.predict(Xt)))
                row = {"dataset": ds, "model": model_name, "seed": seed,
                       "n_test": len(y_test_a),
                       "fit_time_s": round(fit_s, 2)}
                for j, tau in enumerate(QUANTILES):
                    row[f"cost_q{int(tau * 100):02d}"] = round(
                        float(pinball(y_test_a, Q[:, j], tau)), 6)
                append_row(row)
                print(f"  [{model_name:10s} seed={seed}] "
                      f"c80={row['cost_q80']:.4f} c90={row['cost_q90']:.4f} "
                      f"c95={row['cost_q95']:.4f} fit={fit_s:.1f}s",
                      flush=True)
    print("\nDECISION UTILITY DONE", flush=True)


if __name__ == "__main__":
    main()
