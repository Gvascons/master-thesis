#!/usr/bin/env python3
"""E4 — PILOTO exploratório (pré-registrado em
docs/desenho-melhorias-finais.md, 14/10/2026; piloto ≠ resultado):
a vantagem do aluno destilado sobrevive à calibração conformal barata?

Split-conformal por quantil (estilo CQR, uma correção escalar por nível
τ): 10% do pool vira conjunto de calibração; os alunos quant hard/soft
treinam nos 90% restantes; cada quantil τ_j é deslocado pela correção
empírica que torna a cobertura unilateral exata em calibração.
Compara CRPS/PICP de {hard, soft} × {cru, conformal} no hold-out.

Piloto: 5 conjuntos (3 sucessos, 2 falhas), 3 sementes.
Saída: results/distillation/pilot_conformal.csv
Uso: uv run python scripts/pilot_conformal.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

CSV_PATH = REPO / "results" / "distillation" / "pilot_conformal.csv"
DATASETS = ["california_housing", "cpu_activity", "cps88wages",
            "wine_quality", "fifa"]
SEEDS = [0, 1, 2]
FIELDS = ["dataset", "target", "seed", "variant", "crps", "picp80",
          "picp90", "fit_time_s"]


def already_done():
    if not CSV_PATH.exists():
        return set()
    d = pd.read_csv(CSV_PATH)
    return {(r.dataset, r.target, int(r.seed), r.variant)
            for r in d.itertuples()}


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def conformal_shift(Q_cal, y_cal):
    """Correção escalar por nível: delta_j = quantil-τ_j dos resíduos
    (y - q_j) em calibração; q_j_conf(x) = q_j(x) + delta_j torna a
    cobertura unilateral exata em calibração (split-conformal)."""
    deltas = np.empty(len(QUANTILES))
    for j, tau in enumerate(QUANTILES):
        deltas[j] = np.quantile(y_cal - Q_cal[:, j], tau)
    return deltas


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
        Q_oof = oof[qcols].to_numpy()
        print(f"=== {ds} ===", flush=True)
        for seed in SEEDS:
            idx = np.arange(len(y_pool_a))
            itr, ical = train_test_split(idx, test_size=0.1,
                                         random_state=seed)
            jobs = {
                "hard": lambda: fit_xgb_quant_hard(
                    Xp[itr], y_pool_a[itr], params, seed),
                "soft": lambda: fit_xgb_quant_soft(
                    Xp[itr], Q_oof[itr], params, seed),
            }
            for target, fit_fn in jobs.items():
                # resume só quando AMBAS as variantes já estão gravadas
                if ((ds, target, seed, "conformal") in done
                        and (ds, target, seed, "raw") in done):
                    continue
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit_fn()
                fit_s = time.perf_counter() - t0
                Q_cal = sort_quantiles(np.asarray(model.predict(Xp[ical])))
                Q_t = sort_quantiles(np.asarray(model.predict(Xt)))
                deltas = conformal_shift(Q_cal, y_pool_a[ical])
                Q_conf = sort_quantiles(Q_t + deltas[None, :])
                for variant, Q in (("raw", Q_t), ("conformal", Q_conf)):
                    if (ds, target, seed, variant) in done:
                        continue
                    im = interval_metrics(y_test_a, Q)
                    row = {"dataset": ds, "target": target, "seed": seed,
                           "variant": variant,
                           "crps": round(crps_from_quantiles(y_test_a, Q), 6),
                           "picp80": round(im["picp80"], 6),
                           "picp90": round(im["picp90"], 6),
                           "fit_time_s": round(fit_s, 2)}
                    append_row(row)
                print(f"  [{target} seed={seed}] ok ({fit_s:.1f}s)",
                      flush=True)
    print("\nPILOT CONFORMAL DONE", flush=True)


if __name__ == "__main__":
    main()
