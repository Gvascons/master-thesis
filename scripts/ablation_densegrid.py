#!/usr/bin/env python3
"""A-GRID (pré-registrado em docs/desenho-melhorias-finais.md,
14/10/2026 23h30): estabilidade do estimador de CRPS sob grade densa.

4 conjuntos (incl. fifa, cauda pesada). Professor TabPFN re-predito no
hold-out em τ=0,01..0,99 (99 níveis). Alunos quant hard/soft (3
sementes) avaliados na grade densa por interpolação linear monótona da
função quantil (método declarado no pré-registro).

H-G1: CRPS99/CRPS19 aproximadamente constante entre sistemas
(variação < 3 p.p. por conjunto), preservando ordenações/retenções.

Saída: results/distillation/ablation_densegrid.csv
Uso: uv run python scripts/ablation_densegrid.py
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
from distill import (OOF_DIR, QUANTILES, fit_teacher, fit_xgb_quant_hard,
                     fit_xgb_quant_soft, free_teacher, make_teacher,
                     ordinal_encode, pinball, prep_pool, sort_quantiles)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

CSV_PATH = REPO / "results" / "distillation" / "ablation_densegrid.csv"
DATASETS = ["california_housing", "cpu_activity", "wine_quality", "fifa"]
SEEDS = [0, 1, 2]
DENSE = [round(t, 2) for t in np.arange(0.01, 0.995, 0.01)]  # 99 níveis
FIELDS = ["dataset", "system", "seed", "crps19", "crps99", "ratio"]


def crps_grid(y, Q, taus):
    return float(2.0 * np.mean([pinball(y, Q[:, k], t)
                                for k, t in enumerate(taus)]))


def interp_rows(Q19, taus_from, taus_to):
    """Extensão canônica da função quantil do aluno: interpolação linear
    monótona dos 19 nós; fora de [0,05, 0,95] clampa nos extremos."""
    out = np.empty((Q19.shape[0], len(taus_to)))
    for i in range(Q19.shape[0]):
        out[i] = np.interp(taus_to, taus_from, Q19[i])
    return out


def already_done():
    if not CSV_PATH.exists():
        return set()
    d = pd.read_csv(CSV_PATH)
    return {(r.dataset, r.system, int(r.seed)) for r in d.itertuples()}


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
        print(f"=== {ds} ===", flush=True)

        # professor: grades 19 e 99 nativas (bar distribution na GPU)
        if (ds, "teacher", 0) not in done:
            t0 = time.perf_counter()
            teacher = make_teacher("auto")
            fit_teacher(teacher, Xp, y_pool_a)
            Q19s, Q99s = [], []
            for i in range(0, len(Xt), 2000):
                xb = Xt[i:i + 2000]
                q19 = teacher.predict(xb, output_type="quantiles",
                                      quantiles=QUANTILES)
                q99 = teacher.predict(xb, output_type="quantiles",
                                      quantiles=DENSE)
                Q19s.append(np.column_stack([np.asarray(q) for q in q19]))
                Q99s.append(np.column_stack([np.asarray(q) for q in q99]))
            Q19 = sort_quantiles(np.vstack(Q19s))
            Q99 = sort_quantiles(np.vstack(Q99s))
            free_teacher(teacher)
            c19 = crps_grid(y_test_a, Q19, QUANTILES)
            c99 = crps_grid(y_test_a, Q99, DENSE)
            append_row({"dataset": ds, "system": "teacher", "seed": 0,
                        "crps19": round(c19, 6), "crps99": round(c99, 6),
                        "ratio": round(c99 / c19, 6)})
            print(f"  [teacher] crps19={c19:.4f} crps99={c99:.4f} "
                  f"ratio={c99/c19:.4f} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

        for seed in SEEDS:
            jobs = {
                "hard": lambda: fit_xgb_quant_hard(Xp, y_pool_a, params, seed),
                "soft": lambda: fit_xgb_quant_soft(
                    Xp, oof[qcols].to_numpy(), params, seed),
            }
            for system, fit_fn in jobs.items():
                if (ds, system, seed) in done:
                    continue
                set_seed(seed)
                model = fit_fn()
                Q19 = sort_quantiles(np.asarray(model.predict(Xt)))
                Q99 = interp_rows(Q19, QUANTILES, DENSE)
                c19 = crps_grid(y_test_a, Q19, QUANTILES)
                c99 = crps_grid(y_test_a, Q99, DENSE)
                append_row({"dataset": ds, "system": system, "seed": seed,
                            "crps19": round(c19, 6), "crps99": round(c99, 6),
                            "ratio": round(c99 / c19, 6)})
                print(f"  [{system} seed={seed}] ratio={c99/c19:.4f}",
                      flush=True)
    print("\nDENSE GRID DONE", flush=True)


if __name__ == "__main__":
    main()
