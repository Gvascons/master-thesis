#!/usr/bin/env python3
"""R-TUNE (pré-registrado em docs/desenho-melhorias-finais.md,
14/10/2026 23h30): teto de capacidade dos alunos — re-tuning por alvo.

Nos 3 conjuntos de maior vantagem genuína com retenção positiva
(california_housing, cpu_activity, cps88wages): Optuna 25 trials por
braço (hard: quantílico nativo em y; soft: MSE multi-saída nos quantis
OOF), validação interna 80/20 do pool, objetivo = pinball média na
grade de 19 (CRPS/2) com quantis ordenados. HPs vencedores refitados
no pool completo, 3 sementes, avaliação no hold-out.

H-T1: retenção com re-tuning >= retenção com HPs herdados.
Saída: results/distillation/retune_students.csv
Uso: uv run python scripts/retune_students.py
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
                     interval_metrics, ordinal_encode, pinball, prep_pool,
                     sort_quantiles)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

CSV_PATH = REPO / "results" / "distillation" / "retune_students.csv"
DATASETS = ["california_housing", "cpu_activity", "cps88wages"]
SEEDS = [0, 1, 2]
N_TRIALS = 25
FIELDS = ["dataset", "target", "hp_source", "seed", "crps", "picp80",
          "fit_time_s", "best_params"]


def mean_pinball(y, Q):
    return float(np.mean([pinball(y, Q[:, k], t)
                          for k, t in enumerate(QUANTILES)]))


def search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3,
                                             log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2,
                                                 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def tune(fit_variant, Xtr, tgt_tr, Xval, y_val):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = search_space(trial)
        model = fit_variant(Xtr, tgt_tr, params, 0)
        Q = sort_quantiles(np.asarray(model.predict(Xval)))
        return mean_pinball(y_val, Q)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def already_done():
    if not CSV_PATH.exists():
        return set()
    d = pd.read_csv(CSV_PATH)
    return {(r.dataset, r.target, r.hp_source, int(r.seed))
            for r in d.itertuples()}


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    from sklearn.model_selection import train_test_split
    exp_cfg = load_experiment_config()
    done = already_done()
    for ds in DATASETS:
        oof = pd.read_parquet(OOF_DIR / f"tabpfn_{ds}.parquet")
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        y_pool_a = np.asarray(y_pool, dtype=float)
        y_test_a = np.asarray(y_test, dtype=float)
        inherited = load_params(ds)
        qcols = [f"q{int(t*100):02d}" for t in QUANTILES]
        Q_oof = oof[qcols].to_numpy()
        idx = np.arange(len(y_pool_a))
        itr, ival = train_test_split(idx, test_size=0.2, random_state=0)
        print(f"=== {ds} ===", flush=True)

        arms = {
            "hard": (fit_xgb_quant_hard, y_pool_a),
            "soft": (fit_xgb_quant_soft, Q_oof),
        }
        for target, (fit_fn, tgt) in arms.items():
            tgt_tr = tgt[itr]
            best = None
            for hp_source in ("inherited", "retuned"):
                if all((ds, target, hp_source, s) in done for s in SEEDS):
                    continue
                if hp_source == "retuned" and best is None:
                    t0 = time.perf_counter()
                    best = tune(fit_fn, Xp[itr], tgt_tr, Xp[ival],
                                y_pool_a[ival])
                    print(f"  [{target}] tuning ok "
                          f"({time.perf_counter()-t0:.0f}s): {best}",
                          flush=True)
                params = inherited if hp_source == "inherited" else best
                for seed in SEEDS:
                    if (ds, target, hp_source, seed) in done:
                        continue
                    set_seed(seed)
                    t0 = time.perf_counter()
                    model = fit_fn(Xp, tgt, params, seed)
                    fit_s = time.perf_counter() - t0
                    Q = sort_quantiles(np.asarray(model.predict(Xt)))
                    im = interval_metrics(y_test_a, Q)
                    append_row({
                        "dataset": ds, "target": target,
                        "hp_source": hp_source, "seed": seed,
                        "crps": round(crps_from_quantiles(y_test_a, Q), 6),
                        "picp80": round(im["picp80"], 6),
                        "fit_time_s": round(fit_s, 2),
                        "best_params": str(params) if hp_source == "retuned"
                        else ""})
                    print(f"  [{target}/{hp_source} seed={seed}] ok "
                          f"({fit_s:.0f}s)", flush=True)
    print("\nRETUNE DONE", flush=True)


if __name__ == "__main__":
    main()
