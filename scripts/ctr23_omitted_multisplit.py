"""Sensibilidade dos 4 CTR23 omitidos a MÚLTIPLOS splits (extensão do
adendo de 23/11/2026; o limite "um split só" declarado na PROVENANCE).

Repete o protocolo completo (tuning Optuna, professor OOF K=5, alunos
quantílicos nas sementes {0,1,2}) nos 4 conjuntos omitidos para splits de
hold-out adicionais. Resumível por (dataset, split, cell, seed).

Uso: uv run python scripts/ctr23_omitted_multisplit.py [splits...]
     (default: 43 44 45 46 — o split 42 é o do adendo)

Saída: results/distillation/ctr23_omitidos/multisplit.csv
       + resumo do placar por split.
"""
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

DIR = REPO / "results" / "distillation" / "ctr23_omitidos"

import openml

openml.config.set_root_cache_directory(str(DIR / "openml_cache"))

from distill import (QUANTILES, crps_from_quantiles, fit_teacher,
                     fit_xgb_quant_hard, fit_xgb_quant_soft, free_teacher,
                     interval_metrics, make_teacher, ordinal_encode,
                     sort_quantiles, teacher_predict)
from src.data.registry import DatasetInfo, get_holdout_split
from src.tuning.tuner import tune_model
from src.utils.reproducibility import set_seed

DS = {"naval_propulsion_plant": 44969, "superconductivity": 44964,
      "california_ctr23": 44977, "white_wine": 44971}
qcols = [f"q{int(t*100):02d}" for t in QUANTILES]
CSV = DIR / "multisplit.csv"


def load(name, did):
    d = openml.datasets.get_dataset(did, download_data=True,
                                    download_qualities=False,
                                    download_features_meta_data=True)
    X, y, _, names = d.get_data(target=d.default_target_attribute)
    X = pd.DataFrame(X, columns=names)
    for f in d.features.values():
        if f.data_type == "nominal" and f.name in X.columns:
            X[f.name] = X[f.name].astype("category")
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    info = DatasetInfo(name=name, task_type="regression", n_classes=None,
                       feature_types="mixed" if cat else "numerical",
                       cat_columns=cat,
                       num_columns=X.select_dtypes(include=["number"]).columns.tolist())
    return X, np.asarray(y, dtype=np.float64), info


def done():
    if not CSV.exists():
        return set()
    d = pd.read_csv(CSV)
    return set(zip(d.dataset, d.split, d.cell, d.seed))


def append(row):
    pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(),
                               index=False)


def main():
    splits = [int(x) for x in sys.argv[1:]] or [43, 44, 45, 46]
    for split in splits:
        for ds, did in DS.items():
            print(f"\n=== {ds} split={split} ===", flush=True)
            X, y, info = load(ds, did)
            Xp_df, y_pool, Xt_df, y_test = get_holdout_split(
                X, y, info, seed=split, test_size=0.2)
            Xp, Xt = ordinal_encode(Xp_df, Xt_df)
            pj = DIR / f"xgboost_{ds}_s{split}.json"
            if pj.exists():
                params = json.loads(pj.read_text())["best_params"]
            else:
                t0 = time.perf_counter()
                res = tune_model("xgboost", Xp_df, y_pool, info, seed=split)
                params = res["best_params"]
                params.pop("cat_feature_indices", None)
                pj.write_text(json.dumps(
                    {"best_params": params,
                     "tuning_time_s": round(time.perf_counter() - t0, 1)},
                    indent=2))
            print("  tuning ok", flush=True)
            d = done()
            po = DIR / f"oof_tabpfn_{ds}_s{split}.parquet"
            if not po.exists():
                n = len(y_pool)
                mean = np.full(n, np.nan)
                Q = np.full((n, len(QUANTILES)), np.nan)
                for tr, ho in KFold(n_splits=5, shuffle=True,
                                    random_state=42).split(Xp):
                    m_ = make_teacher("auto")
                    fit_teacher(m_, Xp[tr], y_pool[tr])
                    m, q = teacher_predict(m_, Xp[ho])
                    mean[ho], Q[ho] = m, q
                    free_teacher(m_)
                df = pd.DataFrame(Q, columns=qcols)
                df.insert(0, "mean", mean)
                df.insert(0, "y", y_pool)
                df.to_parquet(po)
            oof = pd.read_parquet(po)
            if (ds, split, "teacher", 0) not in d:
                m_ = make_teacher("auto")
                t0 = time.perf_counter()
                fit_teacher(m_, Xp, y_pool)
                m, Q = teacher_predict(m_, Xt)
                free_teacher(m_)
                append(dict(dataset=ds, split=split, cell="teacher", seed=0,
                            crps=crps_from_quantiles(y_test, Q),
                            picp80=interval_metrics(y_test, Q)["picp80"],
                            time_s=round(time.perf_counter() - t0, 1)))
            print("  OOF/professor ok", flush=True)
            for seed in (0, 1, 2):
                for cell, fit in (
                        ("student_quant_hard",
                         lambda: fit_xgb_quant_hard(Xp, y_pool, params, seed)),
                        ("student_quant_soft",
                         lambda: fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(),
                                                    params, seed))):
                    if (ds, split, cell, seed) in done():
                        continue
                    set_seed(seed)
                    t0 = time.perf_counter()
                    model = fit()
                    Q = sort_quantiles(np.asarray(model.predict(Xt)))
                    append(dict(dataset=ds, split=split, cell=cell, seed=seed,
                                crps=crps_from_quantiles(y_test, Q),
                                picp80=interval_metrics(y_test, Q)["picp80"],
                                time_s=round(time.perf_counter() - t0, 1)))
                    print(f"  [{cell} seed={seed}] ok", flush=True)

    r = pd.read_csv(CSV)
    g = r.groupby(["split", "dataset", "cell"]).crps.mean().unstack()
    g["ret"] = (g.student_quant_hard - g.student_quant_soft) / \
               (g.student_quant_hard - g.teacher)
    print("\n=== retenção por (split, conjunto) ===")
    print(g["ret"].unstack(level=1).round(3).to_string())
    print("MULTISPLIT DONE", flush=True)


if __name__ == "__main__":
    main()
