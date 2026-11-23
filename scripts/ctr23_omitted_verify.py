"""Verificação e incorporação da sensibilidade aos 4 conjuntos CTR23 omitidos.

Contexto (adendo de 23/11/2026 ao desenho experimental): a auditoria externa
de 23/11 identificou que a regra de elegibilidade declarada (CTR23, 3k-100k
linhas, sem vazamento, sem sobreposição com o núcleo) alcança 19 conjuntos,
não os 15 rodados. Os 4 ausentes: white_wine (44971; subconjunto do
wine_quality do núcleo), california_housing contínuo (44977; versão não
binarizada do conjunto do núcleo), superconductivity (44964; à época da
seleção acreditava-se sobreposto ao núcleo — a errata de 19/10 mostrou que a
entrada "superconduct" sempre foi o online_news) e naval_propulsion_plant
(44969; sem justificativa registrada).

A auditoria rodou os 4 com o protocolo de scripts/distill.py (tuning 100
trials, professor OOF K=5 semente 42, alunos em sementes {0,1,2}, um split).
Artefatos copiados para results/distillation/ctr23_omitidos/ (PROVENANCE.md).

Este script RE-TREINA os alunos quantílicos e o professor de hold-out a
partir dos caches (params + OOF) e compara com resultados.csv, depois monta
a tabela de sensibilidade do placar:

  results/distillation/ctr23_omitidos/verify.csv     (minha re-execução)
  results/distillation/ctr23_sensitivity.csv         (placares e p-valores)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

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
from src.utils.reproducibility import set_seed

DS = {"naval_propulsion_plant": 44969, "superconductivity": 44964,
      "california_ctr23": 44977, "white_wine": 44971}
qcols = [f"q{int(t*100):02d}" for t in QUANTILES]
VERIFY = DIR / "verify.csv"


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


def main():
    rows = []
    for ds, did in DS.items():
        print(f"\n=== {ds} ===", flush=True)
        X, y, info = load(ds, did)
        X_pool, y_pool, X_test, y_test = get_holdout_split(X, y, info, seed=42,
                                                           test_size=0.2)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        params = json.loads((DIR / f"xgboost_{ds}.json").read_text())["best_params"]
        oof = pd.read_parquet(DIR / f"oof_tabpfn_{ds}.parquet")
        assert len(oof) == len(y_pool), f"{ds}: OOF desalinhado"
        assert np.allclose(oof["y"].to_numpy(), y_pool), f"{ds}: OOF y != pool"

        # diagnóstico de gêmeos (covariáveis idênticas teste -> pool)
        pool_keys = set(map(tuple, X_pool.astype(str).values.tolist()))
        twin = np.mean([tuple(r) in pool_keys
                        for r in X_test.astype(str).values.tolist()])
        print(f"  gêmeos teste->pool: {100*twin:.1f}%", flush=True)

        m_ = make_teacher("auto")
        t0 = time.perf_counter()
        fit_teacher(m_, Xp, y_pool)
        tm, tQ = teacher_predict(m_, Xt)
        free_teacher(m_)
        rows.append(dict(dataset=ds, cell="teacher", seed=0,
                         crps=crps_from_quantiles(y_test, tQ),
                         rmse=float(np.sqrt(np.mean((y_test - tm) ** 2))),
                         picp80=interval_metrics(y_test, tQ)["picp80"],
                         twin_frac=round(float(twin), 4),
                         time_s=round(time.perf_counter() - t0, 1)))
        for seed in (0, 1, 2):
            for cell, fit in (("student_quant_hard",
                               lambda: fit_xgb_quant_hard(Xp, y_pool, params, seed)),
                              ("student_quant_soft",
                               lambda: fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(), params, seed))):
                set_seed(seed)
                t0 = time.perf_counter()
                model = fit()
                Q = sort_quantiles(np.asarray(model.predict(Xt)))
                rows.append(dict(dataset=ds, cell=cell, seed=seed,
                                 crps=crps_from_quantiles(y_test, Q),
                                 rmse=float(np.sqrt(np.mean(
                                     (y_test - Q[:, QUANTILES.index(0.50)]) ** 2))),
                                 picp80=interval_metrics(y_test, Q)["picp80"],
                                 twin_frac=round(float(twin), 4),
                                 time_s=round(time.perf_counter() - t0, 1)))
                print(f"  [{cell} seed={seed}] crps={rows[-1]['crps']:.6g}",
                      flush=True)
        pd.DataFrame(rows).to_csv(VERIFY, index=False)

    mine = pd.DataFrame(rows)
    audit = pd.read_csv(DIR / "resultados.csv")
    comp = mine.merge(audit, on=["dataset", "cell", "seed"],
                      suffixes=("_mine", "_audit"))
    comp["rel_diff_crps"] = ((comp.crps_mine - comp.crps_audit).abs()
                             / comp.crps_audit.abs())
    print("\n=== comparação com a auditoria (CRPS) ===")
    print(comp[["dataset", "cell", "seed", "crps_mine", "crps_audit",
                "rel_diff_crps"]].to_string(index=False))
    print(f"máx desvio relativo: {comp.rel_diff_crps.max():.2e}")

    # sensibilidade do placar
    g = mine.groupby(["dataset", "cell"]).crps.mean().unstack()
    g["ret"] = (g.student_quant_hard - g.student_quant_soft) / \
               (g.student_quant_hard - g.teacher)
    base = pd.read_csv(REPO / "results" / "distillation" / "paper_retention.csv")
    pos20 = int((base.ret_crps > 0).sum())
    new = g["ret"]
    out = []

    def add(name, wins, n):
        out.append(dict(cenario=name, positivos=wins, n=n,
                        p_binomial=binomtest(wins, n, 0.5,
                                             alternative="greater").pvalue))

    add("publicado (20)", pos20, 20)
    add("com os 4 omitidos (24)", pos20 + int((new > 0).sum()), 24)
    keep = [d for d in new.index if d not in ("white_wine", "california_ctr23")]
    add("sem os 2 sobrepostos ao nucleo (22)",
        pos20 + int((new[keep] > 0).sum()), 22)
    sens = pd.DataFrame(out)
    det = g[["teacher", "student_quant_hard", "student_quant_soft", "ret"]].copy()
    det["ganho_rel_pct"] = 100 * (g.student_quant_hard - g.student_quant_soft) \
        / g.student_quant_hard
    det["twin_frac"] = mine.groupby("dataset").twin_frac.first()
    print("\n=== 4 omitidos (minha re-execução) ===")
    print(det.round(4).to_string())
    print("\n=== sensibilidade do placar ===")
    print(sens.to_string(index=False))
    sens.to_csv(REPO / "results" / "distillation" / "ctr23_sensitivity.csv",
                index=False)
    det.to_csv(DIR / "detalhe_retencao.csv")
    print("CTR23 VERIFY DONE", flush=True)


if __name__ == "__main__":
    main()
