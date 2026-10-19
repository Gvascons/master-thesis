#!/usr/bin/env python3
"""Robustez do proxy de latência do E2 (condicionada no registro de
15/10/2026 em docs/desenho-melhorias-finais.md à re-execução da latência
multi-dataset, concluída em 19/10/2026).

Pergunta pré-fixada: o veredito do E2 (H-W1: cruzamento mediano FM→GBDT
entre w=0,9 e 0,8; GBDT regret-ótima para w<=0,6) depende do dataset
usado como proxy de latência? Re-executa a varredura de regret trocando
o ranking de latência do adult (49k) pelos do credit_g (1k) e do
cardiovascular_disease (70k), medidos sob o mesmo protocolo (dispositivo
natural, 1 warm-up + mediana de 5 passadas).

Saída: results/aggregated/lodo_weighted_regret_robustness.csv + resumo
(correlação de Spearman entre rankings; cruzamentos por proxy).
Uso: uv run python scripts/latency_proxy_robustness.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PRIMARY = {"binary": ("roc_auc", True), "multiclass": ("log_loss", False),
           "regression": ("rmse", False)}
FAMILIES = {
    "xgboost": "GBDT", "lightgbm": "GBDT", "catboost": "GBDT",
    "ft_transformer": "DL", "tabnet": "DL", "saint": "DL", "stab": "DL",
    "tabm": "DL", "mlp": "DL", "realmlp": "DL", "kan": "DL", "tabkan": "DL",
    "tabpfn": "FM", "tabfm": "FM",
}
POLICIES = {"Always-GBDT": "GBDT", "Always-DL": "DL", "Always-FM": "FM"}
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def load_latencies():
    adult = pd.read_csv(REPO / "results/latency/latency_adult.csv")
    multi = pd.read_csv(REPO / "results/latency/latency_multi.csv")
    out = {"adult (49k)": adult.set_index("model")["us_per_row"]}
    for ds, label in [("credit_g", "credit_g (1k)"),
                      ("cardiovascular_disease", "cardiovascular (70k)")]:
        sub = multi[multi.dataset == ds].set_index("model")["us_per_row"]
        out[label] = sub
    return out


def regret_sweep(lat_series, tr, tree_pred):
    lat_rank = lat_series.rank()
    lat_norm = (lat_rank - 1) / (len(lat_rank) - 1)
    rows = []
    for ds, sub in tr.groupby("dataset"):
        task = sub.task_type.iloc[0]
        metric, higher = PRIMARY[task]
        s = sub.dropna(subset=[metric]).set_index("model")[metric]
        if len(s) < 4:
            continue
        acc_rank = s.rank(ascending=not higher)
        acc_norm = (acc_rank - 1) / (len(acc_rank) - 1)
        for w in WEIGHTS:
            score = (w * acc_norm + (1 - w) * lat_norm.reindex(acc_norm.index)).dropna()
            fam = pd.Series({m: FAMILIES[m] for m in score.index})
            best, worst = score.min(), score.max()
            denom = (worst - best) or 1.0
            pol_fam = dict(POLICIES)
            if ds in tree_pred.index:
                pol_fam["Tree (LODO)"] = tree_pred.loc[ds]
            for pol, family in pol_fam.items():
                in_fam = score[fam == family]
                if in_fam.empty:
                    continue
                rows.append({"dataset": ds, "w_acc": w, "policy": pol,
                             "regret": float((in_fam.min() - best) / denom)})
    return pd.DataFrame(rows)


def main():
    tr = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    tree_pred = (pd.read_csv(REPO / "results/aggregated/lodo_validation_14.csv")
                 .set_index("dataset")["pred"])
    lats = load_latencies()

    # 1) estabilidade dos rankings entre datasets de medição
    models = sorted(FAMILIES)
    mat = pd.DataFrame({k: v.reindex(models) for k, v in lats.items()})
    print("== Spearman entre rankings de latência (14 modelos) ==")
    keys = list(lats)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rho, _ = spearmanr(mat[keys[i]], mat[keys[j]])
            print(f"  {keys[i]} x {keys[j]}: rho={rho:.3f}")

    # 2) varredura de regret por proxy
    all_rows = []
    for label, lat in lats.items():
        df = regret_sweep(lat, tr, tree_pred)
        df["lat_source"] = label
        all_rows.append(df)
        med = (df.groupby(["policy", "w_acc"])["regret"].median()
               .unstack("policy").round(3))
        print(f"\n== mediana do regret — proxy: {label} ==")
        print(med.to_string())
    full = pd.concat(all_rows, ignore_index=True)
    out = REPO / "results/aggregated/lodo_weighted_regret_robustness.csv"
    full.round(6).to_csv(out, index=False)

    # 3) veredito mecânico: menor w em que FM ainda é regret-ótima (mediana)
    #    e maior w em que GBDT tem mediana 0
    print("\n== cruzamentos por proxy (mediana) ==")
    for label in lats:
        med = (full[full.lat_source == label]
               .groupby(["policy", "w_acc"])["regret"].median().unstack("policy"))
        fm_opt = [w for w in WEIGHTS
                  if med.loc[w, "Always-FM"] <= med.loc[w].min() + 1e-9]
        gbdt_zero = [w for w in WEIGHTS if med.loc[w, "Always-GBDT"] <= 1e-9]
        print(f"  {label}: FM regret-ótima em w={sorted(fm_opt)}; "
              f"GBDT mediana 0 em w<={max(gbdt_zero) if gbdt_zero else None}")
    print(f"\nsalvo: {out}")


if __name__ == "__main__":
    main()
