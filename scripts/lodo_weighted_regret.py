#!/usr/bin/env python3
"""E2 (fase final, pré-registrado em docs/desenho-melhorias-finais.md):
regret multicritério ponderado das políticas fixas, @14 modelos.

Score por modelo x dataset: S = w*r_acc + (1-w)*r_lat, com r_acc = rank
normalizado [0,1] da métrica primária no dataset e r_lat = rank
normalizado da latência de inferência (proxy: medição padronizada no
adult — limitação declarada no pré-registro). Varredura w ∈ {1.0..0.0}.
Regret normalizado por política e dataset como no estudo original:
(best_in_family(S) - best_overall(S)) / (worst_overall(S) - best_overall(S)).

Saída: results/aggregated/lodo_weighted_regret.csv + figura
results/figures/lodo_weighted_regret.{png,pdf}
Uso: uv run python scripts/lodo_weighted_regret.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def main():
    tr = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    lat = pd.read_csv(REPO / "results/latency/latency_adult.csv")
    lat_rank = lat.set_index("model")["us_per_row"].rank()  # 1 = mais rapido
    lat_norm = (lat_rank - 1) / (len(lat_rank) - 1)         # [0,1]

    rows = []
    for ds, sub in tr.groupby("dataset"):
        task = sub.task_type.iloc[0]
        metric, higher = PRIMARY[task]
        s = sub.dropna(subset=[metric]).set_index("model")[metric]
        if len(s) < 4:
            continue
        acc_rank = s.rank(ascending=not higher)             # 1 = melhor
        acc_norm = (acc_rank - 1) / (len(acc_rank) - 1)
        for w in WEIGHTS:
            score = w * acc_norm + (1 - w) * lat_norm.reindex(acc_norm.index)
            score = score.dropna()
            fam = pd.Series({m: FAMILIES[m] for m in score.index})
            best, worst = score.min(), score.max()
            denom = (worst - best) or 1.0
            for pol, family in POLICIES.items():
                in_fam = score[fam == family]
                if in_fam.empty:
                    continue
                regret = (in_fam.min() - best) / denom
                rows.append({"dataset": ds, "task": task, "w_acc": w,
                             "policy": pol, "regret": round(float(regret), 6)})
    df = pd.DataFrame(rows)
    out = REPO / "results/aggregated/lodo_weighted_regret.csv"
    df.to_csv(out, index=False)

    agg = (df.groupby(["policy", "w_acc"])["regret"]
           .agg(["mean", "median"]).reset_index())
    print(agg.pivot(index="w_acc", columns="policy", values="median")
          .round(3).to_string())

    # figura: mediana do regret vs peso da acuracia (validated palette)
    COLORS = {"Always-GBDT": "#2a78d6", "Always-DL": "#eda100",
              "Always-FM": "#008300"}
    MARKERS = {"Always-GBDT": "o", "Always-DL": "s", "Always-FM": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, stat in zip(axes, ("median", "mean")):
        for pol in POLICIES:
            a = agg[agg.policy == pol].sort_values("w_acc")
            ax.plot(a.w_acc, a[stat], "-", color=COLORS[pol], lw=2,
                    marker=MARKERS[pol], ms=6, label=pol,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlabel("accuracy weight $w$  (1$-$$w$ on latency)")
        ax.set_ylabel(f"{stat} normalized regret (14 models)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, 1)
    axes[0].legend(fontsize=9, frameon=False)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / f"results/figures/lodo_weighted_regret.{ext}",
                    dpi=150, bbox_inches="tight")
    print(f"\nsalvos: {out.name} + figura lodo_weighted_regret.png|pdf")


if __name__ == "__main__":
    main()
