#!/usr/bin/env python3
"""Figura do cap. 5 (Ato V) — leitura operacional no limiar padrão.

Painel (a): a armadilha da acurácia no amazon_employee --- acurácias
quase indistinguíveis (0,943--0,949) ao lado de recalls da classe
minoritária de 0,032 (XGBoost) a 0,298 (TabFM).
Painel (b): recall da classe minoritária nos 10 conjuntos binários,
ordenados por desbalanceamento (fração minoritária crescente).

Fonte: results/aggregated/secondary_metrics_holdout.csv (gate 37/37
contra test_results; gerado por scripts/secondary_metrics_report.py).
Saída: results/figures/metricas_operacionais_diss.{png,pdf}
Uso:   uv run python scripts/fig_metricas_operacionais_diss.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
# mesma semântica de cor da Figura 5.8 (fig_complementarity_diss):
# TabPFN teal, TabFM roxo; XGBoost (linha de base) em cinza-escuro.
CMAP = {"xgboost": "#666666", "tabpfn": "#2a9d8f", "tabfm": "#6a4c93"}
NAME = {"xgboost": "XGBoost", "tabpfn": "TabPFN", "tabfm": "TabFM"}
MODELS = ["xgboost", "tabpfn", "tabfm"]


def fmt(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def main():
    df = pd.read_csv(REPO / "results/aggregated/secondary_metrics_holdout.csv")
    assert (df.sanity_gate == "OK").all(), "gate de sanidade violado"
    bi = df[df.task_type == "binary"].set_index(["dataset", "model"])

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )

    # ---- painel (a): amazon_employee, acurácia vs recall da minoritária ----
    az = bi.loc["amazon_employee"]
    width, xs = 0.26, np.arange(2)
    for k, m in enumerate(MODELS):
        vals = [az.loc[m, "accuracy"], az.loc[m, "recall_min"]]
        bars = a1.bar(xs + (k - 1) * width, vals, width * 0.92,
                      color=CMAP[m], label=NAME[m])
        for b, v in zip(bars, vals):
            a1.text(b.get_x() + b.get_width() / 2, v + 0.015, fmt(v),
                    ha="center", va="bottom", fontsize=8.5, color="#333")
        print(f"amazon {m}: acc={vals[0]:.3f} recall_min={vals[1]:.3f}")
    a1.set_xticks(xs)
    a1.set_xticklabels(["acurácia", "recall da classe\nminoritária"])
    a1.set_ylim(0, 1.06)
    a1.set_ylabel("valor da métrica (limiar 0,5)")
    a1.set_title("(a) amazon_employee (minoritária: 5,8%)",
                 fontsize=11, loc="left")
    a1.legend(loc="upper right", fontsize=9, frameon=False)

    # ---- painel (b): recall da minoritária nas 10 binárias -----------------
    order = (bi.reset_index().groupby("dataset").minority_frac.first()
             .sort_values().index.tolist())
    labels = {
        "amazon_employee": "amazon_employee",
        "give_me_some_credit": "give_me_some_credit",
        "bank_marketing": "bank_marketing",
        "adult": "adult",
        "telco_customer_churn": "telco_customer_churn",
        "phoneme": "phoneme",
        "credit_g": "credit_g",
        "higgs": "higgs",
        "cardiovascular_disease": "cardiovascular_disease",
        "magictelescope": "magictelescope",
    }
    ypos = np.arange(len(order))[::-1]  # mais desbalanceado no topo
    for y, ds in zip(ypos, order):
        vals = [bi.loc[(ds, m), "recall_min"] for m in MODELS]
        a2.plot([min(vals), max(vals)], [y, y], color="#bbbbbb",
                lw=1.4, zorder=1)
        for m, v in zip(MODELS, vals):
            a2.plot(v, y, "o", ms=8, color=CMAP[m], mec="white",
                    mew=1.6, zorder=3, label=NAME[m] if y == ypos[0] else None)
    frac = bi.reset_index().groupby("dataset").minority_frac.first()
    a2.set_yticks(ypos)
    a2.set_yticklabels(
        [f"{labels[d]} ({frac[d] * 100:.1f}%)".replace(".", ",") for d in order],
        fontsize=9)
    a2.set_xlim(0, 1.0)
    a2.set_xlabel("recall da classe minoritária (limiar 0,5)")
    a2.set_title("(b) As 10 binárias, da mais à menos desbalanceada",
                 fontsize=11, loc="left")
    a2.legend(loc="upper right", fontsize=9, frameon=False)

    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6, axis="x" if ax is a2 else "y")
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = REPO / "results/figures/metricas_operacionais_diss"
    fig.savefig(f"{out}.png", dpi=180)
    fig.savefig(f"{out}.pdf")
    print(f"OK: {out}.png/.pdf")


if __name__ == "__main__":
    main()
