#!/usr/bin/env python3
"""Figura pedagógica do cap. 2 — métricas de classificação binária.

Painel (a): curva ROC do XGBoost no hold-out do adult (predições por
instância do E7), com três limiares marcados --- a AUC é a área sob a
curva, e cada ponto da curva é uma escolha de limiar.
Painel (b): precisão e recall da classe minoritária (renda > 50K) em
função do limiar de decisão --- o trade-off que a figura (a) resume.

Fonte: results/complementarity/preds_xgboost_adult.parquet.
Saída: results/figures/metricas_clf_diss.{png,pdf}
Uso:   uv run python scripts/fig_metricas_clf_diss.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
TEAL, PURPLE, INK = "#2a9d8f", "#6a4c93", "#333333"


def main():
    df = pd.read_parquet(REPO / "results/complementarity/preds_xgboost_adult.parquet")
    y, p = df.y_true.values, df.p1.values
    auc = roc_auc_score(y, p)
    fpr, tpr, thr = roc_curve(y, p)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- painel (a): ROC ---------------------------------------------------
    a1.plot(fpr, tpr, color=TEAL, lw=2.2)
    a1.plot([0, 1], [0, 1], ls="--", color="#999", lw=1.2)
    a1.fill_between(fpr, 0, tpr, color=TEAL, alpha=0.08)
    for t_mark in (0.25, 0.50, 0.75):
        i = int(np.argmin(np.abs(thr - t_mark)))
        a1.plot(fpr[i], tpr[i], "o", ms=7, color=INK, zorder=5)
        a1.annotate(f"limiar {t_mark:.2f}".replace(".", ","),
                    (fpr[i], tpr[i]), textcoords="offset points",
                    xytext=(10, -4), fontsize=9, color=INK)
        print(f"limiar {t_mark:.2f}: FPR={fpr[i]:.3f} TPR(recall)={tpr[i]:.3f}")
    a1.text(0.55, 0.28, f"AUC = {auc:.3f}".replace(".", ","),
            fontsize=12, color=TEAL, fontweight="bold")
    a1.text(0.45, 0.38, "acaso (AUC = 0,5)", fontsize=9, color="#999",
            rotation=38, rotation_mode="anchor")
    a1.set_xlabel("taxa de falsos positivos (1 $-$ especificidade)")
    a1.set_ylabel("taxa de verdadeiros positivos (recall)")
    a1.set_title("(a) Curva ROC — XGBoost no hold-out do adult",
                 fontsize=11, loc="left")

    # ---- painel (b): precisão/recall vs limiar -----------------------------
    ts = np.linspace(0.02, 0.98, 97)
    prec = [precision_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    rec = [recall_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    a2.plot(ts, prec, color=TEAL, lw=2.2, label="precisão (classe > 50K)")
    a2.plot(ts, rec, color=PURPLE, lw=2.2, label="recall (classe > 50K)")
    a2.axvline(0.5, color="#555", lw=1, ls=":")
    a2.text(0.51, 0.04, "limiar padrão 0,5", fontsize=9, color="#555")
    i05 = int(np.argmin(np.abs(ts - 0.5)))
    print(f"t=0,5: precisão={prec[i05]:.3f} recall={rec[i05]:.3f}")
    a2.set_xlabel("limiar de decisão sobre $\\hat p(\\text{renda} > 50K)$")
    a2.set_ylabel("valor da métrica")
    a2.set_ylim(0, 1.02)
    a2.set_title("(b) O trade-off que o limiar governa", fontsize=11, loc="left")
    a2.legend(loc="center left", fontsize=9, frameon=False)

    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = REPO / "results/figures/metricas_clf_diss"
    fig.savefig(f"{out}.png", dpi=180)
    fig.savefig(f"{out}.pdf")
    print(f"OK: {out}.png/.pdf")


if __name__ == "__main__":
    main()
