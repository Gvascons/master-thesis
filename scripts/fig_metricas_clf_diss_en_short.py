"""Variante -short do fig_metricas_clf_diss.py (B19, 27/11/2026):
titulo do painel (a) sem travessao e saida SOMENTE em
dissertacao-en-short/images/ (a longa permanece intacta).
Rodar: uv run python scripts/fig_metricas_clf_diss_en_short.py en
"""
import sys
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
    en = len(sys.argv) > 1 and sys.argv[1] == "en"

    def num(s):
        return s if en else s.replace(".", ",")

    df = pd.read_parquet(REPO / "results/complementarity/preds_xgboost_adult.parquet")
    y, p = df.y_true.values, df.p1.values
    auc = roc_auc_score(y, p)
    fpr, tpr, thr = roc_curve(y, p)

    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- painel (a): ROC ---------------------------------------------------
    a1.plot(fpr, tpr, color=TEAL, lw=2.2)
    a1.plot([0, 1], [0, 1], ls="--", color="#999", lw=1.2)
    a1.fill_between(fpr, 0, tpr, color=TEAL, alpha=0.08)
    for t_mark in (0.25, 0.50, 0.75):
        i = int(np.argmin(np.abs(thr - t_mark)))
        a1.plot(fpr[i], tpr[i], "o", ms=7, color=INK, zorder=5)
        a1.annotate(num(f"{'threshold' if en else 'limiar'} {t_mark:.2f}"),
                    (fpr[i], tpr[i]), textcoords="offset points",
                    xytext=(10, -4), fontsize=12, color=INK)
        print(f"limiar {t_mark:.2f}: FPR={fpr[i]:.3f} TPR(recall)={tpr[i]:.3f}")
    a1.text(0.55, 0.28, num(f"AUC = {auc:.3f}"),
            fontsize=14, color=TEAL, fontweight="bold")
    a1.text(0.45, 0.38, "chance (AUC = 0.5)" if en else "acaso (AUC = 0,5)",
            fontsize=12, color="#999", rotation=38, rotation_mode="anchor")
    if en:
        a1.set_xlabel("false positive rate (1 $-$ specificity)")
        a1.set_ylabel("true positive rate (recall)")
        a1.set_title("(a) ROC curve: XGBoost on the adult hold-out",
                     fontsize=13, loc="left")
    else:
        a1.set_xlabel("taxa de falsos positivos (1 $-$ especificidade)")
        from matplotlib.ticker import FuncFormatter
        _fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        for _ax in (a1, a2):
            _ax.xaxis.set_major_formatter(_fmt)
            _ax.yaxis.set_major_formatter(_fmt)

        a1.set_ylabel("taxa de verdadeiros positivos (recall)")
        a1.set_title("(a) Curva ROC — XGBoost no hold-out do adult",
                     fontsize=13, loc="left")

    # ---- painel (b): precisão/recall vs limiar -----------------------------
    ts = np.linspace(0.02, 0.98, 97)
    prec = [precision_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    rec = [recall_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    lab_p = "precision (class > 50K)" if en else "precisão (classe > 50K)"
    lab_r = "recall (class > 50K)" if en else "recall (classe > 50K)"
    a2.plot(ts, prec, color=TEAL, lw=2.2, label=lab_p)
    a2.plot(ts, rec, color=PURPLE, lw=2.2, label=lab_r)
    a2.axvline(0.5, color="#555", lw=1, ls=":")
    a2.text(0.51, 0.04,
            "default threshold 0.5" if en else "limiar padrão 0,5",
            fontsize=12, color="#555")
    i05 = int(np.argmin(np.abs(ts - 0.5)))
    print(f"t=0,5: precisão={prec[i05]:.3f} recall={rec[i05]:.3f}")
    if en:
        a2.set_xlabel("decision threshold on $\\hat p(\\text{income} > 50K)$")
        a2.set_ylabel("metric value")
        a2.set_title("(b) The trade-off the threshold governs",
                     fontsize=13, loc="left")
    else:
        a2.set_xlabel("limiar de decisão sobre $\\hat p(\\text{renda} > 50K)$")
        a2.set_ylabel("valor da métrica")
        a2.set_title("(b) O trade-off que o limiar governa", fontsize=13, loc="left")
    a2.set_ylim(0, 1.02)
    a2.legend(loc="upper center", fontsize=12, frameon=False,
              ncol=2, bbox_to_anchor=(0.5, -0.16))

    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout()
    if en:
        outs = [REPO / "dissertacao-en-short/images/metricas_clf_diss_en"]
    else:
        outs = [REPO / "results/figures/metricas_clf_diss"]
    for out in outs:
        fig.savefig(f"{out}.png", dpi=180)
        fig.savefig(f"{out}.pdf")
        print(f"OK: {out}.png/.pdf")


if __name__ == "__main__":
    main()
