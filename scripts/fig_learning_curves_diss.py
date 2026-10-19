#!/usr/bin/env python3
"""Variante da figura de curvas de aprendizado para a DISSERTAÇÃO
(auditoria de layout 19/10/2026): os 5 painéis em fileira única de
figures_ai1_display.py ficam ilegíveis quando escalados à largura de
texto A4 retrato; esta versão usa grade 2x3 (5 painéis + célula de
legenda) com fontes maiores. Mesmo artefato de dados e mesma semântica.

Saída: results/figures/learning_curves_diss.{png,pdf}
Uso: uv run python scripts/fig_learning_curves_diss.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DISPLAY = {"xgboost": "XGBoost", "catboost": "CatBoost", "tabm": "TabM",
           "realmlp": "RealMLP", "ft_transformer": "FT-Transformer",
           "tabpfn": "TabPFN"}
CMAP = {"xgboost": "#c1121f", "catboost": "#e8590c", "tabm": "#1d4e89",
        "realmlp": "#4895ef", "ft_transformer": "#3a0ca3",
        "tabpfn": "#2a9d8f"}
PRIMARY = {"binary": ("roc_auc", True), "multiclass": ("log_loss", False),
           "regression": ("rmse", False)}


def main():
    df = pd.read_csv(REPO / "results/learning_curve/curve.csv")
    df["primary"] = df.apply(lambda r: r[PRIMARY[r.task_type][0]], axis=1)

    def agg(d):
        g = d.groupby(["model", "train_size"])["primary"]
        return (g.mean().rename("mean").reset_index()
                .merge(g.std().rename("std").reset_index(),
                       on=["model", "train_size"]))

    datasets = [x for x in ("give_me_some_credit", "higgs", "adult",
                            "jannis", "year_prediction")
                if x in df.dataset.unique()]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    axes = axes.ravel()
    for ax, ds in zip(axes, datasets):
        sub = df[df.dataset == ds]
        task = sub.task_type.iloc[0]
        col, higher = PRIMARY[task]
        a = agg(sub)
        for mdl in sub.model.unique():
            m = a[a.model == mdl].sort_values("train_size")
            ax.plot(m.train_size, m["mean"], "-o", ms=3.5, lw=2.0,
                    color=CMAP.get(mdl, "gray"))
            ax.fill_between(m.train_size, m["mean"] - m["std"],
                            m["mean"] + m["std"],
                            color=CMAP.get(mdl, "gray"), alpha=0.12)
        ax.set_xscale("log")
        yl = ""
        if not higher:
            ax.set_yscale("log")
            yl = " (log)"
        nice = {"roc_auc": "AUC", "log_loss": "log-loss", "rmse": "RMSE"}
        arrow = " ↑" if higher else " ↓"
        ax.set_title(f"{ds}\n({task}, {nice[col]}{arrow}{yl})", fontsize=11)
        ax.set_xlabel("tamanho do pool de treino", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("métrica primária", fontsize=10)
    axes[3].set_ylabel("métrica primária", fontsize=10)
    # última célula vira painel de legenda
    axes[5].axis("off")
    handles = [Line2D([0], [0], color=c, lw=2.2, marker="o", ms=5)
               for c in CMAP.values()]
    axes[5].legend(handles, [DISPLAY[m] for m in CMAP], loc="center",
                   fontsize=12, frameon=False, title="Modelos",
                   title_fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / f"results/figures/learning_curves_diss.{ext}",
                    dpi=200, bbox_inches="tight")
    print("salvo: results/figures/learning_curves_diss.png|pdf")


if __name__ == "__main__":
    main()
