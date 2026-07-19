#!/usr/bin/env python3
"""Pareto frontier figure (AI-2, H3): accuracy x inference latency per strategy.

Small multiples: rows = metric (RMSE; CRPS for distributional systems),
cols = datasets. Log-x latency. Categorical palette: validated 6-slot
reference order (dataviz skill, ALL CHECKS PASS light mode); marker shapes
are the secondary encoding (CVD/contrast relief). One axis per facet.

Usage: uv run python scripts/plot_pareto.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "distillation" / "pareto.csv"
FIGDIR = REPO / "results" / "figures"

# fixed identity order (never re-sorted): color + marker per strategy
STYLE = {
    "teacher_ctx":        ("#2a78d6", "o", "-",  "TabPFN (contexto reduzido)"),
    "teacher_ens":        ("#008300", "^", "",   "TabPFN (ensemble reduzido)"),
    "student_quant_soft": ("#e87ba4", "*", "",   "Aluno-quantil DESTILADO"),
    "student_point_hard": ("#eda100", "s", "",   "Aluno ponto (controle)"),
    "student_quant_hard": ("#1baf7a", "D", "",   "Aluno-quantil (controle)"),
    "tabfm_full":         ("#eb6834", "P", "",   "TabFM (contexto cheio)"),
}


def facet(ax, sub, metric):
    for system, (color, marker, ls, label) in STYLE.items():
        s = sub[(sub.system == system) & sub[metric].notna()]
        if s.empty:
            continue
        s = s.sort_values("us_per_row")
        ms = 14 if marker == "*" else 8
        if ls:  # the context curve is a connected line
            ax.plot(s.us_per_row, s[metric], ls, color=color, lw=2,
                    marker=marker, ms=ms, label=label, zorder=3)
        else:
            ax.scatter(s.us_per_row, s[metric], c=color, marker=marker,
                       s=ms**2, label=label, zorder=4,
                       edgecolors="white", linewidths=1.5)
        if system == "student_quant_soft":
            for _, r in s.iterrows():
                ax.annotate("destilado", (r.us_per_row, r[metric]),
                            textcoords="offset points", xytext=(8, -12),
                            fontsize=8, color=color, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.tick_params(labelsize=8)


def main():
    df = pd.read_csv(CSV)
    df["crps"] = pd.to_numeric(df["crps"], errors="coerce")
    datasets = [d for d in ("california_housing", "diamonds", "year_prediction")
                if d in set(df.dataset)]
    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 7.4))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, ds in enumerate(datasets):
        sub = df[df.dataset == ds]
        facet(axes[0, j], sub, "rmse")
        axes[0, j].set_title(ds, fontsize=11)
        facet(axes[1, j], sub, "crps")
        axes[1, j].set_xlabel("latência de inferência (µs/linha, log)", fontsize=9)
    axes[0, 0].set_ylabel("RMSE ↓", fontsize=10)
    axes[1, 0].set_ylabel("CRPS ↓ (sistemas distribucionais)", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Destilar × comprimir contexto × reduzir ensemble — "
                 "a fronteira acurácia × latência (hold-out do benchmark)",
                 fontsize=12, y=1.05)
    plt.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pareto_distill.{ext}", dpi=150,
                    bbox_inches="tight")
    print(f"figura salva: {FIGDIR}/pareto_distill.png|pdf "
          f"({n} datasets, {len(df)} pontos)")


if __name__ == "__main__":
    main()
