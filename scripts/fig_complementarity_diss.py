#!/usr/bin/env python3
"""Figura do E7 para a dissertação (cap. 5, Ato V — complementaridade).

Painel (a): vitórias líquidas do FM sobre o XGBoost por 1.000 instâncias
do hold-out, por conjunto de classificação (asterisco: McNemar p<0,01).
Painel (b): vitórias líquidas por quintil de margem do XGBoost nas 10
binárias (linhas finas: conjuntos; grossa: mediana) — H-C2.

Fonte: results/complementarity/complementarity_{pairs,margens}.csv.
Saída: results/figures/complementarity_diss.{png,pdf}
Uso:   uv run python scripts/fig_complementarity_diss.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
CMAP = {"tabpfn": "#2a9d8f", "tabfm": "#6a4c93"}
DISPLAY = {"tabpfn": "TabPFN", "tabfm": "TabFM"}


def main():
    pairs = pd.read_csv(REPO / "results/complementarity/complementarity_pairs.csv")
    marg = pd.read_csv(REPO / "results/complementarity/complementarity_margens.csv")
    cla = pairs[pairs.task_type.isin(["binary", "multiclass"])].copy()
    cla["net_mil"] = cla.net_fm_rate * 1000

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.4),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- painel (a): vitórias líquidas por conjunto -----------------------
    order = (cla[cla.fm == "tabfm"].set_index("dataset").net_mil
             .sort_values().index.tolist())
    ypos = np.arange(len(order))
    h = 0.38
    for off, fm in ((+h / 2, "tabpfn"), (-h / 2, "tabfm")):
        sub = cla[cla.fm == fm].set_index("dataset").reindex(order)
        a1.barh(ypos + off, sub.net_mil, height=h, color=CMAP[fm],
                label=DISPLAY[fm])
        for yy, (v, p) in zip(ypos + off, zip(sub.net_mil, sub.mcnemar_p)):
            if pd.notna(p) and p < 0.01:
                a1.text(v + (1.5 if v >= 0 else -1.5), yy, "*",
                        va="center_baseline", ha="left" if v >= 0 else "right",
                        fontsize=11, color="#333")
    a1.axvline(0, color="#555", lw=1)
    a1.set_yticks(ypos)
    a1.set_yticklabels([d.replace("_", "\\_") if False else d for d in order],
                       fontsize=9)
    a1.set_xlabel("vitórias líquidas do FM por 1.000 instâncias\n"
                  "($\\leftarrow$ XGBoost melhor $\\;|\\;$ FM melhor $\\rightarrow$)")
    a1.set_title("(a) Onde cada FM ganha do XGBoost (classificação)",
                 fontsize=11, loc="left")
    a1.legend(loc="lower right", fontsize=9, frameon=False)
    a1.grid(axis="x", alpha=0.25, lw=0.6)
    a1.set_axisbelow(True)

    # ---- painel (b): quintis de margem do XGBoost (binárias) --------------
    for fm in ("tabpfn", "tabfm"):
        sub = marg[marg.fm == fm]
        piv = sub.pivot_table(index="dataset", columns="quintil_margem",
                              values="net_fm_rate") * 1000
        for _, row in piv.iterrows():
            a2.plot(piv.columns, row.values, color=CMAP[fm], alpha=0.22, lw=0.9)
        med = piv.median(axis=0)
        a2.plot(med.index, med.values, color=CMAP[fm], lw=2.6, marker="o",
                ms=5, label=f"{DISPLAY[fm]} (mediana)")
    a2.axhline(0, color="#555", lw=1)
    a2.set_xticks(sorted(marg.quintil_margem.unique()))
    a2.set_xticklabels(["Q1\nincerto", "Q2", "Q3", "Q4", "Q5\nconfiante"],
                       fontsize=9)
    a2.set_xlabel("quintil de margem do XGBoost ($|\\hat p - 0{,}5|$)")
    a2.set_ylabel("vitórias líquidas do FM por 1.000 instâncias")
    a2.set_title("(b) Ganho por margem do XGBoost (10 binárias)",
                 fontsize=11, loc="left")
    a2.legend(loc="upper right", fontsize=9, frameon=False)
    a2.grid(axis="y", alpha=0.25, lw=0.6)
    a2.set_axisbelow(True)

    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = REPO / "results/figures/complementarity_diss"
    fig.savefig(f"{out}.png", dpi=180)
    fig.savefig(f"{out}.pdf")
    print(f"OK: {out}.png/.pdf")


if __name__ == "__main__":
    main()
