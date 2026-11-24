"""Gera o gráfico de floresta da destilação distribucional (Fig. do cap. 7):
ganho relativo de CRPS do aluno destilado sobre o controle, por conjunto,
com IC95 por bootstrap pareado de linhas (B=10.000) no desenho herdado, e o
ganho pontual com os dois braços re-ajustados como marcador vazado.

Fontes: results/distillation/bootstrap_lines.csv (ganho + IC, herdado)
        results/distillation/retune_analysis.csv (ganho re-ajustado)
Saída:  dissertacao/images/forest_distill.pdf (+ results/figures/)
Com argumento "en": rótulos em inglês, saída
dissertacao-en/images/forest_distill_en.pdf (não toca na PT).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def main():
    en = len(sys.argv) > 1 and sys.argv[1] == "en"
    b = pd.read_csv(REPO / "results/distillation/bootstrap_lines.csv").set_index("dataset")
    r = pd.read_csv(REPO / "results/distillation/retune_analysis.csv").set_index("dataset")
    # ganho re-ajustado (%): controle re-tunado -> destilado re-tunado
    r["gain_retuned_pct"] = (r.crps_hard_retuned - r.crps_soft_retuned) / r.crps_hard_retuned * 100
    d = b.join(r[["gain_retuned_pct"]]).sort_values("gain_pct")
    y = range(len(d))
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.axvline(0, color="grey", lw=0.8)
    cores = ["#2e7d32" if lo > 0 else ("#c62828" if hi < 0 else "#757575")
             for lo, hi in zip(d.gain_ci_lo, d.gain_ci_hi)]
    for yi, (g, lo, hi, c) in enumerate(zip(d.gain_pct, d.gain_ci_lo, d.gain_ci_hi, cores)):
        ax.errorbar(g, yi, xerr=[[g - lo], [hi - g]], fmt="none", ecolor=c,
                    elinewidth=1.6, capsize=2.5)
    ax.scatter(d.gain_pct, y, c=cores, s=28, zorder=3)
    ax.scatter(d.gain_retuned_pct, y, facecolors="none", edgecolors="#1565c0",
               s=34, zorder=3)
    from matplotlib.lines import Line2D
    if en:
        leg = ["inherited (95% CI by row bootstrap)",
               "both arms re-tuned (point estimate)"]
        xlab = "CRPS gain of the distilled student over the control (%)  —  higher is better"
        outs = (REPO / "dissertacao-en/images/forest_distill_en.pdf",
                REPO / "results/figures/forest_distill_en.pdf")
    else:
        leg = ["herdado (IC95 por bootstrap de linhas)",
               "dois braços re-ajustados (pontual)"]
        xlab = "Ganho de CRPS do destilado sobre o controle (%)  —  maior é melhor"
        outs = (REPO / "dissertacao/images/forest_distill.pdf",
                REPO / "results/figures/forest_distill.pdf")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#424242", label=leg[0]),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor="#1565c0", color="#1565c0", label=leg[1]),
    ], loc="lower right", fontsize=8.5)
    nomes = [n + ("†" if n == "california_housing" else "") for n in d.index]
    ax.set_yticks(list(y))
    ax.set_yticklabels(nomes, fontsize=8.5)
    ax.set_xlabel(xlab)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    for out in outs:
        plt.savefig(out, bbox_inches="tight")
    print("ok")


if __name__ == "__main__":
    main()
