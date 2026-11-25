#!/usr/bin/env python3
"""Figura pedagógica do cap. 2 — da curva de quantis à cobertura.

Painel (a): o que um regressor distribucional devolve para UMA linha ---
os 19 quantis preditos pelo professor (TabPFN, OOF) para uma instância
real do fps_benchmark, com o alvo observado. O eixo x é o nível nominal
tau (fixado pelo avaliador), NÃO uma probabilidade prevista pelo modelo.
Painel (b): a agregação que gera a curva de confiabilidade --- para cada
tau, a fração dos alvos que caem abaixo do quantil tau predito, sobre
todas as linhas do mesmo artefato.

Fonte: results/distillation/oof/tabpfn_fps_benchmark.parquet.
Saída: results/figures/quantis_cobertura_diss.{png,pdf}
Uso:   uv run python scripts/fig_quantis_cobertura_diss.py
Com argumento "en": rótulos em inglês, ponto decimal, saída
dissertacao-en/images/quantis_cobertura_diss_en.{png,pdf} (PT intocada).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TEAL, PURPLE, INK = "#2a9d8f", "#6a4c93", "#333333"
QUANTILES = [round(q, 2) for q in np.arange(0.05, 0.96, 0.05)]
QCOLS = [f"q{int(t * 100):02d}" for t in QUANTILES]


def main():
    en = len(sys.argv) > 1 and sys.argv[1] == "en"

    def num(s):
        return s if en else s.replace(".", ",")

    df = pd.read_parquet(REPO / "results/distillation/oof/tabpfn_fps_benchmark.parquet")
    Q, y = df[QCOLS].to_numpy(), df["y"].to_numpy()

    # instância didática: alvo dentro do leque, cruzando perto de tau ~ 0,6,
    # com leque largo o bastante para ser visível
    inside = (y > Q[:, 0]) & (y < Q[:, -1])
    width = Q[:, -1] - Q[:, 0]
    k_cross = (y[:, None] > Q).sum(axis=1)  # primeiro índice com y <= q
    cand = np.where(inside & (k_cross == 12) & (width > np.quantile(width, 0.8)))[0]
    i = int(cand[0])
    qi, yi = Q[i], y[i]
    print(f"instância {i}: y={yi:.1f}, q05={qi[0]:.1f}, q95={qi[-1]:.1f}, "
          f"cruza em tau={QUANTILES[int(k_cross[i])]}")

    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- painel (a): a curva de quantis de UMA linha -----------------------
    a1.plot(QUANTILES, qi, "-o", color=PURPLE, lw=2.0, ms=5.5,
            markeredgecolor="white", markeredgewidth=0.8)
    a1.axhline(yi, color=INK, lw=1.4, ls="--")
    a1.text(0.06, yi,
            "observed target $y$  " if en else "alvo observado $y$  ",
            fontsize=12, color=INK, va="bottom")
    tau_c = QUANTILES[int(k_cross[i])]
    a1.axvline(tau_c, color="#999999", lw=1.0, ls=":")
    if en:
        ann = f"$y$ falls below $q_\\tau$\nfrom $\\tau = {tau_c:.2f}$ on"
    else:
        ann = f"$y$ cai abaixo de $q_\\tau$\na partir de $\\tau$ = " + f"{tau_c:.2f}".replace(".", ",")
    a1.annotate(ann,
                (tau_c, qi[-1]), xytext=(-8, -2), textcoords="offset points",
                fontsize=12, color="#555555", ha="right", va="top")
    if en:
        a1.set_xlabel("nominal level $\\tau$ (fixed by the evaluator)")
        a1.set_ylabel("predicted quantile $q_\\tau(\\mathbf{x})$ (target unit)")
        a1.set_title("(a) The distributional prediction for one row",
                     fontsize=13, loc="left")
    else:
        a1.set_xlabel("nível nominal $\\tau$ (fixado pelo avaliador)")
        a1.set_ylabel("quantil predito $q_\\tau(\\mathbf{x})$ (unidade do alvo)")
        a1.set_title("(a) A predição distribucional de uma linha",
                     fontsize=13, loc="left")
    a1.set_xlim(0, 1)

    # ---- painel (b): a agregação que vira a curva de confiabilidade --------
    cov = [float(np.mean(y <= Q[:, k])) for k in range(len(QUANTILES))]
    a2.plot([0, 1], [0, 1], "--", color="#999999", lw=1.2)
    a2.plot(QUANTILES, cov, "-o", color=TEAL, lw=2.0, ms=5.5,
            markeredgecolor="white", markeredgewidth=0.8)
    k20 = QUANTILES.index(0.20)
    a2.plot(0.20, cov[k20], "o", ms=8, color=INK, zorder=5)
    if en:
        ann2 = "promised: 20%\ndelivered: " + f"{cov[k20] * 100:.1f}%"
    else:
        ann2 = ("prometido: 20%\ncumprido: " +
                f"{cov[k20] * 100:.1f}%".replace(".", ","))
    a2.annotate(ann2,
                (0.20, cov[k20]), xytext=(14, -6), textcoords="offset points",
                fontsize=12, color=INK)
    a2.text(0.68, 0.56,
            "perfect calibration\n(diagonal)" if en else "calibração perfeita\n(diagonal)",
            fontsize=12, color="#999999", rotation=38, rotation_mode="anchor")
    if en:
        a2.set_xlabel("nominal level $\\tau$")
        a2.set_ylabel("empirical coverage $\\hat{P}(y \\leq q_\\tau)$")
        a2.set_title("(b) The fraction of targets below each quantile",
                     fontsize=13, loc="left")
    else:
        a2.set_xlabel("nível nominal $\\tau$")
        a2.set_ylabel("cobertura empírica $\\hat{P}(y \\leq q_\\tau)$")
        from matplotlib.ticker import FuncFormatter
        _fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        for _ax in (a1, a2):
            _ax.xaxis.set_major_formatter(_fmt)
            _ax.yaxis.set_major_formatter(_fmt)
        a2.set_title("(b) A fração de alvos abaixo de cada quantil",
                     fontsize=13, loc="left")
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1)
    print("cobertura em tau=0,20:", f"{cov[k20]:.4f}")

    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout()
    if en:
        outs = [REPO / "dissertacao-en/images/quantis_cobertura_diss_en",
                REPO / "results/figures/quantis_cobertura_diss_en"]
    else:
        outs = [REPO / "results/figures/quantis_cobertura_diss"]
    for out in outs:
        fig.savefig(f"{out}.png", dpi=180)
        fig.savefig(f"{out}.pdf")
        print(f"OK: {out}.png/.pdf")


if __name__ == "__main__":
    main()
