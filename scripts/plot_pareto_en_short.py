"""Variante -short do plot_pareto.py (B21, 27/11/2026): estrela do
destilado com zorder acima do losango do controle (no painel RMSE do
kin8nm os dois quase coincidem e a estrela ficava oculta) e marcadores
um ponto maiores. Saida SOMENTE em dissertacao-en-short/images/
(longa e PT intactas). Rodar: uv run python scripts/plot_pareto_en_short.py --lang en
"""
import sys
from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "distillation" / "pareto.csv"
FIGDIR = REPO / "dissertacao-en-short" / "images"

# fixed identity order (never re-sorted): color + marker per strategy
LABELS = {
    "pt": {
        "teacher_ctx": "TabPFN (contexto reduzido)",
        "teacher_ens": "TabPFN (ensemble reduzido)",
        "student_quant_soft": "Aluno-quantil DESTILADO",
        "student_point_hard": "Aluno ponto (controle)",
        "student_quant_hard": "Aluno-quantil (controle)",
        "tabfm_full": "TabFM (contexto cheio)",
        "annot": "destilado",
        "xlabel": "lat\u00eancia de infer\u00eancia (\u00b5s/linha, log)",
        "rmse": "RMSE \u2193", "crps": "CRPS \u2193 (sistemas distribucionais)",
        "suptitle": ("Destilar \u00d7 comprimir contexto \u00d7 reduzir ensemble \u2014 "
                     "a fronteira acur\u00e1cia \u00d7 lat\u00eancia (hold-out do benchmark)"),
        "suffix": "",
    },
    "pt_diss": {
        "teacher_ctx": "TabPFN (contexto reduzido)",
        "teacher_ens": "TabPFN (ensemble reduzido)",
        "student_quant_soft": "Aluno-quantil DESTILADO",
        "student_point_hard": "Aluno ponto (controle)",
        "student_quant_hard": "Aluno-quantil (controle)",
        "tabfm_full": "TabFM (contexto cheio)",
        "annot": "destilado",
        "xlabel": "lat\u00eancia de infer\u00eancia (\u00b5s/linha, log)",
        "rmse": "RMSE \u2193", "crps": "CRPS \u2193 (sistemas distribucionais)",
        "suptitle": None,  # a legenda LaTeX carrega o titulo
        "suffix": "_pt",
    },
    "en": {
        "teacher_ctx": "TabPFN (reduced context)",
        "teacher_ens": "TabPFN (reduced ensemble)",
        "student_quant_soft": "DISTILLED quantile student",
        "student_point_hard": "Point student (control)",
        "student_quant_hard": "Quantile student (control)",
        "tabfm_full": "TabFM (full context)",
        "annot": "distilled",
        "xlabel": "inference latency (\u00b5s/row, log)",
        "rmse": "RMSE \u2193", "crps": "CRPS \u2193 (distributional systems)",
        "suptitle": None,  # the LaTeX caption carries the title
        "suffix": "_en",
    },
}
STYLE = {
    "teacher_ctx":        ("#2a78d6", "o", "-"),
    "teacher_ens":        ("#008300", "^", ""),
    "student_quant_soft": ("#e87ba4", "*", ""),
    "student_point_hard": ("#eda100", "s", ""),
    "student_quant_hard": ("#1baf7a", "D", ""),
    "tabfm_full":         ("#eb6834", "P", ""),
}


def facet(ax, sub, metric, L):
    for system, (color, marker, ls) in STYLE.items():
        label = L[system]
        s = sub[(sub.system == system) & sub[metric].notna()]
        if s.empty:
            continue
        s = s.sort_values("us_per_row")
        ms = 17 if marker == "*" else 9
        if ls:  # the context curve is a connected line
            ax.plot(s.us_per_row, s[metric], ls, color=color, lw=2,
                    marker=marker, ms=ms, label=label, zorder=3)
        else:
            ax.scatter(s.us_per_row, s[metric], c=color, marker=marker,
                       s=ms**2, label=label,
                       zorder=6 if marker == "*" else 4,
                       edgecolors="white",
                       linewidths=0.7 if marker == "*" else 1.5)
        if system == "student_quant_soft":
            for _, r in s.iterrows():
                ax.annotate(L["annot"], (r.us_per_row, r[metric]),
                            textcoords="offset points", xytext=(9, -15),
                            fontsize=14, color=color, fontweight="bold",
                            bbox=dict(fc="white", ec="none", pad=1.0,
                                      alpha=0.85))
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.tick_params(labelsize=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="pt", choices=["pt", "en", "pt_diss"])
    args = ap.parse_args()
    L = LABELS[args.lang]
    df = pd.read_csv(CSV)
    df["crps"] = pd.to_numeric(df["crps"], errors="coerce")
    datasets = [d for d in ("california_housing", "kin8nm", "year_prediction")
                if d in set(df.dataset)]
    n = len(datasets)
    plt.rcParams["pdf.fonttype"] = 42
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 7.4))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, ds in enumerate(datasets):
        sub = df[df.dataset == ds]
        facet(axes[0, j], sub, "rmse", L)
        axes[0, j].set_title(ds, fontsize=16)
        facet(axes[1, j], sub, "crps", L)
        axes[1, j].set_xlabel(L["xlabel"], fontsize=15)
    axes[0, 0].set_ylabel(L["rmse"], fontsize=15)
    axes[1, 0].set_ylabel(L["crps"], fontsize=15)

    if args.lang in ("pt_diss", "en"):
        import math
        from matplotlib.ticker import FuncFormatter, NullFormatter
        fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        for ax in axes.ravel():
            if args.lang == "pt_diss":
                ax.yaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_formatter(NullFormatter())
            # regra: todo eixo log com ao menos duas decadas rotuladas
            lo, hi = ax.get_xlim()
            if math.floor(math.log10(hi)) - math.ceil(math.log10(lo)) < 1:
                lo = 10 ** (math.floor(math.log10(hi)) - 1)
            ax.set_xlim(lo, hi * 1.35)  # folga p/ o marcador no extremo
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_y = 1.0 if L["suptitle"] else 1.135
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=15,
               frameon=False, bbox_to_anchor=(0.5, legend_y))
    if L["suptitle"]:
        fig.suptitle(L["suptitle"], fontsize=12, y=1.05)
    plt.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pareto_distill{L['suffix']}.{ext}", dpi=150,
                    bbox_inches="tight")
    print(f"figura salva: {FIGDIR}/pareto_distill.png|pdf "
          f"({n} datasets, {len(df)} pontos)")


if __name__ == "__main__":
    main()
