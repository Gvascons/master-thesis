#!/usr/bin/env python3
"""Display-name (English, article-grade) variants of the four AI-1 figures
plus the updated @14 decision flowchart.

Mirrors the canonical notebook generators (02/03/05/10 and 09) exactly —
same data, same aggregation — changing only cosmetics: model display names
(TabFM, FT-Transformer, ...), no embedded suptitles (the LaTeX captions
carry them), and label de-collision in the Pareto scatter. Canonical
figures used by the deck are untouched.

Outputs: results/figures/{cd_diagram_binary,average_ranks,pareto_binary,
learning_curves}_display.{png,pdf} and decision_flowchart_14.{png,pdf}.

Usage: uv run python scripts/figures_ai1_display.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FIGDIR = REPO / "results" / "figures"

from src.evaluation.statistical_tests import plot_cd_diagram  # noqa: E402

DISPLAY = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "mlp": "MLP", "realmlp": "RealMLP", "ft_transformer": "FT-Transformer",
    "saint": "SAINT", "stab": "STab", "tabnet": "TabNet", "tabm": "TabM",
    "kan": "KAN", "tabkan": "TabKAN", "tabpfn": "TabPFN", "tabfm": "TabFM",
}
FAMILIES = {
    "xgboost": "GBDT", "lightgbm": "GBDT", "catboost": "GBDT",
    "ft_transformer": "DL", "tabnet": "DL", "saint": "DL", "stab": "DL",
    "tabm": "DL", "mlp": "DL", "realmlp": "DL", "kan": "DL", "tabkan": "DL",
    "tabpfn": "FM", "tabfm": "FM",
}
FAMILY_COLORS = {"GBDT": "#2196F3", "DL": "#FF9800", "FM": "#4CAF50"}

TASKS = [("binary", "roc_auc", True), ("multiclass", "log_loss", False),
         ("regression", "rmse", False)]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"salva: {name}.png|pdf")


def fig_cd(test_df):
    sub = test_df[test_df.task_type == "binary"]
    pivot = sub.pivot(index="dataset", columns="model", values="roc_auc")
    pivot = pivot.rename(columns=DISPLAY)
    fig = plot_cd_diagram(pivot, title="", higher_is_better=True)
    save(fig, "cd_diagram_binary_display")
    plt.close(fig)


def fig_average_ranks_diss(test_df, lang):
    """Variante da DISSERTACAO (lote de diagramacao 25/11): mesmos
    valores e mesma ordem de barras da fig_average_ranks, paineis
    EMPILHADOS (3x1), figsize (7, 9), fontes 8.5pt (escala ~0.84 na
    pagina -> ~7.2pt impressos), fonttype 42; PT com virgula decimal.
    Nao substitui a average_ranks do AI-1."""
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    L = {"pt": {"xlabel": "rank médio (menor = melhor)",
                "titles": {"binary": "Binária (AUC-ROC)",
                           "multiclass": "Multiclasse (log-loss)",
                           "regression": "Regressão (RMSE)"},
                "fmt": lambda v: f"{v:.2f}".replace(".", ",")},
         "en": {"xlabel": "average rank (lower = better)",
                "titles": {"binary": "Binary (AUC-ROC)",
                           "multiclass": "Multiclass (log-loss)",
                           "regression": "Regression (RMSE)"},
                "fmt": lambda v: f"{v:.2f}"}}[lang]
    fig, axes = plt.subplots(3, 1, figsize=(7, 9))
    for ax, (task, metric, higher) in zip(axes, TASKS):
        sub = test_df[test_df.task_type == task]
        pivot = (sub.pivot(index="dataset", columns="model", values=metric)
                 .dropna(axis=1))
        ranks = pivot.rank(axis=1, ascending=not higher)
        avg = ranks.mean(axis=0).sort_values()
        avg.index = [DISPLAY.get(m, m) for m in avg.index]
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(avg)))
        bars = ax.barh(avg.index, avg.values, color=colors)
        ax.set_xlabel(L["xlabel"], fontsize=8.5)
        ax.set_title(L["titles"][task], fontsize=9.5, loc="left")
        ax.tick_params(labelsize=8.5)
        ax.invert_yaxis()
        for bar, val in zip(bars, avg.values):
            ax.text(val + 0.07, bar.get_y() + bar.get_height() / 2,
                    L["fmt"](val), va="center", fontsize=8.5)
        ax.set_xlim(0, max(avg.values) * 1.12)
    plt.tight_layout()
    out = (REPO / ("dissertacao/images/average_ranks_diss_pt.pdf" if lang == "pt"
                   else "dissertacao-en/images/average_ranks_diss_en.pdf"))
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIGDIR / out.name, bbox_inches="tight")
    print("salva:", out.name)
    plt.close(fig)


def fig_average_ranks(test_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (task, metric, higher) in zip(axes, TASKS):
        sub = test_df[test_df.task_type == task]
        pivot = (sub.pivot(index="dataset", columns="model", values=metric)
                 .dropna(axis=1))
        ranks = pivot.rank(axis=1, ascending=not higher)
        avg = ranks.mean(axis=0).sort_values()
        avg.index = [DISPLAY.get(m, m) for m in avg.index]
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(avg)))
        bars = ax.barh(avg.index, avg.values, color=colors)
        ax.set_xlabel("Average rank (lower = better)")
        nice = {"roc_auc": "AUC", "log_loss": "log-loss", "rmse": "RMSE"}
        ax.set_title(f"{task.title()} ({nice[metric]})")
        ax.invert_yaxis()
        for bar, val in zip(bars, avg.values):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    save(fig, "average_ranks_display")
    plt.close(fig)


def _pareto_mask(times, perfs, higher):
    n = len(times)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            faster = times[j] < times[i]
            geq = perfs[j] >= perfs[i] if higher else perfs[j] <= perfs[i]
            if faster and geq:
                keep[i] = False
                break
    return keep


def fig_pareto(test_df):
    # performance = mean binary RANK (the paper's primary instrument);
    # cost = MEDIAN training time (matching the text's ~300x claim)
    sub = test_df[test_df.task_type == "binary"]
    ranks = (sub.pivot(index="dataset", columns="model", values="roc_auc")
             .rank(axis=1, ascending=False))
    ms = pd.DataFrame({"model": ranks.columns,
                       "performance": ranks.mean().values})
    ms["train_time_s"] = ms.model.map(
        sub.groupby("model")["train_time_s"].median())
    ms["family"] = ms.model.map(FAMILIES)
    ms["pareto"] = _pareto_mask(ms.train_time_s.values,
                                ms.performance.values, higher=False)
    # manual annotation offsets/alignment where points or edges collide
    OFF = {"kan": (8, -2), "tabkan": (-8, -2), "mlp": (6, -12),
           "realmlp": (6, -12), "lightgbm": (-8, -4), "stab": (-8, 4),
           "saint": (0, -16), "ft_transformer": (6, 6),
           "tabnet": (-8, 2)}
    HA = {"lightgbm": "right", "stab": "right", "tabkan": "right",
          "tabnet": "right", "saint": "center"}
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, r in ms.iterrows():
        ax.scatter(r.train_time_s, r.performance,
                   color=FAMILY_COLORS[r.family],
                   marker="*" if r.pareto else "o",
                   s=250 if r.pareto else 80, alpha=0.9, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(DISPLAY[r.model], (r.train_time_s, r.performance),
                    textcoords="offset points",
                    xytext=OFF.get(r.model, (6, 4)),
                    ha=HA.get(r.model, "left"), fontsize=9)
    pp = ms[ms.pareto].sort_values("train_time_s")
    if len(pp) >= 2:
        ax.step(pp.train_time_s, pp.performance, where="post", ls="--",
                color="red", lw=1.5, zorder=1)
    ax.set_xscale("log")
    ax.invert_yaxis()  # lower rank = better, plotted upward: up-left wins
    ax.set_xlabel("Median training time (s, log scale)")
    ax.set_ylabel("Mean binary rank (lower = better; axis inverted)")
    ax.grid(alpha=0.3, which="both")
    handles = [Line2D([0], [0], marker="o", ls="", color=c, ms=9, label=f)
               for f, c in FAMILY_COLORS.items()]
    handles.append(Line2D([0], [0], marker="*", ls="", color="#555", ms=14,
                          label="Pareto-optimal"))
    if len(pp) >= 2:
        handles.append(Line2D([0], [0], ls="--", color="red",
                              label="Pareto front"))
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="lower left")
    save(fig, "pareto_binary_display")
    plt.close(fig)


def fig_learning_curves():
    df = pd.read_csv(REPO / "results/learning_curve/curve.csv")
    PRIMARY = {"binary": ("roc_auc", True), "multiclass": ("log_loss", False),
               "regression": ("rmse", False)}
    df["primary"] = df.apply(lambda r: r[PRIMARY[r.task_type][0]], axis=1)
    CMAP = {"xgboost": "#c1121f", "catboost": "#e8590c", "tabm": "#1d4e89",
            "realmlp": "#4895ef", "ft_transformer": "#3a0ca3",
            "tabpfn": "#2a9d8f"}

    def agg(d):
        g = d.groupby(["model", "family", "train_size"])["primary"]
        return (g.mean().rename("mean").reset_index()
                .merge(g.std().rename("std").reset_index(),
                       on=["model", "family", "train_size"]))

    datasets = [x for x in ("give_me_some_credit", "higgs", "adult",
                            "jannis", "year_prediction")
                if x in df.dataset.unique()]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        sub = df[df.dataset == ds]
        task = sub.task_type.iloc[0]
        col, higher = PRIMARY[task]
        a = agg(sub)
        for mdl in sub.model.unique():
            m = a[a.model == mdl].sort_values("train_size")
            ax.plot(m.train_size, m["mean"], "-o", ms=3, lw=1.8,
                    color=CMAP.get(mdl, "gray"))
            ax.fill_between(m.train_size, m["mean"] - m["std"],
                            m["mean"] + m["std"],
                            color=CMAP.get(mdl, "gray"), alpha=0.12)
        ax.set_xscale("log")
        if not higher:
            ax.set_yscale("log")
            yl = " (log)"
        else:
            yl = ""
        nice = {"roc_auc": "AUC", "log_loss": "log-loss", "rmse": "RMSE"}
        arrow = " ↑" if higher else " ↓"
        ax.set_title(f"{ds}\n({task}, {nice[col]}{arrow}{yl})", fontsize=10)
        ax.set_xlabel("training pool size")
        ax.grid(alpha=0.3, which="both")
    np.atleast_1d(axes)[0].set_ylabel("primary metric")
    handles = [Line2D([0], [0], color=c, lw=2, marker="o", ms=4)
               for c in CMAP.values()]
    fig.legend(handles, [DISPLAY[m] for m in CMAP], loc="upper center",
               ncol=6, bbox_to_anchor=(0.5, 1.09), frameon=False)
    plt.tight_layout()
    save(fig, "learning_curves_display")
    plt.close(fig)


def fig_flowchart14_pt():
    """Variante PT-BR do fluxograma @14 (dissertacao, Fig. 6.x).
    Layout refeito no lote de diagramacao de 25/11: figura maior,
    perguntas a direita, resultados a esquerda, sem sobreposicao."""
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.axis("off")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0.03, 1.03)

    def box(x, y, t, c="#eef"):
        ax.annotate(t, (x, y), ha="center", va="center", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.45", fc=c, ec="#333"))

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333"))
        if lbl:
            ax.text((x1 + x2) / 2 + 0.012, (y1 + y2) / 2, lbl, fontsize=12,
                    color="#a00", ha="left",
                    bbox=dict(fc="white", ec="none", pad=1.0))

    box(0.58, 0.985, "Problema tabular supervisionado")
    box(0.58, 0.86, "1. Or\u00e7amento de lat\u00eancia de servi\u00e7o cr\u00edtico?\n"
                   "(FMs custam 4\u20135 ordens de magnitude mais por linha)")
    arrow(0.58, 0.955, 0.58, 0.905)
    box(0.17, 0.68, "Fam\u00edlia GBDT\n(XGBoost / LightGBM / CatBoost)\n"
                    "em regress\u00e3o distribucional, a destila\u00e7\u00e3o\n"
                    "recupera parte da vantagem do TabPFN (Cap. 7)",
        "#fde")
    arrow(0.40, 0.835, 0.23, 0.745, "sim")
    box(0.58, 0.62, "2. Limites arquiteturais atingidos?\n"
                   "(>10 classes; teto de contexto)")
    arrow(0.58, 0.815, 0.58, 0.665, "n\u00e3o")
    box(0.17, 0.46, "GBDT ou DL\n(conforme a tarefa; ver revers\u00f5es)", "#fde")
    arrow(0.40, 0.60, 0.23, 0.50, "sim")
    box(0.58, 0.40, "3. Explica\u00e7\u00f5es audit\u00e1veis exigidas?\n(regula\u00e7\u00e3o; atribui\u00e7\u00e3o post-hoc)")
    arrow(0.58, 0.575, 0.58, 0.445, "n\u00e3o")
    box(0.16, 0.24, "GBDT\n(explic\u00e1vel por atribui\u00e7\u00e3o,\naceito em pr\u00e1tica regulada)", "#fde")
    arrow(0.40, 0.38, 0.23, 0.295, "sim")
    box(0.54, 0.12, "4. Nenhuma restri\u00e7\u00e3o ativa \u2192 FOUNDATION MODEL PRIMEIRO\n"
                   "TabPFN (maduro) \u00b7 TabFM (fronteira, ressalvas v1.0.1)\n"
                   "regret mediano 0,000 com 14 modelos (descritivo; execut\u00e1vel:\n"
                   "sempre-TabFM 0,000 mediana / 0,080 m\u00e9dia; sempre-TabPFN 0,242 mediana)",
        "#efe")
    arrow(0.58, 0.355, 0.56, 0.195, "n\u00e3o")
    save(fig, "decision_flowchart_14_pt")
    plt.close(fig)


def fig_flowchart14(ref="AI-2", out="decision_flowchart_14"):
    """Updated practitioner flowchart: the evaluated 4-constraint framework
    at 14 models (AI-1 section 5.3), replacing the pre-2026 @11 version.
    ref: label for the distillation cross-reference ("AI-2" for the paper,
    "Ch. 7" for the EN dissertation, saved as decision_flowchart_14_en).
    Layout redone in the 2026-11-25 typographic pass (no box overlaps)."""
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.axis("off")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0.03, 1.03)

    def box(x, y, t, c="#eef"):
        ax.annotate(t, (x, y), ha="center", va="center", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.45", fc=c, ec="#333"))

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333"))
        if lbl:
            ax.text((x1 + x2) / 2 + 0.012, (y1 + y2) / 2, lbl, fontsize=12,
                    color="#a00", ha="left",
                    bbox=dict(fc="white", ec="none", pad=1.0))

    box(0.58, 0.985, "Tabular supervised problem")
    box(0.58, 0.86, "1. Serving-latency budget critical?\n"
                   "(FMs cost 4\u20135 orders of magnitude more per row)")
    arrow(0.58, 0.955, 0.58, 0.905)
    box(0.17, 0.68, "GBDT family\n(XGBoost / LightGBM / CatBoost)\n"
                    "in distributional regression, distillation\n"
                    f"recovers part of the TabPFN edge ({ref})",
        "#fde")
    arrow(0.40, 0.835, 0.23, 0.745, "yes")
    box(0.58, 0.62, "2. Architectural limits hit?\n"
                   "(>10 classes; context cap)")
    arrow(0.58, 0.815, 0.58, 0.665, "no")
    box(0.17, 0.46, "GBDT or DL\n(task-matched; see reversals)", "#fde")
    arrow(0.40, 0.60, 0.23, 0.50, "yes")
    box(0.58, 0.40, "3. Auditable explanations required?\n(regulation; post-hoc attribution)")
    arrow(0.58, 0.575, 0.58, 0.445, "no")
    box(0.16, 0.24, "GBDT\n(attribution-explainable,\naccepted in regulated practice)", "#fde")
    arrow(0.40, 0.38, 0.23, 0.295, "yes")
    box(0.54, 0.12, "4. No active constraint \u2192 FOUNDATION MODEL FIRST\n"
                   "TabPFN (mature) \u00b7 TabFM (frontier, v1.0.1 caveats)\n"
                   "median normalized regret 0.000 @14 models (descriptive; executable:\n"
                   "always-TabFM 0.000 median / 0.080 mean; always-TabPFN 0.242 median)",
        "#efe")
    arrow(0.58, 0.355, 0.56, 0.195, "no")
    save(fig, out)
    plt.close(fig)


def main():
    test_df = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    fig_cd(test_df)
    fig_average_ranks(test_df)
    fig_pareto(test_df)
    fig_learning_curves()
    fig_flowchart14()
    fig_flowchart14(ref="Ch. 7", out="decision_flowchart_14_en")
    fig_flowchart14_pt()


if __name__ == "__main__":
    main()
