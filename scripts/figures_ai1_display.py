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
    """Variante PT-BR do fluxograma @14 (dissertacao, Fig. 6.x)."""
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.axis("off")

    def box(x, y, t, c="#eef"):
        ax.annotate(t, (x, y), ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc=c, ec="#333"))

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333"))
        if lbl:
            ax.text((x1 + x2) / 2 + 0.015, (y1 + y2) / 2, lbl, fontsize=9,
                    color="#a00", ha="left")

    box(0.5, 0.96, "Problema tabular supervisionado")
    box(0.5, 0.82, "1. Or\u00e7amento de lat\u00eancia de servi\u00e7o cr\u00edtico?\n"
                   "(FMs custam 4\u20135 ordens de magnitude mais por linha)")
    arrow(0.5, 0.93, 0.5, 0.87)
    box(0.13, 0.66, "Fam\u00edlia GBDT\n(XGBoost / LightGBM / CatBoost)\n"
                    "a destila\u00e7\u00e3o recupera parte\nda vantagem do FM (Cap. 7)",
        "#fde")
    arrow(0.36, 0.80, 0.17, 0.71, "sim")
    box(0.5, 0.62, "2. Limites arquiteturais atingidos?\n"
                   "(>10 classes; teto de contexto)")
    arrow(0.5, 0.77, 0.5, 0.67, "n\u00e3o")
    box(0.13, 0.46, "GBDT ou DL\n(conforme a tarefa; ver revers\u00f5es)", "#fde")
    arrow(0.36, 0.60, 0.17, 0.50, "sim")
    box(0.5, 0.42, "3. Interpretabilidade intr\u00ednseca exigida?\n(regula\u00e7\u00e3o)")
    arrow(0.5, 0.57, 0.5, 0.47, "n\u00e3o")
    box(0.13, 0.26, "GBDT\n(intrinsecamente interpret\u00e1vel)", "#fde")
    arrow(0.36, 0.40, 0.17, 0.30, "sim")
    box(0.5, 0.18, "4. Nenhuma restri\u00e7\u00e3o ativa \u2192 FOUNDATION MODEL PRIMEIRO\n"
                   "TabPFN (maduro) \u00b7 TabFM (fronteira, ressalvas v1.0.1)\n"
                   "regret normalizado mediano 0,000 @14 modelos (LODO)",
        "#efe")
    arrow(0.5, 0.37, 0.5, 0.25, "n\u00e3o")
    ax.set_ylim(0.08, 1.0)
    save(fig, "decision_flowchart_14_pt")
    plt.close(fig)


def fig_flowchart14():
    """Updated practitioner flowchart: the validated 4-constraint framework
    at 14 models (AI-1 section 5.3), replacing the pre-2026 @11 version."""
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.axis("off")

    def box(x, y, t, c="#eef"):
        ax.annotate(t, (x, y), ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc=c, ec="#333"))

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333"))
        if lbl:
            ax.text((x1 + x2) / 2 + 0.015, (y1 + y2) / 2, lbl, fontsize=9,
                    color="#a00", ha="left")

    box(0.5, 0.96, "Tabular supervised problem")
    box(0.5, 0.82, "1. Serving-latency budget critical?\n"
                   "(FMs cost 4–5 orders of magnitude more per row)")
    arrow(0.5, 0.93, 0.5, 0.87)
    box(0.13, 0.66, "GBDT family\n(XGBoost / LightGBM / CatBoost)\n"
                    "distillation can recover part\nof the FM edge (AI-2)",
        "#fde")
    arrow(0.36, 0.80, 0.17, 0.71, "yes")
    box(0.5, 0.62, "2. Architectural limits hit?\n"
                   "(>10 classes; context cap)")
    arrow(0.5, 0.77, 0.5, 0.67, "no")
    box(0.13, 0.46, "GBDT or DL\n(task-matched; see reversals)", "#fde")
    arrow(0.36, 0.60, 0.17, 0.50, "yes")
    box(0.5, 0.42, "3. Intrinsic interpretability required?\n(regulation)")
    arrow(0.5, 0.57, 0.5, 0.47, "no")
    box(0.13, 0.26, "GBDT\n(intrinsically interpretable)", "#fde")
    arrow(0.36, 0.40, 0.17, 0.30, "yes")
    box(0.5, 0.18, "4. No active constraint → FOUNDATION MODEL FIRST\n"
                   "TabPFN (mature) · TabFM (frontier, v1.0.1 caveats)\n"
                   "median normalized regret 0.000 @14 models (LODO)",
        "#efe")
    arrow(0.5, 0.37, 0.5, 0.25, "no")
    ax.set_ylim(0.08, 1.0)
    save(fig, "decision_flowchart_14")
    plt.close(fig)


def main():
    test_df = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    fig_cd(test_df)
    fig_average_ranks(test_df)
    fig_pareto(test_df)
    fig_learning_curves()
    fig_flowchart14()
    fig_flowchart14_pt()


if __name__ == "__main__":
    main()
