"""Regenera a Fig. 5.4 da dissertação em português e em base consistente
com a Tabela A.11: custo = tempo mediano de treino sobre os 18 conjuntos
(a figura do artigo usava a mediana só das binárias, o que invertia a
ordem XGBoost/LightGBM e contradizia a fronteira enunciada no texto).

Dois painéis: (a) rank binário médio x tempo mediano de treino;
(b) rank binário médio x latência de inferência (adult). Em cada painel,
fronteira do núcleo de 11 (tracejada, ranks recomputados no recorte) e da
grade completa de 14 (cheia).

Fontes: results/aggregated/test_results.csv, results/latency/latency_adult.csv
Saída:  dissertacao/images/pareto_binary_pt.pdf (+ results/figures/)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
DISPLAY = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "mlp": "MLP", "realmlp": "RealMLP", "ft_transformer": "FT-Transformer",
    "saint": "SAINT", "stab": "STab", "tabnet": "TabNet", "tabm": "TabM",
    "kan": "KAN", "tabkan": "TabKAN", "tabpfn": "TabPFN", "tabfm": "TabFM",
}
FAM = {
    "xgboost": "GBDT", "lightgbm": "GBDT", "catboost": "GBDT",
    "ft_transformer": "DL", "tabnet": "DL", "saint": "DL", "stab": "DL",
    "tabm": "DL", "mlp": "DL", "realmlp": "DL", "kan": "DL", "tabkan": "DL",
    "tabpfn": "FM", "tabfm": "FM",
}
CORES = {"GBDT": "#2196F3", "DL": "#FF9800", "FM": "#4CAF50"}
EXTENSAO = {"tabfm", "kan", "tabkan"}


def fronteira(df, xcol, ycol):
    """Pontos não dominados (menor x, menor y), ordenados por x."""
    pts = df.sort_values(xcol)
    front = []
    best = float("inf")
    for _, r in pts.iterrows():
        if r[ycol] < best:
            front.append(r)
            best = r[ycol]
    return pd.DataFrame(front)


def main():
    t = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    b = t[t.task_type == "binary"]
    piv = b.pivot(index="dataset", columns="model", values="roc_auc")
    rank14 = piv.rank(axis=1, ascending=False).mean()
    core = [c for c in piv.columns if c not in EXTENSAO]
    rank11 = piv[core].rank(axis=1, ascending=False).mean()
    treino = t.groupby("model")["train_time_s"].median()
    lat = pd.read_csv(REPO / "results/latency/latency_adult.csv").set_index("model")["us_per_row"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    for ax, (custo, xlabel, titulo) in zip(axes, [
        (treino, "Tempo mediano de treino (s, escala log)",
         "(a) rank binário × custo de treino"),
        (lat, "Latência de inferência (µs/linha, escala log; adult)",
         "(b) rank binário × latência"),
    ]):
        df = pd.DataFrame({"rank14": rank14, "custo": custo}).dropna()
        df["fam"] = [FAM[m] for m in df.index]
        f14 = fronteira(df.reset_index(), "custo", "rank14")
        core_df = pd.DataFrame({"rank": rank11, "custo": custo}).dropna()
        f11 = fronteira(core_df.reset_index(), "custo", "rank")
        # pertinência à fronteira do núcleo decidida nos ranks do recorte;
        # a linha é desenhada nas posições plotadas (rank14), para tocar
        # os pontos
        f11_plot = df.loc[list(f11["model"])].sort_values("custo")
        ax.step(f14.custo, f14.rank14, where="post", color="#424242",
                lw=1.4, zorder=1)
        ax.step(f11_plot.custo, f11_plot.rank14, where="post",
                color="#9e9e9e", lw=1.2, ls="--", zorder=1)
        for m, r in df.iterrows():
            na_f = m in set(f14["model"])
            ax.scatter(r.custo, r.rank14, color=CORES[r.fam],
                       marker="*" if na_f else "o",
                       s=230 if na_f else 70, zorder=3,
                       edgecolors="white", linewidths=0.7)
            OFF = {"saint": (5, -11), "ft_transformer": (5, 6),
                   "catboost": (5, -10), "lightgbm": (-8, -11),
                   "tabm": (5, 5)}
            HA = {"lightgbm": "right"}
            ax.annotate(DISPLAY[m], (r.custo, r.rank14), fontsize=8,
                        textcoords="offset points",
                        ha=HA.get(m, "left"),
                        xytext=OFF.get(m, (5, 4)))
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(titulo, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Rank binário médio (menor é melhor; eixo invertido)")
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=8, label=f)
               for f, c in CORES.items()]
    handles += [
        Line2D([], [], marker="*", ls="", color="#555", ms=13,
               label="na fronteira (grade de 14)"),
        Line2D([], [], ls="-", color="#424242", label="fronteira, grade de 14"),
        Line2D([], [], ls="--", color="#9e9e9e",
               label="fronteira, núcleo de 11 (ranks do recorte)"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, frameon=False,
                   loc="lower left")
    plt.tight_layout()
    for out in (REPO / "dissertacao/images/pareto_binary_pt.pdf",
                REPO / "results/figures/pareto_binary_pt.pdf"):
        plt.savefig(out, bbox_inches="tight")
    print("fronteira 14 (treino):", list(fronteira(
        pd.DataFrame({"rank14": rank14, "custo": treino}).dropna().reset_index(),
        "custo", "rank14")["model"]))
    core_df = pd.DataFrame({"rank": rank11, "custo": treino}).dropna()
    print("fronteira 11 (treino):", list(fronteira(core_df.reset_index(), "custo", "rank")["model"]))
    core_df = pd.DataFrame({"rank": rank11, "custo": lat}).dropna()
    print("fronteira 11 (latencia):", list(fronteira(core_df.reset_index(), "custo", "rank")["model"]))


if __name__ == "__main__":
    main()
