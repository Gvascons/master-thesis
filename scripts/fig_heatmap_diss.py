"""Regenera o mapa de calor da classificação (Fig. 5.2) com cor informativa:
anotação = valor da métrica primária (AUC nas binárias, log-loss nas
multiclasse), cor = rank do modelo DENTRO do conjunto (1 = melhor, mais
escuro), de modo que a cor informa na mesma convenção em todas as linhas.
Colunas agrupadas por família e ordenadas pelo rank médio; nomes como no
texto; células vazias = exclusões estruturais (helena).

Fonte: results/aggregated/test_results.csv
Saída: dissertacao/images/heatmap_classification_pt.pdf (+ results/figures/)
Com argumento "en": rótulos em inglês, ponto decimal, saída
dissertacao-en/images/heatmap_classification_en.pdf (não toca na PT).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DISPLAY = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "mlp": "MLP", "realmlp": "RealMLP", "ft_transformer": "FT-Transf.",
    "saint": "SAINT", "stab": "STab", "tabnet": "TabNet", "tabm": "TabM",
    "kan": "KAN", "tabkan": "TabKAN", "tabpfn": "TabPFN", "tabfm": "TabFM",
}
FAMORD = {"xgboost": 0, "lightgbm": 0, "catboost": 0,
          "mlp": 1, "realmlp": 1, "ft_transformer": 1, "saint": 1,
          "stab": 1, "tabnet": 1, "tabm": 1, "kan": 1, "tabkan": 1,
          "tabpfn": 2, "tabfm": 2}

def main():
    en = len(sys.argv) > 1 and sys.argv[1] == "en"
    t = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    cls = t[t.task_type.isin(["binary", "multiclass"])].copy()
    cls["val"] = np.where(cls.task_type == "binary", cls.roc_auc, cls.log_loss)
    piv = cls.pivot(index="dataset", columns="model", values="val")
    task = cls.groupby("dataset").task_type.first()
    # rank dentro da linha (1 = melhor): AUC desc, log-loss asc
    ranks = piv.copy()
    for ds in piv.index:
        asc = task[ds] != "binary"
        ranks.loc[ds] = piv.loc[ds].rank(ascending=asc)
    # ordenar linhas: binárias primeiro (alfabético), depois multiclasse
    rows = sorted(task[task == "binary"].index) + sorted(task[task == "multiclass"].index)
    # colunas: família, depois rank médio
    mean_rank = ranks.mean()
    cols = sorted(piv.columns, key=lambda m: (FAMORD[m], mean_rank[m]))
    piv, ranks = piv.loc[rows, cols], ranks.loc[rows, cols]
    fig, ax = plt.subplots(figsize=(12.5, 7))
    # cor: rank 1 escuro -> 14 claro
    im = ax.imshow(ranks.values.astype(float), cmap="Greens_r", vmin=1, vmax=14,
                   aspect="auto")
    for i, ds in enumerate(rows):
        for j, m in enumerate(cols):
            v = piv.iloc[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#777")
                continue
            r = ranks.iloc[i, j]
            ax.text(j, i, f"{v:.3f}" if en else f"{v:.3f}".replace(".", ","), ha="center",
                    va="center", fontsize=7.2,
                    color="white" if r <= 4 else "#1b1b1b")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([DISPLAY[m] for m in cols], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r + (" (log-loss)" if task[r] != "binary" else "")
                        for r in rows], fontsize=9)
    ax.axvline(2.5, color="white", lw=2)
    ax.axvline(11.5, color="white", lw=2)
    ax.axhline(9.5, color="white", lw=2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    if en:
        cbar.set_label("Within-dataset rank (1 = best)")
        ax.set_title("Primary metric per model and dataset "
                     "(AUC on binary; log-loss on multiclass); color = within-row rank",
                     fontsize=10)
        outs = (REPO / "dissertacao-en/images/heatmap_classification_en.pdf",
                REPO / "results/figures/heatmap_classification_en.pdf")
    else:
        cbar.set_label("Rank dentro do conjunto (1 = melhor)")
        ax.set_title("Métrica primária por modelo e conjunto "
                     "(AUC nas binárias; log-loss nas multiclasse); cor = rank na linha",
                     fontsize=10)
        outs = (REPO / "dissertacao/images/heatmap_classification_pt.pdf",
                REPO / "results/figures/heatmap_classification_pt.pdf")
    plt.tight_layout()
    for out in outs:
        plt.savefig(out, bbox_inches="tight")
    print("ok")

if __name__ == "__main__":
    main()
