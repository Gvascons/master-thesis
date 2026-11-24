"""Gera o diagrama de distância crítica do NÚCLEO de 11 modelos para o
Ato I do cap. 5 (a figura da grade de 14 fica no Ato IV). Ranks
recomputados no recorte; nomes de exibição em português/como no texto.

Fonte: results/aggregated/test_results.csv
Saída: dissertacao/images/cd_diagram_binary_11.pdf (+ results/figures/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.evaluation.statistical_tests import plot_cd_diagram, create_cd_diagram_data

DISPLAY = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "mlp": "MLP", "realmlp": "RealMLP", "ft_transformer": "FT-Transformer",
    "saint": "SAINT", "stab": "STab", "tabnet": "TabNet", "tabm": "TabM",
    "tabpfn": "TabPFN",
}

def main():
    t = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    b = t[t.task_type == "binary"]
    piv = b.pivot(index="dataset", columns="model", values="roc_auc")
    core = [c for c in piv.columns if c not in ("tabfm", "kan", "tabkan")]
    scores = piv[core].rename(columns=DISPLAY)
    data = create_cd_diagram_data(scores, True)
    print("CD (k=11, N=10):", round(data["cd"], 2))
    print({k: round(v, 2) for k, v in sorted(data["average_ranks"].items(), key=lambda x: x[1])})
    plot_cd_diagram(scores, title="", higher_is_better=True,
                    save_path=str(REPO / "dissertacao/images/cd_diagram_binary_11.pdf"))
    plot_cd_diagram(scores, title="", higher_is_better=True,
                    save_path=str(REPO / "results/figures/cd_diagram_binary_11.pdf"))

if __name__ == "__main__":
    main()
