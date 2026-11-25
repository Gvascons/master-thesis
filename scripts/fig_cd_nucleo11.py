"""Gera os diagramas de distância crítica da dissertação PT com
vírgula decimal: o NÚCLEO de 11 modelos (Ato I do cap. 5) e a grade
completa de 14 (Ato IV, cd_diagram_binary_pt.pdf — a variante com
ponto decimal do AI-1/EN é a de figures_ai1_display.py e não é
tocada aqui). Ranks recomputados no recorte; nomes como no texto.

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
    en = len(sys.argv) > 1 and sys.argv[1] == "en"
    comma = not en
    base = REPO / ("dissertacao-en/images" if en else "dissertacao/images")
    b = t[t.task_type == "binary"]
    piv = b.pivot(index="dataset", columns="model", values="roc_auc")
    core = [c for c in piv.columns if c not in ("tabfm", "kan", "tabkan")]
    scores = piv[core].rename(columns=DISPLAY)
    data = create_cd_diagram_data(scores, True)
    print("CD (k=11, N=10):", round(data["cd"], 2))
    print({k: round(v, 2) for k, v in sorted(data["average_ranks"].items(), key=lambda x: x[1])})
    plot_cd_diagram(scores, title="", higher_is_better=True,
                    decimal_comma=comma, font_scale=1.4,
                    save_path=str(base / "cd_diagram_binary_11.pdf"))
    if not en:
        plot_cd_diagram(scores, title="", higher_is_better=True,
                        decimal_comma=True, font_scale=1.4,
                        save_path=str(REPO / "results/figures/cd_diagram_binary_11.pdf"))
    # grade completa de 14 (Ato IV), virgula decimal, saida _pt
    full = piv.rename(columns={**DISPLAY, "kan": "KAN", "tabkan": "TabKAN", "tabfm": "TabFM"})
    data14 = create_cd_diagram_data(full, True)
    print("CD (k=14, N=10):", round(data14["cd"], 2))
    out14 = "cd_diagram_binary.pdf" if en else "cd_diagram_binary_pt.pdf"
    plot_cd_diagram(full, title="", higher_is_better=True,
                    decimal_comma=comma, font_scale=1.4,
                    save_path=str(base / out14))
    if not en:
        plot_cd_diagram(full, title="", higher_is_better=True,
                        decimal_comma=True, font_scale=1.4,
                        save_path=str(REPO / "results/figures/cd_diagram_binary_pt.pdf"))

if __name__ == "__main__":
    main()
