"""Regenera a figura de latência de inferência (Fig. 5.3 da dissertação)
em português e SEM o subtítulo editorial ("~1000x+ slower"), que o corpo
do texto desmente (os fatores medidos são 22.473x e 131.097x).

Fonte de dados: results/latency/latency_adult.csv (medição padronizada).
Saída: dissertacao/images/inference_time.pdf (+ results/figures/ cópia).
Com argumento "en": rótulos em inglês, ponto decimal, saída
dissertacao-en/images/inference_time_en.pdf (PT intocada).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


EN = len(sys.argv) > 1 and sys.argv[1] == "en"


def ptnum(v):
    s = f"{v:,.1f}" if v < 1000 else f"{v:,.0f}"
    if EN:
        return s
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


DISPLAY = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "mlp": "MLP", "resnet": "ResNet", "ft_transformer": "FT-Transformer",
    "realmlp": "RealMLP", "tabm": "TabM", "stab": "STab", "kan": "KAN",
    "tabkan": "TabKAN", "tabnet": "TabNet", "saint": "SAINT",
    "tabpfn": "TabPFN", "tabfm": "TabFM",
}


def main():
    lat = pd.read_csv(REPO / "results/latency/latency_adult.csv")
    lat = lat.sort_values("us_per_row")
    lat["model"] = lat["model"].map(lambda m: DISPLAY.get(m, m))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = lat["family"].map({"gbdt": "#2196F3", "deep_learning": "#FF9800",
                                "foundation_model": "#4CAF50"}).fillna("grey")
    ax.barh(lat["model"], lat["us_per_row"], color=colors)
    ax.set_xscale("log")
    if EN:
        ax.set_xlabel("Inference latency (µs/row, log scale)")
        ax.set_title("Inference latency on the adult dataset (lower is better)")
        dl_label = "Deep learning"
    else:
        ax.set_xlabel("Latência de inferência (µs/linha, escala logarítmica)")
        ax.set_title("Latência de inferência no conjunto adult (menor é melhor)")
        dl_label = "Aprendizado profundo"
    for y, v in enumerate(lat["us_per_row"]):
        ax.text(v, y, f" {ptnum(v)}", va="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2196F3", label="GBDT"),
                       Patch(color="#FF9800", label=dl_label),
                       Patch(color="#4CAF50", label="Foundation model")],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    if EN:
        outs = (REPO / "dissertacao-en/images/inference_time_en.pdf",
                REPO / "results/figures/inference_time_en.pdf")
    else:
        outs = (REPO / "dissertacao/images/inference_time.pdf",
                REPO / "results/figures/inference_time.pdf")
    for out in outs:
        plt.savefig(out, bbox_inches="tight")
    if not EN:
        plt.savefig(REPO / "results/figures/inference_time.png", dpi=300,
                    bbox_inches="tight")
    print("ok")


if __name__ == "__main__":
    main()
