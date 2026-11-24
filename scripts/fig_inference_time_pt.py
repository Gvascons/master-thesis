"""Regenera a figura de latência de inferência (Fig. 5.3 da dissertação)
em português e SEM o subtítulo editorial ("~1000x+ slower"), que o corpo
do texto desmente (os fatores medidos são 22.473x e 131.097x).

Fonte de dados: results/latency/latency_adult.csv (medição padronizada).
Saída: dissertacao/images/inference_time.pdf (+ results/figures/ cópia).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def ptnum(v):
    s = f"{v:,.1f}" if v < 1000 else f"{v:,.0f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def main():
    lat = pd.read_csv(REPO / "results/latency/latency_adult.csv")
    lat = lat.sort_values("us_per_row")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = lat["family"].map({"gbdt": "#2196F3", "deep_learning": "#FF9800",
                                "foundation_model": "#4CAF50"}).fillna("grey")
    ax.barh(lat["model"], lat["us_per_row"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Latência de inferência (µs/linha, escala logarítmica)")
    ax.set_title("Latência de inferência no conjunto adult (menor é melhor)")
    for y, v in enumerate(lat["us_per_row"]):
        ax.text(v, y, f" {ptnum(v)}", va="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2196F3", label="GBDT"),
                       Patch(color="#FF9800", label="Aprendizado profundo"),
                       Patch(color="#4CAF50", label="Foundation model")],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    for out in (REPO / "dissertacao/images/inference_time.pdf",
                REPO / "results/figures/inference_time.pdf"):
        plt.savefig(out, bbox_inches="tight")
    plt.savefig(REPO / "results/figures/inference_time.png", dpi=300,
                bbox_inches="tight")
    print("ok")


if __name__ == "__main__":
    main()
