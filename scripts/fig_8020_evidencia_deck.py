"""Figura de deck: evidencia medida dos dois lados do corte 80:20.

Regenera `novo_8020_evidencia.png` com os rotulos corrigidos na auditoria
de 12/11/2026: o painel direito declara o escopo real das curvas de
aprendizado (6 modelos x 5 conjuntos, partida em 40-50% do pool), em vez
de "14 modelos x 18 conjuntos" e "50%->100%".

Fonte unica: results/aggregated/holdout_split_evidence.csv
(linhas `ic_auc_holdout`: IC95 bootstrap da AUC no hold-out;
linhas `melhora_rel_50_100`: metrica primaria do ponto da grade mais
proximo de metade do tamanho maximo por modelo -> tamanho maximo).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "results" / "aggregated" / "holdout_split_evidence.csv"
OUT = ROOT / "results" / "figures" / "novo_8020_evidencia.png"

BLUE = "#2878d0"
GRAY = "#8e8e89"


def main() -> None:
    df = pd.read_csv(EVID)
    ic = df[df.tipo == "ic_auc_holdout"].set_index("dataset")
    rel = df[df.tipo == "melhora_rel_50_100"].copy()
    rel["pct"] = 100 * rel.melhora_rel
    med = rel.pct.median()
    top = rel.loc[rel.pct.idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.6))
    ax = axes[0]
    n_adult = int(ic.loc["adult", "n_test"])
    n_credit = int(ic.loc["credit_g", "n_test"])
    names = [f"adult\n({n_adult:,} linhas de teste)".replace(",", "."),
             f"credit_g\n({n_credit} linhas — piso da grade)"]
    vals = [ic.loc["adult", "meia_largura"], ic.loc["credit_g", "meia_largura"]]
    bars = ax.barh(names, vals, color=[BLUE, GRAY], height=0.5)
    for b, v in zip(bars, vals):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2,
                ("±" + f"{v:.3f}").replace(".", ","),
                va="center", fontsize=15, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("meia-largura do IC95 bootstrap da AUC no hold-out",
                  fontsize=12)
    ax.set_title("Lado do teste: 20% já mede com precisão\n"
                 "(e o conjunto pequeno está no limite)",
                 loc="left", fontsize=14, color="#444444")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    n_models = rel.modelo.nunique()
    n_ds = rel.dataset.nunique()
    labels = [f"mediana\n({n_models} modelos × {n_ds} conjuntos)",
              f"máximo\n({top.modelo} / {top.dataset})"]
    bars = ax.bar(labels, [med, top.pct], color=[BLUE, GRAY], width=0.5)
    for b, v in zip(bars, [med, top.pct]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08,
                f"+{v:.1f}%".replace(".", ","), ha="center",
                fontsize=15, fontweight="bold")
    ax.set_ylim(0, top.pct * 1.2)
    ax.set_ylabel("melhora relativa da métrica primária (%)", fontsize=12)
    ax.set_title("Lado do treino: dobrar o treino (40–50%→100%)\n"
                 "quase não melhora — a região do corte é plana",
                 loc="left", fontsize=14, color="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)

    fig.suptitle("Por que 80:20 — evidência medida dos dois lados do corte",
                 x=0.01, ha="left", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}; mediana {med:.3f}% max {top.pct:.3f}% "
          f"({top.modelo}/{top.dataset}); IC adult "
          f"±{ic.loc['adult', 'meia_largura']:.4f} credit_g "
          f"±{ic.loc['credit_g', 'meia_largura']:.4f}")


if __name__ == "__main__":
    main()
