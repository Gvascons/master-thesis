"""Figura de deck: vitrine da destilacao (fps_benchmark + cpu_activity).

Regenera `novo_destilacao_vitrine.png` com os textos corrigidos na
auditoria de 12/11/2026:
  - titulo sem "de graca na inferencia" (o aluno destilado custa ~10^2 us/linha,
    um pouco acima do controle no pareto.csv);
  - rodape com a definicao correta CRPS = 2x a media das pinball (Eq. 2.9).
Fonte unica: results/distillation/extension.csv (medias de 10 sementes).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "results" / "distillation" / "extension.csv"
OUT = ROOT / "results" / "figures" / "novo_destilacao_vitrine.png"

DATASETS = ["fps_benchmark", "cpu_activity"]
COLORS = {"hard": "#8e8e89", "soft": "#2878d0", "teacher": "#2b2b2b"}
BLUE = "#2878d0"


def fmt(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def main() -> None:
    ext = pd.read_csv(EXT)
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.0))
    footer_52 = None
    for ax, ds in zip(axes, DATASETS):
        sub = ext[ext.dataset == ds]
        teacher = sub[sub.cell == "teacher"].crps.mean()
        hard = sub[sub.cell == "student_quant_hard"].crps.mean()
        soft = sub[sub.cell == "student_quant_soft"].crps.mean()
        ret = (hard - soft) / (hard - teacher)
        vals = [hard, soft, teacher]
        labels = [
            "Aluno-controle\n(XGBoost-quantil,\nrótulos verdadeiros)",
            "Aluno destilado\n(XGBoost-quantil,\nrótulos do professor)",
            "Professor\n(TabPFN\ndistribucional)",
        ]
        bars = ax.bar(labels, vals,
                      color=[COLORS["hard"], COLORS["soft"], COLORS["teacher"]],
                      width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012 * max(vals),
                    fmt(v), ha="center", va="bottom",
                    fontsize=15, fontweight="bold")
        ax.set_title(ds, loc="left", fontsize=16)
        ax.text(0.98, 0.97, f"{ret * 100:.0f}% do gap fechado",
                transform=ax.transAxes, ha="right", va="top",
                color=BLUE, fontsize=17, fontweight="bold")
        ax.set_ylabel("CRPS no hold-out (menor = melhor)", fontsize=12)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.tick_params(axis="x", labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if ds == "fps_benchmark":
            # base pareada do artigo (Tabela 7.1): sementes 0-2
            s3 = sub[sub.seed <= 2]
            h3 = s3[s3.cell == "student_quant_hard"].crps.mean()
            f3 = s3[s3.cell == "student_quant_soft"].crps.mean()
            footer_52 = (h3 - f3) / (h3 - teacher)
    fig.suptitle("Destilação — quanto do gap até o professor o aluno recupera "
                 "(na latência do tier XGBoost)",
                 x=0.01, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.01, 0.015,
             "Médias de 10 sementes; CRPS = régua distribucional "
             "(2× a média das perdas pinball na grade de quantis — Eq. 2.9).\n"
             f"Na base pareada do artigo (3 sementes), a retenção do fps é "
             f"{footer_52 * 100:.0f}% — mesma leitura: ~metade do gap. "
             "Latência do aluno: ~10² µs/linha (pareto.csv).",
             fontsize=11, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
