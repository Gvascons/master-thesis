#!/usr/bin/env python3
"""Análise pré-registrada do E3 (docs/desenho-melhorias-finais.md,
14/10/2026): custo de decisão realizado por nível τ.

H-D1: quant_soft <= quant_hard em τ=0,90 na maioria dos 20 (sinal, α=0,05).
H-D2: quant_soft <= gauss em τ=0,90 na maioria dos 20 (idem).
H-D3 (exploratória): perfil do edge por τ.

Saída: results/distillation/decision_utility_analysis.csv + stdout.
Uso: uv run python scripts/analyze_decision_utility.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

REPO = Path(__file__).resolve().parents[1]
TAUS = [round(q, 2) for q in np.arange(0.05, 0.96, 0.05)]


def main():
    d = pd.read_csv(REPO / "results/distillation/decision_utility.csv")
    cols = [f"cost_q{int(t * 100):02d}" for t in TAUS]
    m = d.groupby(["dataset", "model"])[cols].mean()

    rows = []
    for ds in sorted(d.dataset.unique()):
        r = {"dataset": ds}
        for tau, c in zip(TAUS, cols):
            soft = m.loc[(ds, "quant_soft"), c]
            hard = m.loc[(ds, "quant_hard"), c]
            gauss = m.loc[(ds, "gauss"), c]
            r[f"edge_hard_{c}"] = (hard - soft) / hard if hard else np.nan
            r[f"edge_gauss_{c}"] = (gauss - soft) / gauss if gauss else np.nan
        r["soft_le_hard_q90"] = (m.loc[(ds, "quant_soft"), "cost_q90"]
                                 <= m.loc[(ds, "quant_hard"), "cost_q90"])
        r["soft_le_gauss_q90"] = (m.loc[(ds, "quant_soft"), "cost_q90"]
                                  <= m.loc[(ds, "gauss"), "cost_q90"])
        rows.append(r)
    R = pd.DataFrame(rows)

    n = len(R)
    if n < 20:
        print(f"*** ANALISE PARCIAL: {n}/20 datasets no CSV — os testes "
              f"pre-registrados de H-D1/H-D2 exigem a grade completa; "
              f"NENHUM p-valor abaixo deve ser citado. ***\n")
    for hyp, col, other in (("H-D1", "soft_le_hard_q90", "quant_hard"),
                            ("H-D2", "soft_le_gauss_q90", "gauss")):
        w = int(R[col].sum())
        bt = binomtest(w, n, alternative="greater")
        diffs = (m.xs(other, level=1)["cost_q90"]
                 - m.xs("quant_soft", level=1)["cost_q90"])
        wx = wilcoxon(diffs, alternative="greater")
        print(f"{hyp}: soft <= {other} em τ=0,90 em {w}/{n} | "
              f"sinal p={bt.pvalue:.4f} | Wilcoxon p={wx.pvalue:.4f} "
              f"(Wilcoxon = suplementar, fora do pre-registro)")

    print("\nH-D3 — edge mediano do soft (redução % de custo) por τ:")
    prof = pd.DataFrame({
        "tau": TAUS,
        "vs_hard_%": [100 * R[f"edge_hard_{c}"].median() for c in cols],
        "vs_gauss_%": [100 * R[f"edge_gauss_{c}"].median() for c in cols],
    })
    print(prof.round(2).to_string(index=False))

    out = REPO / "results/distillation/decision_utility_analysis.csv"
    R.round(6).to_csv(out, index=False)
    print(f"\nsalvo: {out.name}")


if __name__ == "__main__":
    main()
