#!/usr/bin/env python3
"""Análise pré-registrada do E1 (docs/desenho-melhorias-finais.md):
aluno destilado vs baselines distribucionais clássicos (NGBoost, QRF).

H-B1: sinal unilateral sobre os 20 datasets (destilado <= melhor clássico).
H-B2: latências dos clássicos no tier dos students.

Saída: results/distillation/baselines_analysis.csv + stdout.
Uso: uv run python scripts/analyze_baselines.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

REPO = Path(__file__).resolve().parents[1]


def main():
    cb = pd.read_csv(REPO / "results/distillation/classical_baselines.csv")
    e = pd.read_csv(REPO / "results/distillation/extension.csv")
    d = pd.read_csv(REPO / "results/distillation/distill.csv")
    te = pd.read_csv(REPO / "results/distillation/teacher_eval.csv")

    ge = (e[~e.dataset.str.contains("_cap|_insample|_logtarget")]
          .groupby(["dataset", "cell"])["crps"].mean())
    dq = (d[(d.teacher == "tabpfn") & (d.student == "xgb_quant")]
          .groupby(["dataset", "target"])["crps"].mean())
    tpfn = te[te.teacher == "tabpfn"].set_index("dataset")["crps"]

    def ours(ds):
        if (ds, "student_quant_soft") in ge.index:
            return (ge.loc[(ds, "student_quant_soft")],
                    ge.loc[(ds, "student_quant_hard")],
                    ge.loc[(ds, "teacher")])
        return dq.loc[(ds, "soft")], dq.loc[(ds, "hard")], tpfn.loc[ds]

    cbm = cb.groupby(["dataset", "model"]).agg(
        crps=("crps", "mean"), lat=("us_per_row", "mean"),
        picp80=("picp80", "mean"))
    rows = []
    for ds in sorted(cb.dataset.unique()):
        soft, hard, teach = ours(ds)
        ng, qrf = cbm.loc[(ds, "ngboost")], cbm.loc[(ds, "qrf")]
        best_cl = min(ng.crps, qrf.crps)
        rows.append(dict(
            dataset=ds, distilled=soft, hard_ctrl=hard, teacher=teach,
            ngboost=ng.crps, qrf=qrf.crps, best_classical=best_cl,
            best_name="ngboost" if ng.crps <= qrf.crps else "qrf",
            distilled_wins=soft <= best_cl,
            classical_beats_hard=best_cl < hard,
            classical_beats_teacher=best_cl < teach))
    R = pd.DataFrame(rows)
    print(R.round(4).to_string(index=False))
    w, n = int(R.distilled_wins.sum()), len(R)
    bt = binomtest(w, n, alternative="greater")
    wx = wilcoxon(R.best_classical - R.distilled, alternative="greater")
    print(f"\nH-B1: destilado <= melhor classico em {w}/{n} | "
          f"sinal p={bt.pvalue:.4f} | Wilcoxon p={wx.pvalue:.4f}")
    print(f"classico bate o controle hard em "
          f"{int(R.classical_beats_hard.sum())}/{n}; bate o teacher em "
          f"{int(R.classical_beats_teacher.sum())}/{n}")
    ngl = cbm.xs("ngboost", level=1)
    qfl = cbm.xs("qrf", level=1)
    print(f"H-B2: latencias us/linha — ngboost {ngl.lat.min():.1f}-"
          f"{ngl.lat.max():.1f}; qrf {qfl.lat.min():.1f}-{qfl.lat.max():.1f}")
    R.round(6).to_csv(REPO / "results/distillation/baselines_analysis.csv",
                      index=False)
    print("salvo: baselines_analysis.csv")


if __name__ == "__main__":
    main()
