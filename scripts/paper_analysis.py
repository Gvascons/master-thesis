#!/usr/bin/env python3
"""Consolidated inference for the paper's headline claim (AI-2).

Gathers CRPS-gap retention over ALL unique datasets (5 core + CTR23
eligible pool) and reports the frequentist pair (one-sided sign test +
Wilcoxon) alongside the thesis's Bayesian signed-rank with ROPE — the same
dual-instrument identity the companion benchmark uses.

ROPE for retention: ±0.05 (retaining less than 5% of the teacher's gap is
treated as practically nothing; sensitivity at 0.02/0.05/0.10 reported).

Output: results/distillation/paper_stats.csv + stdout table.
Usage: uv run python scripts/paper_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.enrichments import bayesian_signed_rank


def retention_table():
    rows = []
    e = pd.read_csv(REPO / "results/distillation/extension.csv")
    ge = e.groupby(["dataset", "cell"]).agg(crps=("crps", "mean"))
    for ds in e.dataset.unique():
        t = ge.loc[(ds, "teacher"), "crps"]
        qh = ge.loc[(ds, "student_quant_hard"), "crps"]
        qs = ge.loc[(ds, "student_quant_soft"), "crps"]
        gap = qh - t
        rows.append({"dataset": ds, "source": "ctr23",
                     "ret_crps": (qh - qs) / gap if abs(gap) > 1e-9 else np.nan,
                     "teacher_edge": gap})
    d = pd.read_csv(REPO / "results/distillation/distill.csv")
    d = d[d.teacher == "tabpfn"]
    t5 = pd.read_csv(REPO / "results/distillation/teacher_eval.csv")
    t5 = t5[t5.teacher == "tabpfn"].set_index("dataset")
    gd = d.groupby(["dataset", "student", "target"]).agg(crps=("crps", "mean"))
    for ds in t5.index:
        qh = gd.loc[(ds, "xgb_quant", "hard"), "crps"]
        qs = gd.loc[(ds, "xgb_quant", "soft"), "crps"]
        gap = qh - t5.loc[ds, "crps"]
        rows.append({"dataset": ds, "source": "core",
                     "ret_crps": (qh - qs) / gap if abs(gap) > 1e-9 else np.nan,
                     "teacher_edge": gap})
    return pd.DataFrame(rows).sort_values("ret_crps", ascending=False)


def main():
    R = retention_table()
    print(R.round(3).to_string(index=False))
    r = R.ret_crps.dropna().values
    n, pos = len(r), int((r > 0).sum())
    print(f"\n=== N={n} datasets unicos ===")
    print(f"retencao > 0: {pos}/{n} | mediana {np.median(r):.3f} | max {r.max():.2f}")
    bt = binomtest(pos, n, alternative="greater")
    w = wilcoxon(r, alternative="greater")
    print(f"sinal (binomial, unilateral): p={bt.pvalue:.4f}")
    print(f"Wilcoxon (unilateral):        p={w.pvalue:.4f}")
    print("\nbayesiano signed-rank (sensibilidade do ROPE):")
    stats_rows = []
    for rope in (0.02, 0.05, 0.10):
        pl, pr, pg = bayesian_signed_rank(r, rope=rope)
        print(f"  ROPE ±{rope}: P(ret<0)={pl:.2f}  P(equiv)={pr:.2f}  P(ret>0)={pg:.2f}")
        stats_rows.append({"rope": rope, "p_neg": pl, "p_equiv": pr, "p_pos": pg})
    out = REPO / "results/distillation/paper_stats.csv"
    pd.concat([R, pd.DataFrame(stats_rows)], axis=1).to_csv(out, index=False)
    print(f"\nsalvo: {out}")


if __name__ == "__main__":
    main()
