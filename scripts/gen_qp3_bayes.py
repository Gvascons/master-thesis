"""Posterior bayesiana da retenção da QP3, nas duas convenções.

Versiona o número citado no cap. 7 (0,96 de probabilidade posterior em
retenção positiva na convenção de Benavoli; E[theta] = 0,70), sobre o
vetor de retenções de paper_retention.csv, com varredura de ROPE.

Saída: results/distillation/qp3_bayes.csv
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.enrichments import (bayesian_signed_rank,
                                        bayesian_signed_rank_probs)

OUT = REPO / "results" / "distillation" / "qp3_bayes.csv"


def main():
    ret = pd.read_csv(REPO / "results/distillation/paper_retention.csv")
    d = ret["ret_crps"].to_numpy()
    rows = []
    for rope in (0.02, 0.05, 0.10):
        e = bayesian_signed_rank(d, rope=rope, nsamples=50000, seed=0)
        p = bayesian_signed_rank_probs(d, rope=rope, nsamples=50000, seed=0)
        rows.append(dict(rope=rope, E_neg=round(e[0], 3),
                         E_rope=round(e[1], 3), E_pos=round(e[2], 3),
                         P_neg=round(p[0], 3), P_rope=round(p[1], 3),
                         P_pos=round(p[2], 3)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
