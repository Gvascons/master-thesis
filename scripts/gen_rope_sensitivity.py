"""Regenera results/aggregated/rope_sensitivity.csv com as DUAS convenções.

Correção de 23/11/2026 (auditoria externa): a análise bayesiana do cap. 5
reportava E[theta] (massa posterior esperada) sob o rótulo de
probabilidade posterior. Este gerador versiona a sensibilidade do ROPE
nas duas grandezas, com os cortes declarados:

  - n_equiv_Etheta_05 : nº de não-líderes com E[theta_ROPE] > 0,5
  - n_equiv_P_05      : idem, convenção Benavoli (P região mais provável) > 0,5
  - n_equiv_P_95      : idem, no corte estrito P >= 0,95

Núcleo de 11 modelos, AUC binária do hold-out, referência = TabPFN
(líder de rank), prior 0,6, 50.000 amostras, semente 0 — como na análise
original (notebooks/00_presentation.ipynb).
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.enrichments import (bayesian_signed_rank,
                                        bayesian_signed_rank_probs)

GEN2025 = ["xgboost", "lightgbm", "catboost", "tabpfn", "realmlp", "tabm",
           "tabnet", "mlp", "ft_transformer", "saint", "stab"]
OUT = REPO / "results" / "aggregated" / "rope_sensitivity.csv"


def main():
    test = pd.read_csv(REPO / "results/aggregated/test_results.csv")
    piv = (test[test.task_type == "binary"]
           .pivot(index="dataset", columns="model", values="roc_auc")
           [GEN2025].dropna(axis=1))
    rows = []
    for rope in (0.005, 0.01, 0.02):
        e_rope, p05, p95, e_sum = 0, 0, 0, 0.0
        for mdl in GEN2025:
            if mdl == "tabpfn":
                continue
            diff = (piv[mdl] - piv["tabpfn"]).to_numpy()
            _, e_mid, _ = bayesian_signed_rank(diff, rope=rope,
                                               nsamples=50000, seed=0)
            _, p_mid, _ = bayesian_signed_rank_probs(diff, rope=rope,
                                                     nsamples=50000, seed=0)
            e_sum += e_mid
            e_rope += e_mid > 0.5
            p05 += p_mid > 0.5
            p95 += p_mid >= 0.95
        rows.append(dict(rope=rope, n_equiv_of_10=e_rope,
                         mean_P_equiv=round(e_sum / 10, 3),
                         n_equiv_Etheta_05=e_rope, n_equiv_P_05=p05,
                         n_equiv_P_95=p95))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
