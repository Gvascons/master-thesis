#!/usr/bin/env python3
"""Análise complementar descritiva/inferencial datada (14/10/2026), sobre
artefatos existentes — nenhum experimento novo.

Comparação de 5 sistemas distribucionais {teacher, student_soft,
student_hard, qrf, ngboost} em CRPS sobre os 20 datasets (média sobre
sementes por dataset; montagem core+extensão no padrão de
scripts/analyze_baselines.py). Teste de Friedman (qui-quadrado) + ranks
médios + distância crítica de Nemenyi (alpha=0,05; q_alpha da tabela do
studentized range via scipy.stats.studentized_range, e p-valores
par-a-par via scikit_posthocs quando disponível).

Saída: results/distillation/friedman_systems.csv + stdout.
Uso: uv run python scripts/analysis_extra_friedman_systems.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/distillation"
ALPHA = 0.05


def build_table():
    """20 datasets x 5 sistemas, CRPS médio sobre sementes
    (padrão de analyze_baselines.py)."""
    cb = pd.read_csv(D / "classical_baselines.csv")
    e = pd.read_csv(D / "extension.csv")
    d = pd.read_csv(D / "distill.csv")
    te = pd.read_csv(D / "teacher_eval.csv")

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

    cbm = cb.groupby(["dataset", "model"])["crps"].mean()
    rows = []
    for ds in sorted(cb.dataset.unique()):
        soft, hard, teach = ours(ds)
        rows.append(dict(dataset=ds, teacher=teach, student_soft=soft,
                         student_hard=hard, qrf=cbm.loc[(ds, "qrf")],
                         ngboost=cbm.loc[(ds, "ngboost")]))
    return pd.DataFrame(rows).set_index("dataset")


def main():
    T = build_table()
    systems = list(T.columns)
    k, n = len(systems), len(T)
    print(f"tabela CRPS: {n} datasets x {k} sistemas\n")

    stat, p = friedmanchisquare(*[T[c] for c in systems])
    ranks = T.rank(axis=1)  # menor CRPS = rank 1
    avg = ranks.mean().sort_values()
    print("ranks médios (menor = melhor):")
    for s, r in avg.items():
        print(f"  {s:<13} {r:.3f}")
    print(f"\nFriedman: chi2={stat:.3f}, df={k - 1}, p={p:.3e}")

    q_alpha = studentized_range.ppf(1 - ALPHA, k, np.inf)
    cd = q_alpha / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n))
    print(f"Nemenyi: q_alpha(k={k}, inf)={q_alpha:.4f} -> CD={cd:.4f} "
          f"(alpha={ALPHA})")

    print("\npares com |diferença de rank| > CD:")
    for i, a in enumerate(avg.index):
        for b in avg.index[i + 1:]:
            diff = abs(avg[a] - avg[b])
            if diff > cd:
                print(f"  {a} vs {b}: diff={diff:.3f} > CD")

    try:
        import scikit_posthocs as sp
        pn = sp.posthoc_nemenyi_friedman(T.to_numpy())
        pn.index = pn.columns = systems
        print("\np-valores Nemenyi (scikit_posthocs), pares com p<0.05:")
        for i, a in enumerate(systems):
            for b in systems[i + 1:]:
                if pn.loc[a, b] < ALPHA:
                    print(f"  {a} vs {b}: p={pn.loc[a, b]:.4f}")
    except ImportError:
        print("\n(scikit_posthocs indisponível; apenas CD analítico)")

    out_df = T.copy()
    for s in systems:
        out_df[f"rank_{s}"] = ranks[s]
    out_df.loc["__rank_medio__"] = (
        [np.nan] * k + [ranks[s].mean() for s in systems])
    out = D / "friedman_systems.csv"
    out_df.round(6).to_csv(out)
    print(f"\nsalvo: {out.name}")


if __name__ == "__main__":
    main()
