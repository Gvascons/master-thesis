#!/usr/bin/env python3
"""Análise descritiva datada (14/10/2026, pós-hoc declarada): retenção
distribucional condicionada à vantagem RELATIVA do professor.

Motivação: a mediana incondicional (+0,19) agrega regimes que a regra de
decisão do cap. 7 manda separar. O edge ABSOLUTO não serve de
condicionante (dominado pela escala do alvo: Spearman -0,27 n.s.);
o edge RELATIVO rel = (CRPS_hard - CRPS_teacher)/CRPS_hard é scale-free.

Resultado: o condicionante prediz o SINAL, não a magnitude — na metade
de edge relativo acima da mediana a retenção é positiva em 10/10; as 4
falhas concentram-se todas na metade baixa (Fisher exato p=0,043).
Leitura honesta: o gate da regra elimina o risco de dano (downside),
não garante retenção grande.

Saída: results/distillation/retention_conditional.csv + stdout.
Uso: uv run python scripts/retention_conditional.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, spearmanr

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/distillation"


def main():
    r = pd.read_csv(D / "paper_retention.csv")
    e = pd.read_csv(D / "extension.csv")
    e = e[~e.dataset.str.contains("_cap|_insample|_logtarget")]
    ge = e.groupby(["dataset", "cell"])["crps"].mean().unstack("cell")
    d = pd.read_csv(D / "distill.csv")
    dq = (d[(d.teacher == "tabpfn") & (d.student == "xgb_quant")]
          .groupby(["dataset", "target"])["crps"].mean().unstack("target"))
    te = pd.read_csv(D / "teacher_eval.csv")
    tpfn = te[te.teacher == "tabpfn"].set_index("dataset")["crps"]

    rows = []
    for _, row in r.iterrows():
        ds = row.dataset
        if ds in ge.index:
            hard, teach = ge.loc[ds, "student_quant_hard"], ge.loc[ds, "teacher"]
        else:
            hard, teach = dq.loc[ds, "hard"], tpfn.loc[ds]
        rows.append(dict(dataset=ds, ret_crps=row.ret_crps,
                         rel_edge=(hard - teach) / hard))
    R = pd.DataFrame(rows).sort_values("rel_edge", ascending=False)
    med = R.rel_edge.median()
    R["regime"] = (R.rel_edge > med).map({True: "alto", False: "baixo"})
    print(R.round(4).to_string(index=False))

    hi, lo = R[R.regime == "alto"], R[R.regime == "baixo"]
    print(f"\nmediana do edge relativo: {med:.3f}")
    print(f"edge ALTO  (n={len(hi)}): ret mediana {hi.ret_crps.median():.3f}, "
          f"positiva {(hi.ret_crps > 0).sum()}/{len(hi)}")
    print(f"edge BAIXO (n={len(lo)}): ret mediana {lo.ret_crps.median():.3f}, "
          f"positiva {(lo.ret_crps > 0).sum()}/{len(lo)}")
    rho, p = spearmanr(R.rel_edge, R.ret_crps)
    print(f"Spearman(rel_edge, ret): rho={rho:.3f} p={p:.4f} "
          f"(magnitude NAO e predita)")
    tab = [[(hi.ret_crps < 0).sum(), (hi.ret_crps > 0).sum()],
           [(lo.ret_crps < 0).sum(), (lo.ret_crps > 0).sum()]]
    _, pf = fisher_exact(tab, alternative="less")
    print(f"Fisher exato (falhas na metade baixa): p={pf:.4f}")

    out = D / "retention_conditional.csv"
    R.round(6).to_csv(out, index=False)
    print(f"salvo: {out.name}")


if __name__ == "__main__":
    main()
