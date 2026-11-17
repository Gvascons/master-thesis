#!/usr/bin/env python3
"""Leitura pré-registrada do R-TUNE estendido (adendo de 12/11/2026 em
docs/desenho-melhorias-finais.md): retenção com controle re-tunado,
lado a lado com a base herdada, nos 20 conjuntos de paper_retention.csv.

Retenção = (CRPS_hard - CRPS_soft) / (CRPS_hard - CRPS_professor),
mesma definição de paper_analysis.py; professor fixo (não re-tunado),
CRPS do professor vindo de extension.csv (cell=teacher) para os CTR23 e
de teacher_eval.csv (tabpfn) para os 5 do núcleo. Conjuntos com gap do
professor <= 0 são excluídos (não invertidos), como no veredito
principal. Critério de leitura (adendo): contagem de retenções
positivas e mediana com controle re-tunado, lado a lado com a base
herdada. H-T1: retenção re-tunada >= herdada, conjunto a conjunto.

Saída: results/distillation/retune_analysis.csv + tabela no stdout.
Uso:   uv run python scripts/retune_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

REPO = Path(__file__).resolve().parents[1]

def teacher_crps():
    t = {}
    e = pd.read_csv(REPO / "results/distillation/extension.csv")
    for ds, g in e[e.cell == "teacher"].groupby("dataset"):
        t[ds] = g.crps.mean()
    te = pd.read_csv(REPO / "results/distillation/teacher_eval.csv")
    for _, r in te[te.teacher == "tabpfn"].iterrows():
        t[r.dataset] = r.crps
    return t


def main():
    df = pd.read_csv(REPO / "results/distillation/retune_students.csv")
    agg = df.groupby(["dataset", "target", "hp_source"])["crps"].mean()
    tch = teacher_crps()
    rows = []
    for ds in sorted(df.dataset.unique()):
        row = {"dataset": ds, "teacher_crps": tch.get(ds, np.nan)}
        for src in ("inherited", "retuned"):
            try:
                qh = agg.loc[(ds, "hard", src)]
                qs = agg.loc[(ds, "soft", src)]
            except KeyError:
                row[f"ret_{src}"] = np.nan
                continue
            gap = qh - row["teacher_crps"]
            row[f"crps_hard_{src}"] = qh
            row[f"crps_soft_{src}"] = qs
            row[f"gap_{src}"] = gap
            row[f"ret_{src}"] = (qh - qs) / gap if gap > 1e-9 else np.nan
        rows.append(row)
    R = pd.DataFrame(rows)
    R["ht1_ok"] = R.ret_retuned >= R.ret_inherited
    R.to_csv(REPO / "results/distillation/retune_analysis.csv", index=False)

    print(R[["dataset", "ret_inherited", "ret_retuned", "ht1_ok"]]
          .round(3).to_string(index=False))
    for src in ("inherited", "retuned"):
        r = R[f"ret_{src}"].dropna().values
        n_pos = int((r > 0).sum())
        sign_p = binomtest(n_pos, len(r), 0.5, alternative="greater").pvalue
        try:
            w_p = wilcoxon(r, alternative="greater").pvalue
        except ValueError:
            w_p = np.nan
        print(f"\n[{src}] n={len(r)}  positivas={n_pos}/{len(r)}  "
              f"mediana={np.median(r):.3f}  sinal p={sign_p:.4f}  "
              f"Wilcoxon p={w_p:.4f}")
    ok = R.ht1_ok.dropna()
    print(f"\nH-T1 (ret_retuned >= ret_inherited): {int(R.ht1_ok.sum())}"
          f"/{len(ok)} conjuntos")
    excl = R[~(R.get("gap_retuned", pd.Series(dtype=float)) > 1e-9)]
    if len(excl):
        print("Excluídos/incompletos:",
              ", ".join(excl.dataset.tolist()))


if __name__ == "__main__":
    main()
