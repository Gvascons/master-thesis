#!/usr/bin/env python3
"""Análise complementar descritiva datada (14/10/2026), sobre artefatos
existentes — nenhum experimento novo.

Matriz de taxa de vitória par-a-par em CRPS (20 datasets; mesma montagem
core+extensão de scripts/analyze_baselines.py) entre os 5 sistemas
distribucionais {teacher, student_soft, student_hard, qrf, ngboost},
com uma coluna extra de latência típica (us/linha) por sistema:
  - teacher, student_soft, student_hard: mediana de us_per_row em
    results/distillation/pareto.csv (sistemas teacher_ctx,
    student_quant_soft, student_quant_hard);
  - qrf, ngboost: mediana de us_per_row em classical_baselines.csv.
Célula [i, j] = fração dos 20 datasets em que o sistema i tem CRPS menor
que o sistema j (empate conta 0,5). Latência sem fonte direta vira N/D.

Saída: results/distillation/winrate_by_tier.csv + stdout.
Uso: uv run python scripts/analysis_extra_winrate_by_tier.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/distillation"

PARETO_NAME = dict(teacher="teacher_ctx", student_soft="student_quant_soft",
                   student_hard="student_quant_hard")


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


def latencies():
    """Latência típica (mediana de us/linha) por sistema, de artefatos."""
    lat = {}
    cb = pd.read_csv(D / "classical_baselines.csv")
    for m in ("qrf", "ngboost"):
        lat[m] = cb.loc[cb.model == m, "us_per_row"].median()
    pareto_path = D / "pareto.csv"
    if pareto_path.exists():
        p = pd.read_csv(pareto_path)
        if "us_per_row" in p.columns:
            med = p.groupby("system")["us_per_row"].median()
            for sysname, pname in PARETO_NAME.items():
                lat[sysname] = med.get(pname, np.nan)
    for s in ("teacher", "student_soft", "student_hard"):
        lat.setdefault(s, np.nan)
    return lat


def main():
    T = build_table()
    systems = list(T.columns)
    n = len(T)
    W = pd.DataFrame(index=systems, columns=systems, dtype=float)
    for a in systems:
        for b in systems:
            if a == b:
                W.loc[a, b] = np.nan
                continue
            wins = (T[a] < T[b]).sum() + 0.5 * (T[a] == T[b]).sum()
            W.loc[a, b] = wins / n

    lat = latencies()
    W["lat_us_por_linha"] = [
        round(lat[s], 2) if np.isfinite(lat[s]) else "N/D" for s in systems]

    print(f"taxa de vitória em CRPS (linha vence coluna; {n} datasets; "
          f"empate=0,5):\n")
    print(W.round(3).to_string())
    print("\nfontes de latência: teacher/students = mediana de us_per_row "
          "em pareto.csv (teacher_ctx, student_quant_soft, "
          "student_quant_hard); qrf/ngboost = mediana de us_per_row em "
          "classical_baselines.csv.")
    out = D / "winrate_by_tier.csv"
    W.round(6).to_csv(out, index_label="sistema")
    print(f"salvo: {out.name}")


if __name__ == "__main__":
    main()
