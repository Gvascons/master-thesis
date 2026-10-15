#!/usr/bin/env python3
"""Análise complementar descritiva/inferencial datada (14/10/2026), sobre
artefatos existentes — nenhum experimento novo.

IC de 95% por bootstrap (10.000 reamostras, método do percentil, seed 0)
para as duas manchetes da retenção distribucional sobre os 20 datasets de
results/distillation/paper_retention.csv:
  (a) MEDIANA de ret_crps (manchete: 0,19);
  (b) fração de datasets com retenção positiva (manchete: 16/20).

Saída: results/distillation/bootstrap_retention.csv + stdout.
Uso: uv run python scripts/analysis_extra_bootstrap_retention.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/distillation"
N_BOOT, SEED = 10_000, 0


def main():
    ret = pd.read_csv(D / "paper_retention.csv")["ret_crps"].to_numpy()
    n = len(ret)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    samples = ret[idx]
    med_boot = np.median(samples, axis=1)
    pos_boot = (samples > 0).mean(axis=1)

    rows = []
    for name, point, boot in [
            ("mediana_ret_crps", float(np.median(ret)), med_boot),
            ("fracao_positiva", float((ret > 0).mean()), pos_boot)]:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(estatistica=name, pontual=point,
                         ic95_lo=lo, ic95_hi=hi,
                         n_datasets=n, n_boot=N_BOOT, seed=SEED))
        print(f"{name}: {point:.4f}  IC95% percentil "
              f"[{lo:.4f}, {hi:.4f}]  (n={n}, B={N_BOOT}, seed={SEED})")

    pos = int((ret > 0).sum())
    print(f"(fração positiva = {pos}/{n})")
    out = D / "bootstrap_retention.csv"
    pd.DataFrame(rows).round(6).to_csv(out, index=False)
    print(f"salvo: {out.name}")


if __name__ == "__main__":
    main()
