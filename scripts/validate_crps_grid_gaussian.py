#!/usr/bin/env python3
"""Valida o estimador de CRPS em grade de 19 quantis contra a forma
fechada gaussiana (artefato pedido pela auditoria de 19/10/2026: o
"~4-5% de sobre-estimativa, aprox. independente de locacao e escala"
citado nos textos nao rastreava a CSV versionado; o A-GRID cobre apenas
19-vs-99 quantis).

CRPS fechado de N(mu, sigma) para observacao y:
  sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],  z=(y-mu)/sigma.
Estimador do estudo: media da pinball nas 19 tau (0,05..0,95) vezes 2.

Saida: results/distillation/crps_grid_gaussian_validation.csv
Uso: uv run python scripts/validate_crps_grid_gaussian.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
TAUS = np.arange(0.05, 0.951, 0.05)
rng = np.random.default_rng(42)


def crps_closed(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1)
                    + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_grid(y, mu, sigma):
    q = mu[:, None] + sigma[:, None] * stats.norm.ppf(TAUS)[None, :]
    diff = y[:, None] - q
    pinball = np.maximum(TAUS[None, :] * diff, (TAUS[None, :] - 1) * diff)
    return 2 * pinball.mean(axis=1)


def main():
    rows = []
    for mu0, s0 in [(0.0, 1.0), (10.0, 1.0), (0.0, 0.1), (100.0, 25.0)]:
        n = 200_000
        mu = np.full(n, mu0)
        sigma = np.full(n, s0)
        y = rng.normal(mu0, s0, size=n)
        closed = crps_closed(y, mu, sigma).mean()
        grid = crps_grid(y, mu, sigma).mean()
        rows.append({"mu": mu0, "sigma": s0, "crps_closed": closed,
                     "crps_grid19": grid, "ratio": grid / closed})
        print(f"N({mu0},{s0}): fechado={closed:.5f} grade19={grid:.5f} "
              f"razao={grid/closed:.4f}")
    out = REPO / "results/distillation/crps_grid_gaussian_validation.csv"
    pd.DataFrame(rows).round(6).to_csv(out, index=False)
    print(f"salvo: {out}")


if __name__ == "__main__":
    main()
