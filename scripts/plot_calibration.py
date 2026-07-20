#!/usr/bin/env python3
"""Calibration (reliability) figure for the paper — Fig. 2.

Empirical coverage P(y <= q_tau(x)) vs nominal tau on the benchmark
hold-out, for: teacher (TabPFN), distilled quantile student (soft), and the
hard-label control — on the two strong-retention datasets
(california_housing, year_prediction). Perfect calibration = diagonal.
Recomputes seed-0 students from the cached OOF targets (cheap) and the
teacher's hold-out quantiles (GPU minutes).

Palette: slots 1-3 of the validated reference order (teacher blue, distilled
magenta, control green? — order fixed by identity: teacher #2a78d6,
distilled #e87ba4, control #1baf7a per the pareto figure's identity map);
markers distinct; diagonal in muted gray.

Usage: PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/plot_calibration.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from distill import (QUANTILES, OOF_DIR, fit_teacher, fit_xgb_quant_hard,
                     fit_xgb_quant_soft, free_teacher, load_xgb_params,
                     make_teacher, ordinal_encode, prep_pool, sort_quantiles,
                     teacher_predict)
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_seed

DATASETS = ["california_housing", "year_prediction"]
STYLE = {  # same identity colors as the Pareto figure
    "teacher":   ("#2a78d6", "o", "TabPFN (teacher)"),
    "distilled": ("#e87ba4", "*", "Aluno-quantil destilado"),
    "control":   ("#1baf7a", "D", "Aluno-quantil (controle)"),
}


def coverage_curve(Q, y):
    return [float(np.mean(y <= Q[:, k])) for k in range(len(QUANTILES))]


def main():
    exp_cfg = load_experiment_config()
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.2 * len(DATASETS), 4.6))
    qcols = [f"q{int(t*100):02d}" for t in QUANTILES]

    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        X_pool, y_pool, X_test, y_test, info = prep_pool(ds, exp_cfg)
        Xp, Xt = ordinal_encode(X_pool, X_test)
        oof = pd.read_parquet(OOF_DIR / f"tabpfn_{ds}.parquet")
        params = load_xgb_params(ds)

        curves = {}
        model = make_teacher("auto")
        fit_teacher(model, Xp, y_pool)
        _, Qt = teacher_predict(model, Xt, batch=500)
        free_teacher(model)
        curves["teacher"] = coverage_curve(Qt, y_test)

        set_seed(0)
        m = fit_xgb_quant_soft(Xp, oof[qcols].to_numpy(), params, 0)
        curves["distilled"] = coverage_curve(sort_quantiles(np.asarray(m.predict(Xt))), y_test)
        set_seed(0)
        m = fit_xgb_quant_hard(Xp, y_pool, params, 0)
        curves["control"] = coverage_curve(sort_quantiles(np.asarray(m.predict(Xt))), y_test)

        ax.plot([0, 1], [0, 1], "--", color="#9a9a92", lw=1.2, zorder=1)
        for key, (color, marker, label) in STYLE.items():
            ms = 11 if marker == "*" else 6
            ax.plot(QUANTILES, curves[key], "-", color=color, lw=2,
                    marker=marker, ms=ms, label=label, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("quantil nominal τ", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("cobertura empírica P(y ≤ q_τ)", fontsize=10)
    np.atleast_1d(axes)[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle("Calibração no hold-out: o aluno destilado herda a calibração do teacher",
                 fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / f"results/figures/calibration_distill.{ext}",
                    dpi=150, bbox_inches="tight")
    print("figura salva: results/figures/calibration_distill.png|pdf")


if __name__ == "__main__":
    main()
