#!/usr/bin/env python3
"""Evidência quantitativa da escolha do split 80:20 (cap. 4).

Dois lados, dois artefatos:
(1) precisão da métrica no hold-out — IC95 bootstrap (B=4.000, seed 0) da
    AUC por instância nas predições do E7 (results/complementarity),
    no maior conjunto medido (adult) e no piso da grade (credit_g);
(2) inclinação das curvas de aprendizado na região do corte — melhora
    relativa da métrica primária de 50% para 100% do pool
    (results/learning_curve/curve.csv), por modelo x dataset.

Saída: results/aggregated/holdout_split_evidence.csv
Uso:   uv run python scripts/holdout_split_evidence.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
PRIM = {"binary": ("roc_auc", True), "multiclass": ("log_loss", False),
        "regression": ("rmse", False)}


def ci_rows() -> list[dict]:
    rows = []
    for ds in ("adult", "credit_g"):
        f = REPO / f"results/complementarity/preds_xgboost_{ds}.parquet"
        if not f.exists():
            print(f"aviso: {f} ausente — rode complementarity_analysis.py")
            continue
        df = pd.read_parquet(f)
        y, p = df.y_true.values, df.p1.values
        rng = np.random.default_rng(0)
        n = len(y)
        aucs = []
        for _ in range(4000):
            i = rng.integers(0, n, n)
            if len(np.unique(y[i])) < 2:
                continue
            aucs.append(roc_auc_score(y[i], p[i]))
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        rows.append({"tipo": "ic_auc_holdout", "dataset": ds, "modelo": "xgboost",
                     "n_test": n, "auc": roc_auc_score(y, p),
                     "ic_lo": lo, "ic_hi": hi, "meia_largura": (hi - lo) / 2})
    return rows


def slope_rows() -> list[dict]:
    c = pd.read_csv(REPO / "results/learning_curve/curve.csv")
    rows = []
    for (ds, m), g in c.groupby(["dataset", "model"]):
        col, hi = PRIM[g.task_type.iloc[0]]
        a = g.groupby("train_size")[col].mean().sort_index()
        sizes = a.index.values
        full = sizes.max()
        half = sizes[np.argmin(np.abs(sizes - full / 2))]
        v_half, v_full = a.loc[half], a.loc[full]
        rel = ((v_full - v_half) if hi else (v_half - v_full)) / abs(v_half)
        rows.append({"tipo": "melhora_rel_50_100", "dataset": ds, "modelo": m,
                     "n_half": int(half), "n_full": int(full),
                     "v_half": v_half, "v_full": v_full, "melhora_rel": rel})
    return rows


def main() -> None:
    rows = ci_rows() + slope_rows()
    out = REPO / "results/aggregated/holdout_split_evidence.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    sl = pd.DataFrame([r for r in rows if r["tipo"] == "melhora_rel_50_100"])
    print(f"OK: {out.relative_to(REPO)}")
    print(f"mediana da melhora relativa 50%->100%: {sl.melhora_rel.median():.4f}")
    print(f"máximo: {sl.melhora_rel.max():.4f} "
          f"({sl.loc[sl.melhora_rel.idxmax(), 'modelo']} / "
          f"{sl.loc[sl.melhora_rel.idxmax(), 'dataset']})")
    for r in rows:
        if r["tipo"] == "ic_auc_holdout":
            print(f"{r['dataset']}: n={r['n_test']}, AUC {r['auc']:.4f}, "
                  f"IC95 [{r['ic_lo']:.4f}; {r['ic_hi']:.4f}] "
                  f"(±{r['meia_largura']:.4f})")


if __name__ == "__main__":
    main()
