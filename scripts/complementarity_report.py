#!/usr/bin/env python3
"""E7 — agregação e teste das hipóteses pré-registradas (H-C1..H-C4).

Lê results/complementarity/preds_*.parquet (gerados por
complementarity_analysis.py), produz:
  - complementarity_pairs.csv   (uma linha por dataset x par XGB x FM)
  - complementarity_margens.csv (vitórias líquidas do FM por quintil de
    margem do XGBoost, binárias)
  - complementarity_hipoteses.csv (vereditos H-C1..H-C4)

Uso: uv run python scripts/complementarity_report.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/complementarity"
FMS = ["tabpfn", "tabfm"]


def mcnemar_exact(b: int, c: int) -> float:
    """Teste exato de McNemar (binomial bilateral sobre os discordantes)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def load(model: str, ds: str) -> pd.DataFrame | None:
    p = OUT / f"preds_{model}_{ds}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def main() -> None:
    sanity = pd.read_csv(OUT / "sanity_check.csv")
    bad = sanity[sanity.gate != "OK"]
    if len(bad):
        print("ATENCAO - celulas fora do gate (excluidas):")
        print(bad[["model", "dataset", "gate"]].to_string(index=False))
    ok = {(r.model, r.dataset) for r in sanity.itertuples() if r.gate == "OK"}
    meta = {r.dataset: (r.task_type, r.n_pool, r.n_test)
            for r in sanity.itertuples() if r.gate == "OK"}

    pair_rows, margin_rows = [], []
    for ds, (task, n_pool, n_test) in sorted(meta.items()):
        if ("xgboost", ds) not in ok:
            continue
        dx = load("xgboost", ds)
        for fm in FMS:
            if (fm, ds) not in ok:
                continue
            df = load(fm, ds)
            if df is None or dx is None:
                continue
            assert (df.row.values == dx.row.values).all(), f"linhas desalinhadas em {ds}"
            y = dx.y_true.values
            row = {"dataset": ds, "fm": fm, "task_type": task,
                   "n_pool": n_pool, "n_test": n_test}

            if task in ("binary", "multiclass"):
                cx = (dx.y_pred.values == y)
                cf = (df.y_pred.values == y)
                agree = float((dx.y_pred.values == df.y_pred.values).mean())
                both = int((cx & cf).sum()); only_x = int((cx & ~cf).sum())
                only_f = int((~cx & cf).sum()); neither = int((~cx & ~cf).sum())
                row.update(agreement=agree, both_correct=both, only_xgb=only_x,
                           only_fm=only_f, neither=neither,
                           mcnemar_p=mcnemar_exact(only_x, only_f),
                           net_fm=only_f - only_x,
                           net_fm_rate=(only_f - only_x) / len(y))
                if task == "binary":
                    # vantagem = dAUC (FM - XGB)
                    from sklearn.metrics import roc_auc_score
                    row["advantage"] = float(roc_auc_score(y, df.p1) -
                                             roc_auc_score(y, dx.p1))
                    row["adv_metric"] = "dAUC"
                    # H-C2: quintis de margem do XGBoost
                    margin = np.abs(dx.p1.values - 0.5)
                    q = pd.qcut(margin, 5, labels=False, duplicates="drop")
                    for qi in sorted(np.unique(q)):
                        m = q == qi
                        margin_rows.append({
                            "dataset": ds, "fm": fm, "quintil_margem": int(qi),
                            "n": int(m.sum()),
                            "net_fm_rate": float(((~cx & cf)[m].sum()
                                                  - (cx & ~cf)[m].sum()) / m.sum()),
                            "agreement": float((dx.y_pred.values == df.y_pred.values)[m].mean()),
                        })
                    lo = [r for r in margin_rows if r["dataset"] == ds
                          and r["fm"] == fm and r["quintil_margem"] == 0][0]
                    hi = [r for r in margin_rows if r["dataset"] == ds
                          and r["fm"] == fm][-1]
                    row["hc2_lo_minus_hi"] = lo["net_fm_rate"] - hi["net_fm_rate"]
                    # H-C3: classe minoritaria vs majoritaria
                    minority = int(pd.Series(y).value_counts().idxmin())
                    for lab, nm in ((minority, "min"), (1 - minority, "maj")):
                        m = y == lab
                        row[f"net_fm_rate_{nm}"] = float(
                            ((~cx & cf)[m].sum() - (cx & ~cf)[m].sum()) / m.sum())
                    row["hc3_min_minus_maj"] = (row["net_fm_rate_min"]
                                                - row["net_fm_rate_maj"])
                else:
                    from sklearn.metrics import log_loss
                    pcols = [c for c in dx.columns if c.startswith("p")]
                    row["advantage"] = float(log_loss(y, dx[pcols].values)
                                             - log_loss(y, df[pcols].values))
                    row["adv_metric"] = "dlogloss"
            else:  # regressao
                ex, ef = dx.abs_err.values, df.abs_err.values
                row.update(fm_better_frac=float((ef < ex).mean()),
                           err_spearman=float(stats.spearmanr(ex, ef).statistic),
                           advantage=float(np.median(ex) - np.median(ef)),
                           adv_metric="d_mederr")
                # centro vs cauda do alvo
                dist = np.abs(y - np.median(y))
                q = pd.qcut(dist, 5, labels=False, duplicates="drop")
                row["net_fm_rate_centro"] = float((ef < ex)[q == 0].mean()
                                                  - (ef > ex)[q == 0].mean())
                row["net_fm_rate_cauda"] = float((ef < ex)[q == q.max()].mean()
                                                 - (ef > ex)[q == q.max()].mean())
            pair_rows.append(row)

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "complementarity_pairs.csv", index=False)
    pd.DataFrame(margin_rows).to_csv(OUT / "complementarity_margens.csv", index=False)

    # ---- vereditos pré-registrados ----
    verd = []
    cla = pairs[pairs.task_type.isin(["binary", "multiclass"])]
    med_agree = cla.groupby("fm").agreement.median()
    for fm in FMS:
        if fm in med_agree:
            verd.append({"hipotese": "H-C1", "fm": fm,
                         "estatistica": f"mediana concordancia {med_agree[fm]:.3f}",
                         "veredito": "CONFIRMADA" if med_agree[fm] > 0.85 else "REFUTADA"})
    for fm in FMS:
        b = pairs[(pairs.fm == fm) & (pairs.task_type == "binary")]
        if not len(b):
            continue
        pos = int((b.hc2_lo_minus_hi > 0).sum()); n = len(b)
        p = float(stats.binomtest(pos, n, 0.5, alternative="greater").pvalue)
        verd.append({"hipotese": "H-C2", "fm": fm,
                     "estatistica": f"{pos}/{n} conjuntos com net-win maior na margem baixa; sinal p={p:.4f}",
                     "veredito": "CONFIRMADA" if p < 0.05 else "REFUTADA"})
        pos = int((b.hc3_min_minus_maj > 0).sum())
        p = float(stats.binomtest(pos, n, 0.5, alternative="greater").pvalue)
        verd.append({"hipotese": "H-C3", "fm": fm,
                     "estatistica": f"{pos}/{n} conjuntos com net-win maior na minoritaria; sinal p={p:.4f}",
                     "veredito": "CONFIRMADA" if p < 0.05 else "REFUTADA"})
        sub = pairs[pairs.fm == fm]
        rho = stats.spearmanr(sub.n_pool, sub.advantage)
        verd.append({"hipotese": "H-C4", "fm": fm,
                     "estatistica": f"Spearman(n_pool, vantagem)={rho.statistic:.3f} (p={rho.pvalue:.3f}, N={len(sub)})",
                     "veredito": "CONFIRMADA" if rho.statistic < 0 and rho.pvalue < 0.05
                     else ("DIRECIONALMENTE NEGATIVA (n.s.)" if rho.statistic < 0 else "REFUTADA")})
    vdf = pd.DataFrame(verd)
    vdf.to_csv(OUT / "complementarity_hipoteses.csv", index=False)
    print(pairs.to_string(index=False))
    print()
    print(vdf.to_string(index=False))


if __name__ == "__main__":
    main()
