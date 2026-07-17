#!/usr/bin/env python3
"""LODO validation of the decision framework (AI-2, Pillar B).

For each dataset (leave-one-dataset-out): fit the depth-2 meta-feature tree
(the deck's descriptive tree) on the remaining datasets and recommend a
family for the held-out one. Metrics: hit-rate and normalized REGRET
(0 = picked the best family's best model; 1 = the worst) against fixed
policies (always-GBDT / always-DL / always-FM).

First run (11 models, 15/07/2026) established: the tree does NOT generalize
(hit 0.11 vs 0.44 majority baseline) and always-FM has the lowest regret
(median 0.018) — the empirical basis for the constraint-driven,
FM-first framing of the framework. Re-run after the 14-model refresh.

Usage:
    uv run python scripts/lodo_validation.py
    uv run python scripts/lodo_validation.py --out results/aggregated/lodo_14.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.registry import load_dataset
from src.evaluation.enrichments import build_meta_features, winning_family_table
from src.models.factory import get_model_family

PRIMARY = {"binary": ("roc_auc", True), "multiclass": ("log_loss", False),
           "regression": ("rmse", False)}
SHORT = {"gbdt": "GBDT", "deep_learning": "DL", "foundation_model": "FM"}
FEATS = ["log_n", "log_feat", "imbalance", "cat_frac",
         "is_binary", "is_multiclass", "is_regression"]


def regret(t, fam_map, ds, family_short):
    """Normalized regret of recommending `family_short` on dataset `ds`."""
    sub = t[t.dataset == ds]
    task = sub.task_type.iloc[0]
    col, higher = PRIMARY[task]
    s = sub.dropna(subset=[col]).copy()
    s["fam"] = s.model.map(lambda m: SHORT[fam_map[m]])
    best = s[col].max() if higher else s[col].min()
    worst = s[col].min() if higher else s[col].max()
    fam_scores = s[s.fam == family_short][col]
    if fam_scores.empty:
        return np.nan
    got = fam_scores.max() if higher else fam_scores.min()
    rng = abs(worst - best)
    return abs(got - best) / rng if rng > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/aggregated/lodo_validation.csv")
    args = ap.parse_args()

    t = pd.read_csv(REPO / "results" / "aggregated" / "test_results.csv")
    fam_map = {m: get_model_family(m) for m in t.model.unique()}
    n_models = t.model.nunique()

    meta = build_meta_features(sorted(t.dataset.unique()), load_dataset,
                               datasets_yaml=str(REPO / "configs" / "datasets.yaml"))
    W = winning_family_table(t, fam_map, meta).copy()
    W["imbalance"] = W["imbalance"].fillna(1.0)
    W["log_n"] = np.log10(W["n"])
    W["log_feat"] = np.log10(W["n_feat"])
    for tk in ("binary", "multiclass", "regression"):
        W[f"is_{tk}"] = (W["task"] == tk).astype(int)

    rows = []
    for left_out in W.index:
        tr = W.drop(index=left_out)
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf.fit(tr[FEATS].values, tr["win_family"].values)
        pred = clf.predict(W.loc[[left_out], FEATS].values)[0]
        rows.append({
            "dataset": left_out, "task": W.loc[left_out, "task"],
            "pred": pred, "actual": W.loc[left_out, "win_family"],
            "hit": pred == W.loc[left_out, "win_family"],
            "regret_lodo": regret(t, fam_map, left_out, pred),
            "regret_GBDT": regret(t, fam_map, left_out, "GBDT"),
            "regret_DL": regret(t, fam_map, left_out, "DL"),
            "regret_FM": regret(t, fam_map, left_out, "FM"),
        })
    res = pd.DataFrame(rows).set_index("dataset")

    print(f"[{n_models} modelos no test_results.csv]")
    print(res.round(3).to_string())
    print(f"\nhit-rate LODO: {res.hit.mean():.2f}  "
          f"(baseline familia majoritaria: {W.win_family.value_counts(normalize=True).max():.2f})")
    cols = ["regret_lodo", "regret_GBDT", "regret_DL", "regret_FM"]
    print("\nregret medio:");   print(res[cols].mean().round(3).to_string())
    print("\nregret mediano:"); print(res[cols].median().round(3).to_string())

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out)
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
