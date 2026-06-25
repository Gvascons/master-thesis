"""Sophisticated cross-dataset analyses that go beyond mean-metric tables.

All functions operate on the *finished* benchmark (no retraining): the
aggregated test/fold CSVs plus dataset meta-features. They back the
"sophistication layer" described in docs/00-analysis-narrative.md §4.

Contents
--------
- bayesian_signed_rank   : P(A worse)/P(equivalent)/P(A better) with a ROPE.
- normalized_regret      : per-model distance-to-best, scale-invariant.
- family_variance_decomposition : between- vs within-family rank variance (eta^2).
- build_meta_features    : per-dataset (n, n_feat, imbalance, cat_frac, task).
- winning_family_table   : per-dataset winner + family joined to meta-features.
- fold_robustness        : rank stability / worst-case from per-fold results.

Metric convention: PRIMARY maps task_type -> (column, higher_is_better).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY = {
    "binary": ("roc_auc", True),
    "regression": ("rmse", False),
    "multiclass": ("log_loss", False),
}
FAMILY_SHORT = {"gbdt": "GBDT", "deep_learning": "DL", "foundation_model": "FM"}


# --------------------------------------------------------------------------- #
# Bayesian model comparison (Benavoli et al., 2017 signed-rank test with ROPE)
# --------------------------------------------------------------------------- #
def bayesian_signed_rank(diff, rope, prior=0.6, nsamples=30000, seed=0):
    """Bayesian counterpart to the Wilcoxon signed-rank test.

    Parameters
    ----------
    diff : array of per-dataset differences (score_A - score_B) for a
        *higher-is-better* metric. Flip the sign for lower-is-better metrics
        before calling.
    rope : half-width of the Region Of Practical Equivalence. Two models whose
        typical difference falls within +-rope are "practically the same".
    prior : strength of the Dirichlet prior pseudo-observation placed at 0
        (inside the rope); 0.6 is the baycomp default.

    Returns
    -------
    (p_A_worse, p_rope, p_A_better) : posterior probabilities, summing to 1.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    d = np.concatenate(([0.0], d))                 # prior pseudo-obs at 0
    m = len(d)
    avg = (d[:, None] + d[None, :]) / 2.0          # Wilcoxon pairwise averages
    left = (avg < -rope).astype(float)
    right = (avg > rope).astype(float)
    ropem = 1.0 - left - right
    alpha = np.concatenate(([prior], np.ones(m - 1)))
    W = rng.dirichlet(alpha, size=nsamples)
    p_left = np.einsum("si,ij,sj->s", W, left, W).mean()
    p_right = np.einsum("si,ij,sj->s", W, right, W).mean()
    p_rope = np.einsum("si,ij,sj->s", W, ropem, W).mean()
    return float(p_left), float(p_rope), float(p_right)


def bayesian_pairwise_table(pivot, higher_is_better, rope, ref=None, **kw):
    """P(equivalent) / P(better) for every model vs a reference (or all pairs).

    `pivot` is datasets x models of the primary metric. If `ref` is given,
    compares every model against `ref`; otherwise returns the full matrix of
    P(row better than col)."""
    models = list(pivot.columns)
    sign = 1.0 if higher_is_better else -1.0
    if ref is not None:
        out = []
        for mdl in models:
            if mdl == ref:
                continue
            diff = sign * (pivot[mdl] - pivot[ref]).dropna().values
            pl, pr_, pg = bayesian_signed_rank(diff, rope, **kw)
            out.append({"model": mdl, "vs": ref, "P(worse)": pl,
                        "P(equiv)": pr_, "P(better)": pg})
        return pd.DataFrame(out)
    mat = pd.DataFrame(np.nan, index=models, columns=models)
    for a in models:
        for b in models:
            if a == b:
                continue
            diff = sign * (pivot[a] - pivot[b]).dropna().values
            _, _, pg = bayesian_signed_rank(diff, rope, **kw)
            mat.loc[a, b] = pg
    return mat


# --------------------------------------------------------------------------- #
# Scale-invariant aggregation
# --------------------------------------------------------------------------- #
def normalized_regret(pivot, higher_is_better):
    """Per-model mean normalized regret (0 = always the per-dataset best).

    On each dataset, regret = |metric - best| / (max - min across models);
    averaged over datasets. Scale-invariant, so it can be pooled fairly."""
    best = pivot.max(axis=1) if higher_is_better else pivot.min(axis=1)
    rng_ds = (pivot.max(axis=1) - pivot.min(axis=1)).replace(0, np.nan)
    if higher_is_better:
        regret = best.values[:, None] - pivot.values
    else:
        regret = pivot.values - best.values[:, None]
    regret = np.abs(regret) / rng_ds.values[:, None]
    return pd.Series(np.nanmean(regret, axis=0), index=pivot.columns).sort_values()


def family_variance_decomposition(rank_pivot, family_map):
    """Split rank variance into between- vs within-family (one-way, eta^2).

    `rank_pivot` is datasets x models of within-dataset ranks. Returns a dict
    with ss_total/ss_between/ss_within, eta^2 (between-family share), and the
    family mean ranks. eta^2 < 0.5 => within-family spread dominates, i.e.
    'family' is a weak abstraction for predicting performance."""
    long = rank_pivot.reset_index().melt(
        id_vars=rank_pivot.index.name or "index",
        var_name="model", value_name="rank").dropna(subset=["rank"])
    long["family"] = long["model"].map(family_map)
    grand = long["rank"].mean()
    ss_total = ((long["rank"] - grand) ** 2).sum()
    ss_between = long.groupby("family")["rank"].apply(
        lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    ss_within = ss_total - ss_between
    return {
        "ss_total": float(ss_total),
        "ss_between": float(ss_between),
        "ss_within": float(ss_within),
        "eta_squared": float(ss_between / ss_total) if ss_total else np.nan,
        "family_mean_rank": long.groupby("family")["rank"].mean().round(3).to_dict(),
    }


# --------------------------------------------------------------------------- #
# Meta-features and predict-the-winner
# --------------------------------------------------------------------------- #
def build_meta_features(datasets, load_dataset, datasets_yaml=None, data_dir=None):
    """Per-dataset meta-features: n, n_feat, imbalance, cat_frac, feat_types.

    `data_dir` is forwarded to load_dataset so the function works regardless of
    the working directory (notebooks run from notebooks/)."""
    feat_types = {}
    if datasets_yaml is not None:
        import yaml
        cfg = yaml.safe_load(Path(datasets_yaml).read_text())
        for _, dss in cfg.items():
            for name, m in dss.items():
                feat_types[name] = m.get("feature_types")
    dd = Path(data_dir) if data_dir is not None else None
    rows = []
    for ds in datasets:
        X, y, info = load_dataset(ds, dd) if dd is not None else load_dataset(ds)
        if info.task_type in ("binary", "multiclass"):
            vc = pd.Series(y).value_counts()
            imb = float(vc.max() / vc.min())
        else:
            imb = np.nan
        rows.append({
            "dataset": ds, "task": info.task_type, "n": len(y),
            "n_feat": X.shape[1], "imbalance": imb,
            "cat_frac": len(info.cat_columns) / max(1, X.shape[1]),
            "feat_types": feat_types.get(ds),
        })
    return pd.DataFrame(rows).set_index("dataset")


def winning_family_table(test_df, family_map, meta_df):
    """Per-dataset winner (by primary metric) + family, joined to meta-features."""
    winners = []
    for ds in test_df.dataset.unique():
        sub = test_df[test_df.dataset == ds]
        task = sub.task_type.iloc[0]
        col, higher = PRIMARY[task]
        s = sub.dropna(subset=[col])
        if s.empty:
            continue
        idx = s[col].idxmax() if higher else s[col].idxmin()
        best = s.loc[idx, "model"]
        winners.append({"dataset": ds, "winner": best,
                        "win_family": FAMILY_SHORT[family_map[best]]})
    W = pd.DataFrame(winners).set_index("dataset")
    drop = [c for c in ("task",) if c in meta_df.columns and c in W.columns]
    return W.join(meta_df.drop(columns=drop))


def winner_decision_tree(win_table, feature_cols=None, max_depth=2):
    """Tiny interpretable tree: meta-features -> winning family.

    DESCRIPTIVE only (N≈18, no held-out split). The splits illustrate the
    task/size/dimensionality reversals; they are not a validated predictor."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    if feature_cols is None:
        feature_cols = ["n", "n_feat", "imbalance", "cat_frac"]
    W = win_table.copy()
    W["imbalance"] = W["imbalance"].fillna(1.0)   # regression -> treat as balanced
    X = W[feature_cols].values
    y = W["win_family"].values
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
    return clf, export_text(clf, feature_names=list(feature_cols)), clf.score(X, y)


# --------------------------------------------------------------------------- #
# Risk / robustness from per-fold results
# --------------------------------------------------------------------------- #
def fold_robustness(fold_df):
    """Rank stability (std), mean and worst-case rank per model, ranked per
    (dataset, fold) on each task's primary metric. A high worst_rank means the
    model occasionally lands near the bottom on some fold -> production risk."""
    def _rank(g):
        task = g.task_type.iloc[0]
        col, higher = PRIMARY[task]
        return g.assign(rank=g[col].rank(ascending=not higher))
    fr = fold_df.groupby(["dataset", "fold"], group_keys=False).apply(_rank)
    out = fr.groupby("model")["rank"].agg(["mean", "std", "max"]).sort_values("mean")
    out.columns = ["mean_rank", "rank_std", "worst_rank"]
    return out.round(3)
