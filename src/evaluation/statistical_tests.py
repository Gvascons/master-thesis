"""Statistical tests: Friedman, Nemenyi, Wilcoxon, and critical difference diagrams."""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("tabular_benchmark")


def friedman_test(scores_df: pd.DataFrame) -> dict:
    """Run the Friedman test.

    Args:
        scores_df: DataFrame where each row is a dataset and each column is a model.
                   Values are the primary metric for that (dataset, model) pair.

    Returns:
        dict with 'statistic', 'p_value', and 'reject_null' (at alpha=0.05).
    """
    model_scores = [scores_df[col].values for col in scores_df.columns]
    stat, p_value = stats.friedmanchisquare(*model_scores)
    return {
        "statistic": stat,
        "p_value": p_value,
        "reject_null": p_value < 0.05,
    }


def nemenyi_test(
    scores_df: pd.DataFrame,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Compute pairwise Nemenyi post-hoc test p-values after Friedman.

    Uses scikit-posthocs for proper Nemenyi-Friedman p-values.

    Args:
        scores_df: DataFrame where rows=datasets, cols=models.
        higher_is_better: If True, higher metric values are better (rank 1 = highest).
                          If False (e.g. RMSE), lower values are better (rank 1 = lowest).

    Returns:
        DataFrame of pairwise p-values (symmetric, diagonal = 1.0).
    """
    import scikit_posthocs as sp

    # Pass wide-format DataFrame directly (rows=blocks/datasets, cols=groups/models).
    # If lower is better, negate scores so ranking direction is correct.
    data = -scores_df if not higher_is_better else scores_df

    p_values = sp.posthoc_nemenyi_friedman(data)

    return p_values


def pairwise_wilcoxon(
    scores_df: pd.DataFrame,
    correction: str = "holm",
) -> pd.DataFrame:
    """Compute pairwise Wilcoxon signed-rank test p-values with multiple-testing correction.

    Args:
        scores_df: DataFrame where rows=datasets, cols=models.
        correction: Multiple-testing correction method passed to
                    statsmodels.stats.multitest.multipletests.
                    Default is 'holm' (Holm-Bonferroni). Use None to skip correction.

    Returns:
        DataFrame of pairwise corrected p-values.
    """
    from statsmodels.stats.multitest import multipletests

    models = scores_df.columns.tolist()
    n = len(models)
    p_values = pd.DataFrame(np.ones((n, n)), index=models, columns=models)

    # Collect raw p-values for all pairs
    pairs = list(combinations(range(n), 2))
    raw_pvals = []

    for i, j in pairs:
        a = scores_df[models[i]].values
        b = scores_df[models[j]].values
        if np.allclose(a, b):
            raw_pvals.append(1.0)
        else:
            _, p = stats.wilcoxon(a, b, alternative="two-sided")
            raw_pvals.append(p)

    # Apply Holm-Bonferroni (or other) correction
    raw_pvals = np.array(raw_pvals)
    if correction is not None and len(raw_pvals) > 0:
        _, corrected_pvals, _, _ = multipletests(raw_pvals, method=correction)
    else:
        corrected_pvals = raw_pvals

    # Fill symmetric matrix
    for idx, (i, j) in enumerate(pairs):
        p_values.iloc[i, j] = corrected_pvals[idx]
        p_values.iloc[j, i] = corrected_pvals[idx]

    return p_values


def compute_average_ranks(scores_df: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """Compute average ranks across datasets.

    Args:
        scores_df: rows=datasets, cols=models.
        higher_is_better: If True, rank 1 = best (highest) score.

    Returns:
        Series of average ranks indexed by model name.
    """
    ranks = scores_df.rank(axis=1, ascending=not higher_is_better)
    return ranks.mean(axis=0).sort_values()


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        scores: 1-D array of metric values (e.g. per-fold or per-dataset).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means[b] = sample.mean()

    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples.

    Uses the difference scores divided by their standard deviation,
    which is appropriate for paired/repeated-measures designs.

    Args:
        scores_a: Metric values for model A across datasets.
        scores_b: Metric values for model B across datasets.

    Returns:
        Cohen's d (positive means A > B).
    """
    diff = scores_a - scores_b
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def pairwise_cohens_d(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Cohen's d effect sizes between all model pairs.

    Args:
        scores_df: DataFrame where rows=datasets, cols=models.

    Returns:
        DataFrame of pairwise Cohen's d values.
    """
    models = scores_df.columns.tolist()
    n = len(models)
    d_matrix = pd.DataFrame(np.zeros((n, n)), index=models, columns=models)

    for i in range(n):
        for j in range(n):
            if i != j:
                d_matrix.iloc[i, j] = cohens_d(
                    scores_df[models[i]].values,
                    scores_df[models[j]].values,
                )

    return d_matrix


def create_cd_diagram_data(scores_df: pd.DataFrame, higher_is_better: bool = True) -> dict:
    """Prepare data for a critical difference diagram.

    Returns dict with average ranks, CD value, and groups of models that are not
    significantly different (based on Nemenyi post-hoc test at alpha=0.05).
    """
    n_datasets = len(scores_df)
    n_models = len(scores_df.columns)

    avg_ranks = compute_average_ranks(scores_df, higher_is_better)

    # Compute groups from Nemenyi p-values
    nemenyi_pvals = nemenyi_test(scores_df, higher_is_better=higher_is_better)

    # Find groups (cliques) of models not significantly different
    sorted_models = avg_ranks.index.tolist()
    groups = []

    for i in range(n_models):
        group = [sorted_models[i]]
        for j in range(i + 1, n_models):
            if nemenyi_pvals.loc[sorted_models[i], sorted_models[j]] > 0.05:
                group.append(sorted_models[j])
        if len(group) > 1:
            groups.append(group)

    # Remove subgroups
    final_groups = []
    for g in groups:
        is_subgroup = any(set(g) < set(g2) for g2 in groups)
        if not is_subgroup:
            final_groups.append(g)

    # Compute CD for display (using q_alpha from Studentized Range approximation)
    # This is purely for the visual CD bar; actual significance comes from Nemenyi p-values
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(0.95, n_models, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))

    return {
        "average_ranks": avg_ranks.to_dict(),
        "cd": cd,
        "n_datasets": n_datasets,
        "n_models": n_models,
        "groups": final_groups,
    }


def plot_cd_diagram(
    scores_df: pd.DataFrame,
    title: str = "Critical Difference Diagram",
    higher_is_better: bool = True,
    save_path: str | None = None,
):
    """Plot a critical difference diagram.

    Uses matplotlib to draw the CD diagram showing average ranks and
    groups of statistically indistinguishable models.
    """
    import matplotlib.pyplot as plt

    data = create_cd_diagram_data(scores_df, higher_is_better)
    avg_ranks = data["average_ranks"]
    cd = data["cd"]
    groups = data["groups"]

    n_models = len(avg_ranks)
    sorted_items = sorted(avg_ranks.items(), key=lambda x: x[1])

    fig, ax = plt.subplots(1, 1, figsize=(10, max(3, n_models * 0.4)))

    # Draw axis
    min_rank = 1
    max_rank = n_models
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(-0.5, n_models + len(groups) * 0.3)

    # Draw rank axis at top
    ax.hlines(n_models + 0.2, min_rank, max_rank, colors="black", linewidth=1)
    for r in range(min_rank, max_rank + 1):
        ax.vlines(r, n_models + 0.1, n_models + 0.3, colors="black", linewidth=1)
        ax.text(r, n_models + 0.4, str(r), ha="center", va="bottom", fontsize=9)

    # Draw CD indicator
    cd_x = (min_rank + max_rank) / 2
    ax.hlines(n_models + 0.8, cd_x - cd / 2, cd_x + cd / 2, colors="red", linewidth=2)
    ax.text(cd_x, n_models + 0.9, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=9, color="red")

    # Draw model labels and ranks
    for i, (model, rank) in enumerate(sorted_items):
        y = n_models - 1 - i
        ax.plot(rank, y, "ko", markersize=6)
        side = "left" if rank <= (min_rank + max_rank) / 2 else "right"
        offset = -0.15 if side == "left" else 0.15
        ha = "right" if side == "left" else "left"
        ax.text(rank + offset, y, f"{model} ({rank:.2f})", ha=ha, va="center", fontsize=10)
        ax.hlines(y, rank, rank, colors="gray", linewidth=0.5, linestyles="--")

    # Draw group bars (models not significantly different)
    for gi, group in enumerate(groups):
        group_ranks = [avg_ranks[m] for m in group]
        y_bar = -0.3 - gi * 0.25
        ax.hlines(y_bar, min(group_ranks), max(group_ranks), colors="blue",
                  linewidth=3, alpha=0.6)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        logger.info(f"CD diagram saved to {save_path}")

    return fig
