#!/usr/bin/env python3
"""CLI: Aggregate results and run statistical tests.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --results-dir results
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.statistical_tests import (
    bootstrap_ci,
    compute_average_ranks,
    create_cd_diagram_data,
    friedman_test,
    nemenyi_test,
    pairwise_cohens_d,
    pairwise_wilcoxon,
    plot_cd_diagram,
)
from src.utils.config import load_experiment_config
from src.utils.logging import setup_logging, get_logger


def aggregate_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all raw JSON results and aggregate into summary DataFrames."""
    logger = get_logger()
    raw_dir = results_dir / "raw"

    if not raw_dir.exists():
        logger.warning(f"No raw results found in {raw_dir}")
        return {}

    results = []
    for path in sorted(raw_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        results.append(data)

    if not results:
        logger.warning("No result files found")
        return {}

    # Build per-fold summary
    fold_rows = []
    for r in results:
        for fold in r["fold_results"]:
            row = {
                "model": r["model"],
                "dataset": r["dataset"],
                "task_type": r["task_type"],
                **fold,
            }
            fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)

    # Build test-set summary
    test_rows = []
    for r in results:
        row = {
            "model": r["model"],
            "dataset": r["dataset"],
            "task_type": r["task_type"],
            "tuning_time_s": r["tuning_time_s"],
            **r["test_metrics"],
        }
        test_rows.append(row)

    test_df = pd.DataFrame(test_rows)

    # Build cross-validation summary (mean +/- std across folds)
    cv_summary = fold_df.groupby(["model", "dataset"]).agg(
        ["mean", "std"]
    ).reset_index()

    # Save aggregated CSVs
    agg_dir = results_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    fold_df.to_csv(agg_dir / "fold_results.csv", index=False)
    test_df.to_csv(agg_dir / "test_results.csv", index=False)
    cv_summary.to_csv(agg_dir / "cv_summary.csv", index=True)

    logger.info(f"Aggregated results saved to {agg_dir}")

    return {"fold_results": fold_df, "test_results": test_df, "cv_summary": cv_summary}


def _get_task_metric_specs() -> list[tuple[str, str, bool]]:
    """Return (task_type, metric_name, higher_is_better) from experiment config."""
    exp_cfg = load_experiment_config()
    return [
        ("binary", exp_cfg.primary_metric_binary, True),        # roc_auc: higher is better
        ("multiclass", exp_cfg.primary_metric_multiclass, False),  # log_loss: lower is better
        ("regression", exp_cfg.primary_metric_regression, False),  # rmse: lower is better
    ]


def compute_bootstrap_cis(test_df: pd.DataFrame, results_dir: Path):
    """Compute bootstrap 95% CIs for each (model, task_type) on the primary metric."""
    logger = get_logger()
    agg_dir = results_dir / "aggregated"

    ci_rows = []
    for task_group, metric, _higher_is_better in _get_task_metric_specs():
        subset = test_df[test_df["task_type"] == task_group]
        if subset.empty or metric not in subset.columns:
            continue

        for model in subset["model"].unique():
            scores = subset.loc[subset["model"] == model, metric].values
            if len(scores) < 2:
                continue
            lower, upper = bootstrap_ci(scores)
            ci_rows.append({
                "task_type": task_group,
                "metric": metric,
                "model": model,
                "mean": scores.mean(),
                "ci_lower": lower,
                "ci_upper": upper,
                "n_datasets": len(scores),
            })

    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        ci_df.to_csv(agg_dir / "bootstrap_ci.csv", index=False)
        logger.info(f"Bootstrap CIs saved to {agg_dir / 'bootstrap_ci.csv'}")
        logger.info(f"\n{ci_df.to_string(index=False)}")


def run_statistical_tests(results_dir: Path):
    """Run Friedman, Nemenyi, Wilcoxon, and generate CD diagrams."""
    logger = get_logger()
    agg_dir = results_dir / "aggregated"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(agg_dir / "test_results.csv")

    # Bootstrap confidence intervals
    compute_bootstrap_cis(test_df, results_dir)

    # Split by task type group (metrics read from experiment config)
    for task_group, metric, higher_is_better in _get_task_metric_specs():
        subset = test_df[test_df["task_type"] == task_group]
        if subset.empty or metric not in subset.columns:
            continue

        # Pivot: rows=datasets, cols=models
        pivot = subset.pivot(index="dataset", columns="model", values=metric)
        pivot = pivot.dropna(axis=1, how="all")

        # Drop rows with any missing values before statistical tests
        original_len = len(pivot)
        pivot = pivot.dropna()
        if len(pivot) < original_len:
            logger.warning(
                f"Dropped {original_len - len(pivot)} rows with missing values "
                f"from {task_group} pivot before statistical tests"
            )

        if pivot.shape[1] < 3 or pivot.shape[0] < 3:
            logger.warning(f"Not enough data for {task_group} statistical tests")
            continue

        logger.info(f"\n=== Statistical Tests: {task_group} ({metric}) ===")

        # Friedman test
        friedman = friedman_test(pivot)
        logger.info(f"Friedman: stat={friedman['statistic']:.2f}, p={friedman['p_value']:.4f}")

        # Nemenyi post-hoc (only if Friedman rejects the null)
        if friedman["reject_null"]:
            nemenyi_pvals = nemenyi_test(pivot, higher_is_better=higher_is_better)
            nemenyi_pvals.to_csv(agg_dir / f"nemenyi_{task_group}.csv")
            logger.info(f"Nemenyi p-values saved to {agg_dir / f'nemenyi_{task_group}.csv'}")
        else:
            logger.info(f"Friedman not significant (p={friedman['p_value']:.4f}), skipping Nemenyi post-hoc")

        # Wilcoxon pairwise with Holm-Bonferroni correction
        wilcoxon = pairwise_wilcoxon(pivot, correction="holm")
        wilcoxon.to_csv(agg_dir / f"wilcoxon_{task_group}.csv")
        logger.info(f"Wilcoxon p-values (Holm-corrected) saved to {agg_dir / f'wilcoxon_{task_group}.csv'}")

        # Cohen's d effect sizes
        effect_sizes = pairwise_cohens_d(pivot)
        effect_sizes.to_csv(agg_dir / f"cohens_d_{task_group}.csv")
        logger.info(f"Cohen's d effect sizes saved to {agg_dir / f'cohens_d_{task_group}.csv'}")

        # Average ranks
        ranks = compute_average_ranks(pivot, higher_is_better=higher_is_better)
        logger.info(f"Average ranks:\n{ranks}")

        # CD diagram
        save_path = str(fig_dir / f"cd_diagram_{task_group}.png")
        plot_cd_diagram(
            pivot,
            title=f"Critical Difference — {task_group.title()} ({metric})",
            higher_is_better=higher_is_better,
            save_path=save_path,
        )

    logger.info("Statistical tests complete")


@click.command()
@click.option("--results-dir", default="results", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
def main(results_dir, verbose):
    """Aggregate results and run statistical tests."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    results_path = Path(results_dir)

    aggregate_results(results_path)
    run_statistical_tests(results_path)


if __name__ == "__main__":
    main()
