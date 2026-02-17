"""Tests for src.evaluation.statistical_tests — Friedman, Wilcoxon, bootstrap CI, Cohen's d, ranks."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical_tests import (
    bootstrap_ci,
    cohens_d,
    compute_average_ranks,
    friedman_test,
    nemenyi_test,
    pairwise_cohens_d,
    pairwise_wilcoxon,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clearly_different_scores():
    """Three models with clearly separated performance across 10 datasets.

    Model A dominates, B is middling, C is worst — Friedman should reject.
    """
    return pd.DataFrame({
        "ModelA": [0.95, 0.93, 0.94, 0.96, 0.92, 0.91, 0.97, 0.93, 0.95, 0.94],
        "ModelB": [0.80, 0.78, 0.82, 0.81, 0.79, 0.77, 0.83, 0.80, 0.81, 0.79],
        "ModelC": [0.60, 0.62, 0.58, 0.61, 0.59, 0.63, 0.57, 0.60, 0.61, 0.59],
    }, index=[f"d{i}" for i in range(10)])


@pytest.fixture
def identical_scores():
    """Three models with identical scores — Friedman should NOT reject."""
    vals = [0.85, 0.86, 0.84, 0.87, 0.83, 0.88, 0.85, 0.86, 0.84, 0.85]
    return pd.DataFrame({
        "ModelA": vals,
        "ModelB": vals,
        "ModelC": vals,
    }, index=[f"d{i}" for i in range(10)])


@pytest.fixture
def scores_with_ties():
    """Scores where some models tie on some datasets."""
    return pd.DataFrame({
        "ModelA": [0.90, 0.85, 0.90, 0.88],
        "ModelB": [0.90, 0.80, 0.85, 0.88],
        "ModelC": [0.80, 0.85, 0.80, 0.82],
    }, index=["d0", "d1", "d2", "d3"])


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

class TestFriedmanTest:
    def test_rejects_when_models_differ(self, clearly_different_scores):
        """Friedman should reject H0 when models have clearly different rankings."""
        result = friedman_test(clearly_different_scores)

        assert "statistic" in result
        assert "p_value" in result
        assert "reject_null" in result
        assert result["reject_null"] == True
        assert result["p_value"] < 0.05
        assert result["statistic"] > 0

    def test_does_not_reject_when_identical(self, identical_scores):
        """Friedman should NOT reject when all models have identical scores."""
        result = friedman_test(identical_scores)

        assert result["reject_null"] == False
        # p-value may be NaN when all scores are identical (division by zero in
        # Friedman statistic), which correctly yields reject_null=False
        assert result["p_value"] >= 0.05 or np.isnan(result["p_value"])

    def test_return_types(self, clearly_different_scores):
        """Verify return types are numeric and boolean."""
        result = friedman_test(clearly_different_scores)

        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["reject_null"], (bool, np.bool_))

    def test_p_value_bounded(self, clearly_different_scores):
        """p-value should be in [0, 1]."""
        result = friedman_test(clearly_different_scores)
        assert 0 <= result["p_value"] <= 1


# ---------------------------------------------------------------------------
# Nemenyi test
# ---------------------------------------------------------------------------

class TestNemenyiTest:
    def test_pvalue_matrix_shape_and_symmetry(self, clearly_different_scores):
        """Nemenyi p-value matrix should be square, symmetric, with 1s on diagonal."""
        pvals = nemenyi_test(clearly_different_scores, higher_is_better=True)

        n_models = len(clearly_different_scores.columns)
        assert pvals.shape == (n_models, n_models)

        # Symmetry
        for i in range(n_models):
            for j in range(n_models):
                assert pvals.iloc[i, j] == pytest.approx(pvals.iloc[j, i], abs=1e-10)

        # Diagonal should be 1.0
        for i in range(n_models):
            assert pvals.iloc[i, i] == pytest.approx(1.0)

    def test_pvalues_bounded(self, clearly_different_scores):
        """All p-values should be in [0, 1]."""
        pvals = nemenyi_test(clearly_different_scores, higher_is_better=True)

        assert (pvals.values >= 0).all()
        assert (pvals.values <= 1.0 + 1e-10).all()

    def test_clearly_different_models_have_low_pvalues(self, clearly_different_scores):
        """A vs C should have a very low p-value (far apart in ranking)."""
        pvals = nemenyi_test(clearly_different_scores, higher_is_better=True)

        # ModelA vs ModelC: clearly different
        assert pvals.loc["ModelA", "ModelC"] < 0.05

    def test_lower_is_better_flag(self):
        """When higher_is_better=False (e.g. RMSE), lower scores should rank better."""
        # ModelA has lowest RMSE (best), ModelC has highest (worst)
        df = pd.DataFrame({
            "ModelA": [0.10, 0.12, 0.11, 0.09, 0.13, 0.10, 0.11, 0.12, 0.10, 0.11],
            "ModelB": [0.30, 0.32, 0.31, 0.29, 0.33, 0.30, 0.31, 0.32, 0.30, 0.31],
            "ModelC": [0.50, 0.52, 0.51, 0.49, 0.53, 0.50, 0.51, 0.52, 0.50, 0.51],
        }, index=[f"d{i}" for i in range(10)])

        pvals = nemenyi_test(df, higher_is_better=False)

        # A vs C should be significant (A is much better when lower=better)
        assert pvals.loc["ModelA", "ModelC"] < 0.05


# ---------------------------------------------------------------------------
# Pairwise Wilcoxon
# ---------------------------------------------------------------------------

class TestPairwiseWilcoxon:
    def test_identical_scores_not_significant(self, identical_scores):
        """Identical scores should give p=1.0 for all pairs."""
        pvals = pairwise_wilcoxon(identical_scores)

        n = len(identical_scores.columns)
        assert pvals.shape == (n, n)

        # All off-diagonal should be 1.0 (identical scores)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert pvals.iloc[i, j] == pytest.approx(1.0)

    def test_shape_and_symmetry(self, clearly_different_scores):
        """Result should be square, symmetric, diagonal=1."""
        pvals = pairwise_wilcoxon(clearly_different_scores)
        n = len(clearly_different_scores.columns)

        assert pvals.shape == (n, n)

        for i in range(n):
            assert pvals.iloc[i, i] == pytest.approx(1.0)
            for j in range(i + 1, n):
                assert pvals.iloc[i, j] == pytest.approx(pvals.iloc[j, i])

    def test_different_models_low_pvalue(self, clearly_different_scores):
        """Clearly different models should have low p-values even after Holm correction."""
        pvals = pairwise_wilcoxon(clearly_different_scores, correction="holm")

        # A vs C: very different
        assert pvals.loc["ModelA", "ModelC"] < 0.05

    def test_holm_correction_increases_pvalues(self, clearly_different_scores):
        """Holm-corrected p-values should be >= uncorrected p-values."""
        pvals_uncorrected = pairwise_wilcoxon(clearly_different_scores, correction=None)
        pvals_holm = pairwise_wilcoxon(clearly_different_scores, correction="holm")

        # For each off-diagonal pair, corrected >= uncorrected
        n = len(clearly_different_scores.columns)
        for i in range(n):
            for j in range(i + 1, n):
                assert pvals_holm.iloc[i, j] >= pvals_uncorrected.iloc[i, j] - 1e-10

    def test_pvalues_bounded(self, clearly_different_scores):
        """All p-values should be in [0, 1]."""
        pvals = pairwise_wilcoxon(clearly_different_scores)

        assert (pvals.values >= 0).all()
        assert (pvals.values <= 1.0 + 1e-10).all()


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_mean_within_bounds(self):
        """The sample mean should fall within the bootstrap CI."""
        scores = np.array([0.90, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86, 0.89])
        lower, upper = bootstrap_ci(scores, seed=42)

        mean = scores.mean()
        assert lower <= mean <= upper

    def test_ci_bounds_ordered(self):
        """Lower bound should be < upper bound."""
        scores = np.array([0.90, 0.85, 0.88, 0.92, 0.87])
        lower, upper = bootstrap_ci(scores, seed=42)

        assert lower < upper

    def test_narrower_ci_with_less_variance(self):
        """Low-variance data should produce a narrower CI than high-variance data."""
        low_var = np.array([0.90, 0.91, 0.90, 0.91, 0.90, 0.91, 0.90, 0.91])
        high_var = np.array([0.50, 0.99, 0.55, 0.95, 0.52, 0.98, 0.51, 0.97])

        low_lower, low_upper = bootstrap_ci(low_var, seed=42)
        high_lower, high_upper = bootstrap_ci(high_var, seed=42)

        low_width = low_upper - low_lower
        high_width = high_upper - high_lower

        assert low_width < high_width

    def test_deterministic_with_same_seed(self):
        """Same seed should produce the same CI."""
        scores = np.array([0.90, 0.85, 0.88, 0.92, 0.87])

        lower1, upper1 = bootstrap_ci(scores, seed=123)
        lower2, upper2 = bootstrap_ci(scores, seed=123)

        assert lower1 == pytest.approx(lower2)
        assert upper1 == pytest.approx(upper2)

    def test_different_seeds_may_differ(self):
        """Different seeds should (typically) produce slightly different CIs."""
        scores = np.array([0.90, 0.85, 0.88, 0.92, 0.87])

        lower1, upper1 = bootstrap_ci(scores, seed=1)
        lower2, upper2 = bootstrap_ci(scores, seed=999)

        # They should be close but not necessarily identical
        # (probabilistically, they are almost never identical)
        assert isinstance(lower1, float)
        assert isinstance(upper1, float)

    def test_returns_floats(self):
        """Return values should be plain floats."""
        scores = np.array([0.90, 0.85, 0.88])
        lower, upper = bootstrap_ci(scores, seed=42)

        assert isinstance(lower, float)
        assert isinstance(upper, float)


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_identical_scores_return_zero(self):
        """Identical arrays should give d=0."""
        a = np.array([0.90, 0.85, 0.88, 0.92, 0.87])
        b = np.array([0.90, 0.85, 0.88, 0.92, 0.87])

        d = cohens_d(a, b)
        assert d == pytest.approx(0.0)

    def test_clearly_different_large_d(self):
        """Clearly different paired scores should give a large |d|."""
        a = np.array([0.95, 0.93, 0.94, 0.96, 0.92])
        b = np.array([0.60, 0.62, 0.58, 0.61, 0.59])

        d = cohens_d(a, b)
        # Large positive d (A >> B consistently)
        assert d > 2.0

    def test_sign_convention(self):
        """Positive d means A > B; negative means A < B."""
        a = np.array([0.90, 0.91, 0.92, 0.93, 0.94])
        b = np.array([0.80, 0.81, 0.82, 0.83, 0.84])

        d_ab = cohens_d(a, b)
        d_ba = cohens_d(b, a)

        assert d_ab > 0
        assert d_ba < 0
        assert d_ab == pytest.approx(-d_ba)

    def test_hand_computed_value(self):
        """Verify against manual computation: d = mean(diff) / std(diff, ddof=1)."""
        a = np.array([3.0, 5.0, 7.0, 9.0])
        b = np.array([1.0, 3.0, 5.0, 7.0])

        diff = a - b  # [2, 2, 2, 2]
        expected_d = diff.mean() / diff.std(ddof=1)  # 2.0 / 0.0 -> 0 by convention

        d = cohens_d(a, b)

        # All differences are exactly 2, so std=0 -> d=0 by the implementation
        assert d == pytest.approx(0.0)

    def test_nonzero_hand_computed(self):
        """Verify against manual computation with non-constant differences."""
        a = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        b = np.array([8.0, 11.0, 12.0, 13.0, 14.0])

        diff = a - b  # [2, 1, 2, 3, 4]
        expected_d = diff.mean() / diff.std(ddof=1)

        d = cohens_d(a, b)
        assert d == pytest.approx(expected_d)


# ---------------------------------------------------------------------------
# Pairwise Cohen's d
# ---------------------------------------------------------------------------

class TestPairwiseCohensD:
    def test_shape_and_diagonal(self, clearly_different_scores):
        """Matrix should be square with zeros on diagonal."""
        d_matrix = pairwise_cohens_d(clearly_different_scores)
        n = len(clearly_different_scores.columns)

        assert d_matrix.shape == (n, n)
        for i in range(n):
            assert d_matrix.iloc[i, i] == pytest.approx(0.0)

    def test_antisymmetry(self, clearly_different_scores):
        """d(A,B) should equal -d(B,A)."""
        d_matrix = pairwise_cohens_d(clearly_different_scores)
        n = len(clearly_different_scores.columns)

        for i in range(n):
            for j in range(i + 1, n):
                assert d_matrix.iloc[i, j] == pytest.approx(-d_matrix.iloc[j, i])


# ---------------------------------------------------------------------------
# Average ranks
# ---------------------------------------------------------------------------

class TestComputeAverageRanks:
    def test_clear_ranking_higher_is_better(self, clearly_different_scores):
        """ModelA should have best (lowest) rank when higher is better."""
        ranks = compute_average_ranks(clearly_different_scores, higher_is_better=True)

        assert ranks["ModelA"] < ranks["ModelB"]
        assert ranks["ModelB"] < ranks["ModelC"]
        # Best model gets rank 1
        assert ranks["ModelA"] == pytest.approx(1.0)
        assert ranks["ModelC"] == pytest.approx(3.0)

    def test_clear_ranking_lower_is_better(self):
        """When lower is better (e.g. RMSE), model with lowest score should rank 1."""
        df = pd.DataFrame({
            "ModelA": [0.10, 0.12, 0.11],
            "ModelB": [0.30, 0.32, 0.31],
            "ModelC": [0.50, 0.52, 0.51],
        }, index=["d0", "d1", "d2"])

        ranks = compute_average_ranks(df, higher_is_better=False)

        assert ranks["ModelA"] < ranks["ModelB"]
        assert ranks["ModelB"] < ranks["ModelC"]
        assert ranks["ModelA"] == pytest.approx(1.0)

    def test_ties_handled_with_average(self, scores_with_ties):
        """Tied scores should get average rank."""
        ranks = compute_average_ranks(scores_with_ties, higher_is_better=True)

        # On d0: A=0.90, B=0.90, C=0.80 -> A and B tie at rank 1.5, C gets 3
        # On d1: A=0.85, B=0.80, C=0.85 -> A and C tie at rank 1.5, B gets 3
        # On d2: A=0.90, B=0.85, C=0.80 -> A=1, B=2, C=3
        # On d3: A=0.88, B=0.88, C=0.82 -> A and B tie at rank 1.5, C gets 3
        # Average ranks: A=(1.5+1.5+1+1.5)/4=1.375, B=(1.5+3+2+1.5)/4=2.0, C=(3+1.5+3+3)/4=2.625

        assert ranks["ModelA"] == pytest.approx(1.375)
        assert ranks["ModelB"] == pytest.approx(2.0)
        assert ranks["ModelC"] == pytest.approx(2.625)

    def test_returns_sorted_series(self, clearly_different_scores):
        """Result should be a Series sorted by rank (ascending)."""
        ranks = compute_average_ranks(clearly_different_scores, higher_is_better=True)

        assert isinstance(ranks, pd.Series)
        # Verify sorted ascending
        assert list(ranks.values) == sorted(ranks.values)

    def test_ranks_sum_to_expected(self, clearly_different_scores):
        """Average ranks across models should sum to k*(k+1)/2 for k models."""
        ranks = compute_average_ranks(clearly_different_scores, higher_is_better=True)
        k = len(clearly_different_scores.columns)

        # Sum of average ranks should equal sum of 1..k = k(k+1)/2
        assert ranks.sum() == pytest.approx(k * (k + 1) / 2)
