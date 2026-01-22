"""
Statistical Significance Tests for Model Comparison.

This module provides rigorous statistical tests for comparing machine learning
models, as required for academic thesis work.

Tests Implemented
-----------------
1. McNemar's Test: For comparing two classifiers on the same test set
2. Wilcoxon Signed-Rank Test: For comparing fold-wise performance scores
3. Friedman Test: For comparing 3+ models across multiple folds
4. Nemenyi Post-hoc Test: Pairwise comparisons after Friedman

Mathematical Foundation
-----------------------
McNemar's Test:
    Let n01 = instances where Model 1 is wrong and Model 2 is correct
    Let n10 = instances where Model 1 is correct and Model 2 is wrong

    Under H0 (both models have same error rate):
        chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)

    This follows chi-squared distribution with df=1.

Wilcoxon Signed-Rank Test:
    Non-parametric alternative to paired t-test.
    Tests whether the median of differences is zero.

Friedman Test:
    Non-parametric alternative to repeated measures ANOVA.
    Tests whether at least one model differs from others.

    chi2_F = 12N / (k(k+1)) * sum(R_j^2) - 3N(k+1)

    Where N = number of folds, k = number of models, R_j = average rank of model j.

References
----------
Demsar, J. (2006). Statistical comparisons of classifiers over multiple
data sets. JMLR, 7, 1-30.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats


class StatisticalSignificanceTester:
    """
    Statistical significance testing for machine learning model comparison.

    This class provides methods for rigorous comparison of sentiment analysis
    models, following best practices from Demsar (2006).

    Parameters
    ----------
    alpha : float, optional
        Significance level for hypothesis tests.
        Default: 0.05 (95% confidence)

    Attributes
    ----------
    alpha : float
        Significance level.

    Examples
    --------
    >>> tester = StatisticalSignificanceTester(alpha=0.05)
    >>>
    >>> # Compare two models using McNemar's test
    >>> y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    >>> pred_a = [0, 1, 2, 0, 0, 2, 0, 1, 2, 0]
    >>> pred_b = [0, 1, 2, 1, 1, 2, 0, 1, 1, 0]
    >>> result = tester.mcnemars_test(y_true, pred_a, pred_b)
    >>> print(f"p-value: {result['p_value']:.4f}")

    >>> # Compare fold-wise F1 scores using Wilcoxon
    >>> scores_a = [0.75, 0.78, 0.72, 0.80, 0.77]
    >>> scores_b = [0.73, 0.75, 0.70, 0.78, 0.74]
    >>> result = tester.wilcoxon_test(scores_a, scores_b)

    >>> # Compare 3+ models using Friedman test
    >>> scores = {
    ...     'LogReg': [0.74, 0.75, 0.73, 0.76, 0.74],
    ...     'SVM': [0.75, 0.76, 0.74, 0.77, 0.75],
    ...     'BERT': [0.85, 0.86, 0.84, 0.87, 0.85],
    ... }
    >>> result = tester.friedman_test(scores)
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def mcnemars_test(
        self,
        y_true: Union[List, np.ndarray],
        y_pred_1: Union[List, np.ndarray],
        y_pred_2: Union[List, np.ndarray],
        exact: bool = True,
    ) -> Dict[str, Any]:
        """
        McNemar's test for comparing two classifiers.

        This test compares the error rates of two classifiers on the same
        test set. It only considers instances where the two classifiers
        disagree (discordant pairs).

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_pred_1 : array-like
            Predictions from model 1.
        y_pred_2 : array-like
            Predictions from model 2.
        exact : bool, optional
            Whether to use exact binomial test (recommended for small samples).
            Default: True

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - statistic: McNemar's chi-squared statistic (or None if exact)
            - p_value: p-value of the test
            - n01: Model 1 wrong, Model 2 correct
            - n10: Model 1 correct, Model 2 wrong
            - n00: Both wrong
            - n11: Both correct
            - significant: Whether p < alpha
            - interpretation: Human-readable interpretation
            - effect_size: Odds ratio (n01 / n10) if applicable

        Notes
        -----
        Null Hypothesis (H0): Both classifiers have the same error rate.
        Alternative (H1): The classifiers have different error rates.

        Use exact=True when n01 + n10 < 25 (recommended).

        Examples
        --------
        >>> y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10
        >>> pred_a = [0, 1, 2, 0, 0, 2, 0, 1, 2, 0] * 10
        >>> pred_b = [0, 1, 2, 1, 1, 2, 0, 1, 1, 0] * 10
        >>> result = tester.mcnemars_test(y_true, pred_a, pred_b)
        """
        y_true = np.array(y_true)
        y_pred_1 = np.array(y_pred_1)
        y_pred_2 = np.array(y_pred_2)

        # Build contingency table
        correct_1 = y_pred_1 == y_true
        correct_2 = y_pred_2 == y_true

        n11 = np.sum(correct_1 & correct_2)      # Both correct
        n00 = np.sum(~correct_1 & ~correct_2)    # Both wrong
        n01 = np.sum(~correct_1 & correct_2)     # Model 1 wrong, Model 2 correct
        n10 = np.sum(correct_1 & ~correct_2)     # Model 1 correct, Model 2 wrong

        # Perform test
        if exact:
            # Exact binomial test
            n = n01 + n10
            if n == 0:
                p_value = 1.0
                statistic = None
            else:
                # Two-sided binomial test
                k = min(n01, n10)
                p_value = 2 * stats.binom.cdf(k, n, 0.5)
                p_value = min(p_value, 1.0)  # Cap at 1.0
                statistic = None
        else:
            # Chi-squared approximation with continuity correction
            n = n01 + n10
            if n == 0:
                p_value = 1.0
                statistic = 0.0
            else:
                statistic = (abs(n01 - n10) - 1) ** 2 / n
                p_value = 1 - stats.chi2.cdf(statistic, df=1)

        # Effect size (odds ratio)
        if n01 > 0 and n10 > 0:
            odds_ratio = n01 / n10
        elif n01 > 0:
            odds_ratio = float('inf')
        elif n10 > 0:
            odds_ratio = 0.0
        else:
            odds_ratio = 1.0

        # Interpretation
        significant = p_value < self.alpha
        if significant:
            if n01 > n10:
                interpretation = (
                    f"Model 2 significantly outperforms Model 1 "
                    f"(p={p_value:.4f} < {self.alpha}). "
                    f"Model 2 corrected {n01} errors that Model 1 made, "
                    f"while Model 1 only corrected {n10} of Model 2's errors."
                )
            else:
                interpretation = (
                    f"Model 1 significantly outperforms Model 2 "
                    f"(p={p_value:.4f} < {self.alpha}). "
                    f"Model 1 corrected {n10} errors that Model 2 made, "
                    f"while Model 2 only corrected {n01} of Model 1's errors."
                )
        else:
            interpretation = (
                f"No significant difference between models "
                f"(p={p_value:.4f} >= {self.alpha}). "
                f"Discordant pairs: Model 2 better on {n01} instances, "
                f"Model 1 better on {n10} instances."
            )

        return {
            "statistic": statistic,
            "p_value": float(p_value),
            "n01": int(n01),
            "n10": int(n10),
            "n00": int(n00),
            "n11": int(n11),
            "significant": significant,
            "interpretation": interpretation,
            "effect_size": float(odds_ratio),
            "test_type": "exact_binomial" if exact else "chi_squared",
        }

    def wilcoxon_test(
        self,
        scores_1: Union[List[float], np.ndarray],
        scores_2: Union[List[float], np.ndarray],
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test for paired samples.

        Non-parametric test for comparing two related samples
        (e.g., fold-wise performance scores).

        Parameters
        ----------
        scores_1 : array-like
            Performance scores from model 1 (e.g., F1 per fold).
        scores_2 : array-like
            Performance scores from model 2 (must be same length).
        alternative : str, optional
            Alternative hypothesis: 'two-sided', 'less', or 'greater'.
            Default: 'two-sided'

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - statistic: Wilcoxon W statistic
            - p_value: p-value of the test
            - mean_diff: Mean difference (scores_1 - scores_2)
            - median_diff: Median difference
            - significant: Whether p < alpha
            - interpretation: Human-readable interpretation
            - effect_size: r = Z / sqrt(N) (Rosenthal's r)

        Notes
        -----
        Null Hypothesis (H0): The median of differences is zero.

        Requirements:
        - At least 5 paired samples for reliable results
        - Samples should not have many ties

        Examples
        --------
        >>> scores_logreg = [0.74, 0.75, 0.73, 0.76, 0.74, 0.75, 0.72, 0.77, 0.73, 0.76]
        >>> scores_svm = [0.75, 0.76, 0.74, 0.77, 0.75, 0.76, 0.73, 0.78, 0.74, 0.77]
        >>> result = tester.wilcoxon_test(scores_logreg, scores_svm)
        """
        scores_1 = np.array(scores_1)
        scores_2 = np.array(scores_2)

        if len(scores_1) != len(scores_2):
            raise ValueError("Score arrays must have the same length")

        n = len(scores_1)
        differences = scores_1 - scores_2
        mean_diff = float(np.mean(differences))
        median_diff = float(np.median(differences))

        # Remove zeros (ties at 0)
        nonzero_diff = differences[differences != 0]

        if len(nonzero_diff) < 5:
            return {
                "statistic": None,
                "p_value": 1.0,
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "significant": False,
                "interpretation": (
                    f"Insufficient non-zero differences ({len(nonzero_diff)} < 5). "
                    "Cannot perform reliable Wilcoxon test."
                ),
                "effect_size": 0.0,
                "warning": "Sample size too small for reliable test",
            }

        # Perform Wilcoxon signed-rank test
        try:
            statistic, p_value = stats.wilcoxon(
                scores_1, scores_2,
                alternative=alternative,
                zero_method='wilcox',
            )
        except ValueError as e:
            return {
                "statistic": None,
                "p_value": 1.0,
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "significant": False,
                "interpretation": f"Test failed: {str(e)}",
                "effect_size": 0.0,
                "error": str(e),
            }

        # Effect size: r = Z / sqrt(N)
        # Z is approximated from the p-value
        z_score = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
        effect_size = abs(z_score) / np.sqrt(n)

        # Effect size interpretation
        if effect_size < 0.1:
            effect_interp = "negligible"
        elif effect_size < 0.3:
            effect_interp = "small"
        elif effect_size < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        # Interpretation
        significant = p_value < self.alpha
        better_model = "Model 1" if mean_diff > 0 else "Model 2"

        if significant:
            interpretation = (
                f"{better_model} significantly outperforms the other "
                f"(p={p_value:.4f} < {self.alpha}). "
                f"Mean difference: {mean_diff:.4f}, "
                f"Effect size r={effect_size:.3f} ({effect_interp})."
            )
        else:
            interpretation = (
                f"No significant difference between models "
                f"(p={p_value:.4f} >= {self.alpha}). "
                f"Mean difference: {mean_diff:.4f}."
            )

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "significant": significant,
            "interpretation": interpretation,
            "effect_size": float(effect_size),
            "effect_interpretation": effect_interp,
            "n_samples": n,
        }

    def friedman_test(
        self,
        scores: Union[Dict[str, List[float]], List[List[float]]],
    ) -> Dict[str, Any]:
        """
        Friedman test for comparing 3+ models.

        Non-parametric alternative to repeated measures ANOVA.
        Tests whether at least one model's performance differs
        significantly from the others.

        Parameters
        ----------
        scores : dict or list
            If dict: {model_name: [fold_scores]} for each model
            If list: [[fold_scores]] for each model (unnamed)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - statistic: Friedman chi-squared statistic
            - p_value: p-value of the test
            - significant: Whether p < alpha
            - ranks: Average rank for each model
            - interpretation: Human-readable interpretation
            - post_hoc: Results of Nemenyi post-hoc test if significant

        Notes
        -----
        Null Hypothesis (H0): All models have the same median performance.

        If the Friedman test is significant, a Nemenyi post-hoc test is
        automatically performed to identify which models differ.

        Examples
        --------
        >>> scores = {
        ...     'TF-IDF': [0.68, 0.69, 0.67, 0.70, 0.68],
        ...     'LogReg': [0.74, 0.75, 0.73, 0.76, 0.74],
        ...     'SVM': [0.75, 0.76, 0.74, 0.77, 0.75],
        ...     'BERT': [0.85, 0.86, 0.84, 0.87, 0.85],
        ... }
        >>> result = tester.friedman_test(scores)
        >>> print(result['ranks'])
        """
        # Convert to arrays
        if isinstance(scores, dict):
            model_names = list(scores.keys())
            score_matrix = np.array([scores[name] for name in model_names])
        else:
            model_names = [f"Model_{i}" for i in range(len(scores))]
            score_matrix = np.array(scores)

        n_models, n_folds = score_matrix.shape

        if n_models < 3:
            raise ValueError(
                "Friedman test requires at least 3 models. "
                "Use Wilcoxon test for 2 models."
            )

        # Perform Friedman test
        statistic, p_value = stats.friedmanchisquare(*score_matrix)

        # Compute average ranks (higher score = lower rank = better)
        ranks_per_fold = np.zeros_like(score_matrix)
        for fold in range(n_folds):
            fold_scores = score_matrix[:, fold]
            # Rank from 1 (best) to k (worst)
            ranks_per_fold[:, fold] = stats.rankdata(-fold_scores)

        average_ranks = np.mean(ranks_per_fold, axis=1)
        ranks_dict = {name: float(rank) for name, rank in zip(model_names, average_ranks)}

        significant = p_value < self.alpha

        # Sort models by rank
        sorted_models = sorted(ranks_dict.items(), key=lambda x: x[1])
        ranking_str = ", ".join([f"{name} ({rank:.2f})" for name, rank in sorted_models])

        if significant:
            interpretation = (
                f"Significant difference among models (p={p_value:.4f} < {self.alpha}). "
                f"Ranking: {ranking_str}. "
                "See post_hoc results for pairwise comparisons."
            )
            # Perform Nemenyi post-hoc test
            post_hoc = self._nemenyi_posthoc(model_names, average_ranks, n_folds)
        else:
            interpretation = (
                f"No significant difference among models (p={p_value:.4f} >= {self.alpha}). "
                f"Ranking: {ranking_str}."
            )
            post_hoc = None

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": significant,
            "ranks": ranks_dict,
            "interpretation": interpretation,
            "post_hoc": post_hoc,
            "n_models": n_models,
            "n_folds": n_folds,
        }

    def _nemenyi_posthoc(
        self,
        model_names: List[str],
        average_ranks: np.ndarray,
        n_folds: int,
    ) -> Dict[str, Any]:
        """
        Nemenyi post-hoc test for pairwise comparisons.

        Used after a significant Friedman test to identify which
        pairs of models differ significantly.

        Parameters
        ----------
        model_names : List[str]
            Names of the models.
        average_ranks : np.ndarray
            Average ranks for each model.
        n_folds : int
            Number of folds (datasets).

        Returns
        -------
        Dict[str, Any]
            Pairwise comparison results.
        """
        k = len(model_names)
        n = n_folds

        # Critical difference (CD) for Nemenyi test
        # CD = q_alpha * sqrt(k*(k+1)/(6*N))
        # q_alpha values for alpha=0.05 from Demsar (2006)
        q_alpha_table = {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
            6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
        }
        q_alpha = q_alpha_table.get(k, 3.0)  # Default for large k

        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

        # Pairwise comparisons
        comparisons = []
        for i in range(k):
            for j in range(i + 1, k):
                rank_diff = abs(average_ranks[i] - average_ranks[j])
                is_significant = rank_diff >= cd

                comparisons.append({
                    "model_1": model_names[i],
                    "model_2": model_names[j],
                    "rank_diff": float(rank_diff),
                    "critical_difference": float(cd),
                    "significant": is_significant,
                })

        return {
            "critical_difference": float(cd),
            "q_alpha": q_alpha,
            "comparisons": comparisons,
        }

    def bonferroni_correction(
        self,
        p_values: Union[List[float], Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Apply Bonferroni correction for multiple comparisons.

        The Bonferroni correction adjusts p-values to control the
        family-wise error rate when performing multiple hypothesis tests.

        Parameters
        ----------
        p_values : list or dict
            Original p-values from multiple tests.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - original: Original p-values
            - adjusted: Bonferroni-adjusted p-values
            - significant: Which tests remain significant
            - n_tests: Number of tests
            - adjusted_alpha: Adjusted significance level

        Notes
        -----
        Adjusted p-value = min(original_p * n_tests, 1.0)
        Adjusted alpha = alpha / n_tests

        The Bonferroni correction is conservative. For less conservative
        alternatives, consider Holm-Bonferroni or Benjamini-Hochberg.
        """
        if isinstance(p_values, dict):
            names = list(p_values.keys())
            p_array = np.array(list(p_values.values()))
        else:
            names = [f"Test_{i}" for i in range(len(p_values))]
            p_array = np.array(p_values)

        n_tests = len(p_array)
        adjusted_alpha = self.alpha / n_tests
        adjusted_p = np.minimum(p_array * n_tests, 1.0)

        original_dict = {name: float(p) for name, p in zip(names, p_array)}
        adjusted_dict = {name: float(p) for name, p in zip(names, adjusted_p)}
        significant_dict = {name: bool(p < self.alpha) for name, p in zip(names, adjusted_p)}

        return {
            "original": original_dict,
            "adjusted": adjusted_dict,
            "significant": significant_dict,
            "n_tests": n_tests,
            "adjusted_alpha": adjusted_alpha,
            "method": "bonferroni",
        }

    def summary_report(
        self,
        results: Dict[str, Dict],
    ) -> str:
        """
        Generate a formatted summary report of statistical tests.

        Parameters
        ----------
        results : Dict[str, Dict]
            Dictionary of test results from various methods.

        Returns
        -------
        str
            Formatted markdown report.
        """
        lines = [
            "# Statistical Significance Analysis Report",
            "",
            f"**Significance Level (alpha):** {self.alpha}",
            "",
        ]

        for test_name, result in results.items():
            lines.append(f"## {test_name}")
            lines.append("")

            if "p_value" in result:
                lines.append(f"- **p-value:** {result['p_value']:.6f}")

            if "statistic" in result and result["statistic"] is not None:
                lines.append(f"- **Test statistic:** {result['statistic']:.4f}")

            if "significant" in result:
                sig_str = "Yes" if result["significant"] else "No"
                lines.append(f"- **Significant:** {sig_str}")

            if "interpretation" in result:
                lines.append(f"- **Interpretation:** {result['interpretation']}")

            lines.append("")

        return "\n".join(lines)
