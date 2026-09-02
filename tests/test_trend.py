import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, linregress, pearsonr, spearmanr, theilslopes

from csdtools import (
    AVAILABLE_TREND_METHODS,
    analyze_timeseries,
    plot_analysis,
    plot_significance,
)
from csdtools.significance import surrogate_p_value
from csdtools.trend import assess_trend


class TrendAssessmentTests(unittest.TestCase):
    def test_all_trend_methods_are_registered(self):
        self.assertEqual(
            AVAILABLE_TREND_METHODS,
            ("kendall", "spearman", "pearson", "linear", "theil_sen"),
        )

    def test_correlation_methods_match_scipy(self):
        times = np.arange(12, dtype=float) * 0.5
        indicator = np.array(
            [0.2, 0.7, 0.4, 1.1, 0.9, 1.8, 1.5, 2.4, 2.1, 3.0, 2.8, 3.5]
        )
        scipy_functions = {
            "kendall": kendalltau,
            "spearman": spearmanr,
            "pearson": pearsonr,
        }

        for method, scipy_function in scipy_functions.items():
            for alternative in ("two-sided", "greater", "less"):
                with self.subTest(method=method, alternative=alternative):
                    actual = assess_trend(
                        indicator,
                        times=times,
                        method=method,
                        alternative=alternative,
                    )
                    expected = scipy_function(
                        times, indicator, alternative=alternative
                    )
                    self.assertAlmostEqual(actual.statistic, expected.statistic)
                    self.assertAlmostEqual(actual.p_value, expected.pvalue)
                    self.assertEqual(actual.details, {})

    def test_linear_assessment_uses_supplied_time_units(self):
        times = np.arange(12, dtype=float) * 0.5
        indicator = 1.25 + 2.0 * times

        result = assess_trend(indicator, times=times, method="linear")

        self.assertAlmostEqual(result.statistic, 2.0)
        self.assertAlmostEqual(result.details["intercept"], 1.25)
        self.assertAlmostEqual(result.details["r_squared"], 1.0)

    def test_linear_assessment_matches_scipy(self):
        times = np.arange(10, dtype=float)
        indicator = np.array(
            [2.0, 1.0, 2.5, 3.0, 2.7, 4.8, 5.1, 4.9, 6.2, 7.0]
        )

        actual = assess_trend(
            indicator, times=times, method="linear", alternative="less"
        )
        expected = linregress(times, indicator, alternative="less")

        self.assertAlmostEqual(actual.statistic, expected.slope)
        self.assertAlmostEqual(actual.p_value, expected.pvalue)
        self.assertAlmostEqual(actual.details["intercept"], expected.intercept)
        self.assertAlmostEqual(
            actual.details["slope_standard_error"], expected.stderr
        )

    def test_theil_sen_matches_scipy_and_has_no_analytical_p_value(self):
        times = np.arange(10, dtype=float) * 2.0
        indicator = np.array(
            [1.0, 2.2, 2.8, 4.1, 20.0, 5.8, 7.2, 7.9, 9.1, 10.0]
        )

        actual = assess_trend(indicator, times=times, method="theil_sen")
        expected = theilslopes(indicator, times, alpha=0.95)

        self.assertAlmostEqual(actual.statistic, expected.slope)
        self.assertIsNone(actual.p_value)
        self.assertAlmostEqual(actual.details["intercept"], expected.intercept)
        self.assertAlmostEqual(
            actual.details["slope_ci_lower"], expected.low_slope
        )
        self.assertAlmostEqual(
            actual.details["slope_ci_upper"], expected.high_slope
        )
        self.assertEqual(actual.details["confidence_level"], 0.95)

    def test_surrogate_p_value_supports_all_alternatives(self):
        surrogates = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        self.assertAlmostEqual(
            surrogate_p_value(1.0, surrogates, "greater"), 2 / 5
        )
        self.assertAlmostEqual(
            surrogate_p_value(-1.0, surrogates, "less"), 2 / 5
        )
        self.assertAlmostEqual(
            surrogate_p_value(1.0, surrogates, "two-sided"), 4 / 5
        )

    def test_pipeline_supports_every_trend_method(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(160)

        for method in AVAILABLE_TREND_METHODS:
            with self.subTest(method=method):
                result = analyze_timeseries(
                    x,
                    method="variance",
                    window_size=40,
                    step=5,
                    detrend=None,
                    trend_method=method,
                )
                self.assertEqual(result.trend_method, method)
                self.assertTrue(np.isfinite(result.trend_statistic))
                if method == "theil_sen":
                    self.assertIsNone(result.trend_p_value)
                    self.assertIn("slope_ci_lower", result.trend_details)
                else:
                    self.assertTrue(np.isfinite(result.trend_p_value))

    def test_slope_pipeline_reports_per_physical_time(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(180)

        for method in ("linear", "theil_sen"):
            with self.subTest(method=method):
                result = analyze_timeseries(
                    x,
                    method="variance",
                    window_size=45,
                    step=6,
                    dt=0.25,
                    detrend=None,
                    trend_method=method,
                )
                expected = assess_trend(
                    result.indicator,
                    times=result.times * 0.25,
                    method=method,
                )
                self.assertAlmostEqual(result.trend_statistic, expected.statistic)
                self.assertAlmostEqual(
                    result.trend_details["intercept"],
                    expected.details["intercept"],
                )

    def test_surrogate_pipeline_supports_every_trend_method(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal(140)

        for method in AVAILABLE_TREND_METHODS:
            with self.subTest(method=method):
                result = analyze_timeseries(
                    x,
                    method="ac1",
                    window_size=35,
                    step=5,
                    detrend=None,
                    trend_method=method,
                    significance=True,
                    n_surrogates=5,
                    surrogate_seed=123,
                )
                self.assertEqual(result.surrogate_statistics.shape, (5,))
                self.assertTrue(np.all(np.isfinite(result.surrogate_statistics)))
                self.assertTrue(np.isfinite(result.surrogate_p_value))

    def test_surrogate_seed_is_reproducible_for_new_methods(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(120)
        kwargs = dict(
            method="variance",
            window_size=30,
            step=5,
            detrend=None,
            trend_method="pearson",
            significance=True,
            n_surrogates=5,
            surrogate_seed=17,
        )

        first = analyze_timeseries(x, **kwargs)
        second = analyze_timeseries(x, **kwargs)

        np.testing.assert_allclose(
            first.surrogate_statistics, second.surrogate_statistics
        )
        self.assertEqual(first.surrogate_p_value, second.surrogate_p_value)

    def test_theil_sen_results_plot_without_an_analytical_p_value(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(120)
        result = analyze_timeseries(
            x,
            method="variance",
            window_size=30,
            step=5,
            detrend=None,
            trend_method="theil_sen",
            significance=True,
            n_surrogates=5,
            surrogate_seed=3,
        )

        fig_analysis, axes = plot_analysis(x, result)
        fig_significance, _ = plot_significance(result)
        annotation = "\n".join(text.get_text() for text in axes[1].texts)
        self.assertIn("Theil–Sen slope", annotation)
        self.assertIn("95% slope CI", annotation)
        plt.close(fig_analysis)
        plt.close(fig_significance)

    def test_plot_can_show_an_additional_linear_assessment(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal(120)
        result = analyze_timeseries(
            x,
            method="variance",
            window_size=30,
            step=5,
            detrend=None,
            trend_method="kendall",
        )

        fig, axes = plot_analysis(
            x,
            result,
            additional_trend_methods=("linear",),
        )
        annotation = "\n".join(text.get_text() for text in axes[1].texts)
        self.assertIn("Kendall τ", annotation)
        self.assertIn("OLS slope", annotation)
        self.assertIn("p (OLS)", annotation)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
