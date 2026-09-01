import unittest

import numpy as np

from csdtools import analyze_timeseries
from csdtools.significance import surrogate_p_value
from csdtools.trend import assess_trend


class TrendAssessmentTests(unittest.TestCase):
    def test_linear_assessment_uses_supplied_time_units(self):
        times = np.arange(12, dtype=float) * 0.5
        indicator = 1.25 + 2.0 * times

        result = assess_trend(indicator, times=times, method="linear")

        self.assertAlmostEqual(result.statistic, 2.0)
        self.assertAlmostEqual(result.intercept, 1.25)
        self.assertAlmostEqual(result.r_squared, 1.0)

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

    def test_default_pipeline_uses_kendall(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(160)

        result = analyze_timeseries(
            x, method="variance", window_size=40, step=5, detrend=None
        )

        self.assertEqual(result.trend_method, "kendall")
        self.assertTrue(np.isfinite(result.trend_statistic))
        self.assertTrue(np.isfinite(result.trend_p_value))
        self.assertIsNone(result.linear_slope)

    def test_linear_pipeline_reports_slope_per_physical_time(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(180)

        result = analyze_timeseries(
            x,
            method="variance",
            window_size=45,
            step=6,
            dt=0.25,
            detrend=None,
            trend_method="linear",
        )
        expected = assess_trend(
            result.indicator,
            times=result.times * 0.25,
            method="linear",
        )

        self.assertAlmostEqual(result.trend_statistic, expected.statistic)
        self.assertAlmostEqual(result.trend_intercept, expected.intercept)
        self.assertIsNone(result.surrogate_statistics)

    def test_linear_surrogate_pipeline_is_reproducible(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal(140)
        kwargs = dict(
            method="ac1",
            window_size=35,
            step=5,
            detrend=None,
            trend_method="linear",
            significance=True,
            n_surrogates=8,
            surrogate_seed=123,
        )

        first = analyze_timeseries(x, **kwargs)
        second = analyze_timeseries(x, **kwargs)

        np.testing.assert_allclose(
            first.surrogate_statistics, second.surrogate_statistics
        )
        self.assertEqual(first.surrogate_p_value, second.surrogate_p_value)
        self.assertEqual(first.surrogate_statistics.shape, (8,))


if __name__ == "__main__":
    unittest.main()
