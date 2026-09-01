"""
Trend assessment for CSD indicator time series.

The sliding-window pipeline produces one indicator value per window.  This
module turns that indicator series into a scalar trend statistic that can be
used both for the observed data and for every member of a surrogate null
distribution.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


AVAILABLE_TREND_METHODS = ("kendall", "linear")
AVAILABLE_ALTERNATIVES = ("two-sided", "greater", "less")


@dataclasses.dataclass(frozen=True)
class TrendAssessment:
    """Result of assessing a monotonic or linear indicator trend."""

    method: str
    statistic: float
    p_value: float
    alternative: str
    intercept: Optional[float] = None
    standard_error: Optional[float] = None
    r_squared: Optional[float] = None


def trend_statistic_label(method: str) -> str:
    """Return the display label for a trend method's test statistic."""
    if method == "kendall":
        return "Kendall τ"
    if method == "linear":
        return "OLS slope"
    raise ValueError(
        f"Unknown trend method '{method}'. "
        f"Choose from: {list(AVAILABLE_TREND_METHODS)}"
    )


def assess_trend(
    indicator: np.ndarray,
    times: Optional[np.ndarray] = None,
    method: str = "kendall",
    alternative: str = "two-sided",
) -> TrendAssessment:
    """
    Assess the trend in a CSD indicator time series.

    Parameters
    ----------
    indicator : np.ndarray
        One-dimensional indicator time series.
    times : np.ndarray or None
        Time coordinate for the indicator values.  If omitted, consecutive
        integer indices are used.  Linear-regression slopes are expressed per
        unit of this coordinate.
    method : {'kendall', 'linear'}
        ``'kendall'`` computes Kendall's rank correlation against time.
        ``'linear'`` fits an ordinary least-squares line and uses its slope as
        the trend statistic.
    alternative : {'two-sided', 'greater', 'less'}
        Alternative hypothesis for the analytical p-value.

    Returns
    -------
    TrendAssessment
        The selected statistic and analytical p-value.  Linear regression
        additionally supplies the intercept, slope standard error, and R².

    Notes
    -----
    The analytical p-values assume independent observations.  Sliding-window
    indicator values are generally serially dependent, particularly when
    windows overlap.  The surrogate p-value from ``significance_test`` is
    usually the more appropriate pipeline-level significance assessment.
    """
    if method not in AVAILABLE_TREND_METHODS:
        raise ValueError(
            f"Unknown trend method '{method}'. "
            f"Choose from: {list(AVAILABLE_TREND_METHODS)}"
        )
    if alternative not in AVAILABLE_ALTERNATIVES:
        raise ValueError(
            f"Unknown alternative '{alternative}'. "
            f"Choose from: {list(AVAILABLE_ALTERNATIVES)}"
        )

    y = np.asarray(indicator, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"indicator must be 1D, got shape {y.shape}.")

    if times is None:
        t = np.arange(len(y), dtype=float)
    else:
        t = np.asarray(times, dtype=float)
        if t.ndim != 1:
            raise ValueError(f"times must be 1D, got shape {t.shape}.")
        if len(t) != len(y):
            raise ValueError(
                "times and indicator must have the same length; "
                f"got {len(t)} and {len(y)}."
            )

    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 3:
        return TrendAssessment(
            method=method,
            statistic=np.nan,
            p_value=np.nan,
            alternative=alternative,
            intercept=np.nan if method == "linear" else None,
            standard_error=np.nan if method == "linear" else None,
            r_squared=np.nan if method == "linear" else None,
        )

    if method == "kendall":
        from scipy.stats import kendalltau

        statistic, p_value = kendalltau(
            t[mask], y[mask], alternative=alternative
        )
        return TrendAssessment(
            method=method,
            statistic=float(statistic),
            p_value=float(p_value),
            alternative=alternative,
        )

    from scipy.stats import linregress

    fit = linregress(t[mask], y[mask], alternative=alternative)
    r_squared = float(fit.rvalue**2) if np.isfinite(fit.rvalue) else np.nan
    return TrendAssessment(
        method=method,
        statistic=float(fit.slope),
        p_value=float(fit.pvalue),
        alternative=alternative,
        intercept=float(fit.intercept),
        standard_error=float(fit.stderr),
        r_squared=r_squared,
    )
