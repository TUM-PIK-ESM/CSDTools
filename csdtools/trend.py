"""
Trend assessment for CSD indicator time series.

The sliding-window pipeline produces one indicator value per window. This
module turns that indicator series into a scalar trend statistic that can be
used both for the observed data and for every member of a surrogate null
distribution.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np


AVAILABLE_ALTERNATIVES = ("two-sided", "greater", "less")


@dataclasses.dataclass(frozen=True)
class TrendAssessment:
    """Result of assessing an indicator trend against time."""

    method: str
    statistic: float
    p_value: Optional[float]
    alternative: str
    details: Mapping[str, float] = dataclasses.field(default_factory=dict)


TrendAssessor = Callable[[np.ndarray, np.ndarray, str], TrendAssessment]
TrendDetailsFormatter = Callable[[Mapping[str, float]], Tuple[str, ...]]


@dataclasses.dataclass(frozen=True)
class TrendMethodSpec:
    """Metadata and implementation for one registered trend method."""

    statistic_label: str
    analytical_test_label: Optional[str]
    statistic_format: str
    fit_label: str
    statistic_is_slope: bool
    detail_keys: Tuple[str, ...]
    details_formatter: TrendDetailsFormatter
    assessor: TrendAssessor


def _assessment(
    method: str,
    statistic: float,
    p_value: Optional[float],
    alternative: str,
    **details: float,
) -> TrendAssessment:
    """Construct a trend result while normalising NumPy scalar types."""
    return TrendAssessment(
        method=method,
        statistic=float(statistic),
        p_value=None if p_value is None else float(p_value),
        alternative=alternative,
        details={key: float(value) for key, value in details.items()},
    )


def _assess_kendall(
    t: np.ndarray, y: np.ndarray, alternative: str
) -> TrendAssessment:
    from scipy.stats import kendalltau

    statistic, p_value = kendalltau(t, y, alternative=alternative)
    return _assessment("kendall", statistic, p_value, alternative)


def _assess_spearman(
    t: np.ndarray, y: np.ndarray, alternative: str
) -> TrendAssessment:
    from scipy.stats import spearmanr

    statistic, p_value = spearmanr(t, y, alternative=alternative)
    return _assessment("spearman", statistic, p_value, alternative)


def _one_sided_p_value(
    statistic: float, two_sided_p: float, alternative: str
) -> float:
    """Convert a symmetric two-sided correlation p-value to one-sided."""
    if alternative == "two-sided":
        return float(two_sided_p)
    same_direction = (
        (alternative == "greater" and statistic >= 0)
        or (alternative == "less" and statistic <= 0)
    )
    if same_direction:
        return float(two_sided_p / 2.0)
    return float(1.0 - two_sided_p / 2.0)


def _assess_pearson(
    t: np.ndarray, y: np.ndarray, alternative: str
) -> TrendAssessment:
    from scipy.stats import pearsonr

    # ``alternative=`` was added to pearsonr in SciPy 1.9, while CSDTools
    # supports SciPy 1.7. The null distribution is symmetric, so deriving
    # the one-sided result from the two-sided p-value preserves that floor.
    statistic, two_sided_p = pearsonr(t, y)
    p_value = _one_sided_p_value(statistic, two_sided_p, alternative)
    return _assessment("pearson", statistic, p_value, alternative)


def _assess_linear(
    t: np.ndarray, y: np.ndarray, alternative: str
) -> TrendAssessment:
    from scipy.stats import linregress

    fit = linregress(t, y, alternative=alternative)
    r_squared = float(fit.rvalue**2) if np.isfinite(fit.rvalue) else np.nan
    return _assessment(
        "linear",
        fit.slope,
        fit.pvalue,
        alternative,
        intercept=fit.intercept,
        slope_standard_error=fit.stderr,
        r_squared=r_squared,
    )


def _assess_theil_sen(
    t: np.ndarray, y: np.ndarray, alternative: str
) -> TrendAssessment:
    from scipy.stats import theilslopes

    # Calling without ``method=`` retains compatibility with SciPy 1.7 and
    # uses its established separate-medians intercept definition.
    fit = theilslopes(y, t, alpha=0.95)
    return _assessment(
        "theil_sen",
        fit.slope,
        None,
        alternative,
        intercept=fit.intercept,
        slope_ci_lower=fit.low_slope,
        slope_ci_upper=fit.high_slope,
        confidence_level=0.95,
    )


def _format_no_details(details: Mapping[str, float]) -> Tuple[str, ...]:
    return ()


def _format_linear_details(details: Mapping[str, float]) -> Tuple[str, ...]:
    r_squared = details.get("r_squared")
    if r_squared is None or not np.isfinite(r_squared):
        return ()
    return (f"R² = {r_squared:.3f}",)


def _format_theil_sen_details(
    details: Mapping[str, float],
) -> Tuple[str, ...]:
    low = details.get("slope_ci_lower")
    high = details.get("slope_ci_upper")
    if (
        low is None
        or high is None
        or not np.isfinite(low)
        or not np.isfinite(high)
    ):
        return ()
    return (f"95% slope CI = [{low:+.3e}, {high:+.3e}]",)


# Adding a trend method requires one assessor and one registry entry. The
# metadata drives validation, result diagnostics, plot labels, formatting,
# and fitted-line behaviour throughout the package.
TREND_METHODS: Dict[str, TrendMethodSpec] = {
    "kendall": TrendMethodSpec(
        statistic_label="Kendall τ",
        analytical_test_label="Kendall",
        statistic_format="+.3f",
        fit_label="OLS guide",
        statistic_is_slope=False,
        detail_keys=(),
        details_formatter=_format_no_details,
        assessor=_assess_kendall,
    ),
    "spearman": TrendMethodSpec(
        statistic_label="Spearman ρ",
        analytical_test_label="Spearman",
        statistic_format="+.3f",
        fit_label="OLS guide",
        statistic_is_slope=False,
        detail_keys=(),
        details_formatter=_format_no_details,
        assessor=_assess_spearman,
    ),
    "pearson": TrendMethodSpec(
        statistic_label="Pearson r",
        analytical_test_label="Pearson",
        statistic_format="+.3f",
        fit_label="OLS fit",
        statistic_is_slope=False,
        detail_keys=(),
        details_formatter=_format_no_details,
        assessor=_assess_pearson,
    ),
    "linear": TrendMethodSpec(
        statistic_label="OLS slope",
        analytical_test_label="OLS",
        statistic_format="+.3e",
        fit_label="OLS fit",
        statistic_is_slope=True,
        detail_keys=("intercept", "slope_standard_error", "r_squared"),
        details_formatter=_format_linear_details,
        assessor=_assess_linear,
    ),
    "theil_sen": TrendMethodSpec(
        statistic_label="Theil–Sen slope",
        analytical_test_label=None,
        statistic_format="+.3e",
        fit_label="Theil–Sen fit",
        statistic_is_slope=True,
        detail_keys=(
            "intercept",
            "slope_ci_lower",
            "slope_ci_upper",
            "confidence_level",
        ),
        details_formatter=_format_theil_sen_details,
        assessor=_assess_theil_sen,
    ),
}

AVAILABLE_TREND_METHODS = tuple(TREND_METHODS)


def _method_spec(method: str) -> TrendMethodSpec:
    """Return a registered method specification or raise a useful error."""
    try:
        return TREND_METHODS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown trend method '{method}'. "
            f"Choose from: {list(AVAILABLE_TREND_METHODS)}"
        ) from exc


def trend_statistic_label(method: str) -> str:
    """Return the display label for a trend method's statistic."""
    return _method_spec(method).statistic_label


def trend_analytical_test_label(method: str) -> Optional[str]:
    """Return the analytical test label, or ``None`` when unavailable."""
    return _method_spec(method).analytical_test_label


def format_trend_statistic(method: str, value: float) -> str:
    """Format a statistic using its registered display precision."""
    return format(value, _method_spec(method).statistic_format)


def format_trend_details(
    method: str, details: Mapping[str, float]
) -> Tuple[str, ...]:
    """Format registered method-specific diagnostics for plot annotations."""
    return _method_spec(method).details_formatter(details)


def fitted_trend_line(
    indicator: np.ndarray,
    times: np.ndarray,
    method: str,
    statistic: float,
    details: Mapping[str, float],
) -> Tuple[np.ndarray, str]:
    """Return fitted indicator values and their plot label."""
    spec = _method_spec(method)
    y = np.asarray(indicator, dtype=float)
    t = np.asarray(times, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(y, np.nan), spec.fit_label

    intercept = details.get("intercept")
    if (
        spec.statistic_is_slope
        and intercept is not None
        and np.isfinite(intercept)
        and np.isfinite(statistic)
    ):
        return intercept + statistic * t, spec.fit_label

    coefficients = np.polyfit(t[mask], y[mask], 1)
    return np.polyval(coefficients, t), spec.fit_label


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
        Time coordinate for the indicator values. If omitted, consecutive
        integer indices are used. Slope statistics are expressed per unit of
        this coordinate.
    method : str
        Registered trend method. See :data:`AVAILABLE_TREND_METHODS`.
    alternative : {'two-sided', 'greater', 'less'}
        Alternative hypothesis for the analytical p-value, when the selected
        method supplies one.

    Returns
    -------
    TrendAssessment
        The selected statistic, optional analytical p-value, and any
        method-specific values in ``details``. Theil–Sen supplies a 95%
        slope confidence interval but no analytical p-value.

    Notes
    -----
    The analytical p-values assume independent observations. Sliding-window
    indicator values are generally serially dependent, particularly when
    windows overlap. The surrogate p-value from ``significance_test`` is
    usually the more appropriate pipeline-level significance assessment.
    """
    spec = _method_spec(method)
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
    if mask.sum() < 3 or np.unique(t[mask]).size < 2:
        p_value = np.nan if spec.analytical_test_label is not None else None
        return _assessment(
            method,
            np.nan,
            p_value,
            alternative,
            **{key: np.nan for key in spec.detail_keys},
        )

    return spec.assessor(t[mask], y[mask], alternative)
