"""
Statistical significance testing for CSD indicator trends.

A statistically significant CSD signal requires that the observed trend in
an indicator is unlikely to have arisen from a stationary process. Trends can
be measured with one or more registered correlation- or slope-based methods.
This module provides surrogate-based tests:

1. Generate N Fourier phase surrogates of the observed time series.
2. Compute the same CSD indicator on each surrogate.
3. Compute every selected trend statistic for each surrogate indicator.
4. Compare each observed statistic with its surrogate null distribution.

References
----------
Ben-Yami, M., Skiba, V., Bathiany, S., & Boers, N. (2023).
Uncertainties in critical slowing down indicators of observation-based
fingerprints of the Atlantic Overturning Circulation.
Nature Communications, 14, 8344.
https://doi.org/10.1038/s41467-023-44046-9
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np

from .sliding_window import sliding_window_analysis
from .surrogates import fourier_surrogates
from .trend import (
    AVAILABLE_ALTERNATIVES,
    TrendResult,
    assess_trend,
    normalize_trend_methods,
)


def surrogate_p_value(
    observed_statistic: float,
    surrogate_statistics: np.ndarray,
    alternative: str = "greater",
) -> float:
    """
    Compute a surrogate-based p-value.

    Parameters
    ----------
    observed_statistic : float
        Selected trend statistic for the observed CSD indicator.
    surrogate_statistics : np.ndarray
        Selected trend statistic for each surrogate (null distribution).
    alternative : str
        ``'greater'`` — fraction of surrogate statistics greater than or
        equal to the observed statistic. Appropriate when CSD predicts an
        increasing trend. ``'less'`` tests for a decreasing trend and
        ``'two-sided'`` compares absolute magnitudes.

    Returns
    -------
    float
        P-value in [0, 1]. A small value indicates that the observed trend is
        unlikely under the null hypothesis.
    """
    if alternative not in AVAILABLE_ALTERNATIVES:
        raise ValueError(
            f"Unknown alternative '{alternative}'. "
            f"Choose from: {list(AVAILABLE_ALTERNATIVES)}"
        )

    surrogate_statistics = np.asarray(surrogate_statistics, dtype=float)
    surr = surrogate_statistics[np.isfinite(surrogate_statistics)]
    if len(surr) == 0 or not np.isfinite(observed_statistic):
        return np.nan
    if alternative == "greater":
        return float(np.mean(surr >= observed_statistic))
    if alternative == "less":
        return float(np.mean(surr <= observed_statistic))
    return float(np.mean(np.abs(surr) >= np.abs(observed_statistic)))


def _analytical_trend_results(
    indicator: np.ndarray,
    times: np.ndarray,
    trend_methods: Sequence[str],
    trend_alternative: str,
    surrogate_alternative: str,
) -> Dict[str, TrendResult]:
    """Assess multiple trends without a surrogate null distribution."""
    results: Dict[str, TrendResult] = {}
    for trend_method in trend_methods:
        assessment = assess_trend(
            indicator,
            times=times,
            method=trend_method,
            alternative=trend_alternative,
        )
        results[trend_method] = TrendResult(
            method=assessment.method,
            statistic=assessment.statistic,
            p_value=assessment.p_value,
            alternative=assessment.alternative,
            details=dict(assessment.details),
            surrogate_p_value=None,
            surrogate_statistics=None,
            surrogate_alternative=surrogate_alternative,
        )
    return results


def _significance_from_indicator(
    x: np.ndarray,
    indicator: np.ndarray,
    times: np.ndarray,
    method: str,
    window_size: int,
    step: int,
    dt: float,
    local_detrend: bool,
    n_surrogates: int,
    seed: Optional[int],
    trend_methods: Sequence[str],
    trend_alternative: str,
    surrogate_alternative: str,
    **method_kwargs,
) -> Dict[str, TrendResult]:
    """Build observed and surrogate results from a precomputed indicator."""
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be at least 1.")

    physical_times = np.asarray(times, dtype=float) * dt
    observed = _analytical_trend_results(
        indicator,
        physical_times,
        trend_methods,
        trend_alternative,
        surrogate_alternative,
    )

    surrogate_statistics = {
        trend_method: np.empty(n_surrogates, dtype=float)
        for trend_method in trend_methods
    }
    surrogates = fourier_surrogates(x, n_surrogates, seed=seed)

    # The indicator is the expensive part of the surrogate pipeline. Compute
    # it once per surrogate, then apply every requested scalar trend method.
    for k in range(n_surrogates):
        surrogate_indicator, surrogate_times = sliding_window_analysis(
            surrogates[k],
            method,
            window_size,
            step,
            dt,
            local_detrend,
            **method_kwargs,
        )
        surrogate_physical_times = surrogate_times * dt
        for trend_method in trend_methods:
            surrogate_statistics[trend_method][k] = assess_trend(
                surrogate_indicator,
                times=surrogate_physical_times,
                method=trend_method,
                alternative=trend_alternative,
            ).statistic

    results: Dict[str, TrendResult] = {}
    for trend_method in trend_methods:
        assessment = observed[trend_method]
        statistics = surrogate_statistics[trend_method]
        results[trend_method] = TrendResult(
            method=assessment.method,
            statistic=assessment.statistic,
            p_value=assessment.p_value,
            alternative=assessment.alternative,
            details=dict(assessment.details),
            surrogate_p_value=surrogate_p_value(
                assessment.statistic,
                statistics,
                surrogate_alternative,
            ),
            surrogate_statistics=statistics,
            surrogate_alternative=surrogate_alternative,
        )
    return results


def significance_test(
    x: np.ndarray,
    method: str = "lambda_glsar",
    window_size: int = 100,
    step: int = 1,
    dt: float = 1.0,
    local_detrend: bool = True,
    n_surrogates: int = 1000,
    seed: Optional[int] = None,
    trend_methods: Union[str, Sequence[str]] = ("kendall",),
    trend_alternative: str = "two-sided",
    surrogate_alternative: str = "greater",
    **method_kwargs,
) -> Dict[str, TrendResult]:
    """
    Test one or more CSD indicator trend statistics with shared surrogates.

    Each surrogate indicator is computed once and then assessed with every
    requested trend method.

    Parameters
    ----------
    x : np.ndarray
        Input time series after any desired global preprocessing.
    method : str
        CSD indicator method.
    window_size : int
        Sliding window size.
    step : int
        Window step size.
    dt : float
        Physical time step.
    local_detrend : bool
        Remove a linear trend within each window.
    n_surrogates : int
        Number of Fourier phase surrogates.
    seed : int or None
        Random seed for reproducible surrogate generation.
    trend_methods : str or sequence of str
        Registered trend methods to assess. See
        :data:`~csdtools.trend.AVAILABLE_TREND_METHODS`.
    trend_alternative : {'two-sided', 'greater', 'less'}
        Alternative used for analytical p-values when available.
    surrogate_alternative : {'greater', 'less', 'two-sided'}
        Tail used for every surrogate p-value.
    **method_kwargs
        Additional arguments forwarded to the indicator function.

    Returns
    -------
    dict of str to TrendResult
        One stored analytical and surrogate result per requested method.

    Examples
    --------
    >>> trends = significance_test(
    ...     x_processed,
    ...     method='ac1',
    ...     window_size=100,
    ...     n_surrogates=1000,
    ...     trend_methods=('kendall', 'linear'),
    ... )
    >>> print(trends['linear'].statistic, trends['linear'].surrogate_p_value)
    """
    methods = normalize_trend_methods(trend_methods)
    if trend_alternative not in AVAILABLE_ALTERNATIVES:
        raise ValueError(
            f"Unknown trend alternative '{trend_alternative}'. "
            f"Choose from: {list(AVAILABLE_ALTERNATIVES)}"
        )
    if surrogate_alternative not in AVAILABLE_ALTERNATIVES:
        raise ValueError(
            f"Unknown surrogate alternative '{surrogate_alternative}'. "
            f"Choose from: {list(AVAILABLE_ALTERNATIVES)}"
        )

    x = np.asarray(x, dtype=float)
    indicator, times = sliding_window_analysis(
        x,
        method,
        window_size,
        step,
        dt,
        local_detrend,
        **method_kwargs,
    )
    return _significance_from_indicator(
        x,
        indicator,
        times,
        method,
        window_size,
        step,
        dt,
        local_detrend,
        n_surrogates,
        seed,
        methods,
        trend_alternative,
        surrogate_alternative,
        **method_kwargs,
    )
