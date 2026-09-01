"""
Statistical significance testing for CSD indicator trends.

A statistically significant CSD signal requires that the observed trend in
an indicator is unlikely to have arisen from a stationary process.  Trends
can be measured by Kendall τ or by an ordinary least-squares slope.  This
module provides surrogate-based tests:

1. Generate N Fourier phase surrogates of the observed time series.
2. Compute the same CSD indicator on each surrogate.
3. Compute the selected trend statistic for each surrogate indicator.
4. Compare the observed statistic with its surrogate null distribution.

References
----------
Ben-Yami, M., Skiba, V., Bathiany, S., & Boers, N. (2023).
Uncertainties in critical slowing down indicators of observation-based
fingerprints of the Atlantic Overturning Circulation.
Nature Communications, 14, 8344.
https://doi.org/10.1038/s41467-023-44046-9
"""

import numpy as np
from typing import Optional, Tuple

from .surrogates import fourier_surrogates
from .sliding_window import sliding_window_analysis
from .trend import AVAILABLE_ALTERNATIVES, assess_trend


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
        equal to the observed statistic.  Appropriate when CSD predicts an
        increasing trend.  ``'less'`` tests for a decreasing trend and
        ``'two-sided'`` compares absolute magnitudes.

    Returns
    -------
    float
        p-value in [0, 1].  A small value (e.g., < 0.05) indicates that
        the observed trend is unlikely under the null hypothesis.
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
    elif alternative == "less":
        return float(np.mean(surr <= observed_statistic))
    elif alternative == "two-sided":
        return float(np.mean(np.abs(surr) >= np.abs(observed_statistic)))
    raise RuntimeError("unreachable")


def significance_test(
    x: np.ndarray,
    method: str = "lambda_glsar",
    window_size: int = 100,
    step: int = 1,
    dt: float = 1.0,
    local_detrend: bool = True,
    n_surrogates: int = 1000,
    seed: Optional[int] = None,
    trend_method: str = "kendall",
    trend_alternative: str = "two-sided",
    surrogate_alternative: str = "greater",
    **method_kwargs,
) -> Tuple[float, float, np.ndarray]:
    """
    Test the statistical significance of a CSD indicator trend.

    Generates ``n_surrogates`` Fourier phase surrogates, computes the same
    sliding-window CSD indicator on each, and builds a null distribution of
    the selected trend statistic.  The observed statistic is then compared
    with this distribution.

    Parameters
    ----------
    x : np.ndarray
        Input time series (after global preprocessing, if any).
    method : str
        CSD indicator method (see :data:`~csdtools.indicators.AVAILABLE_METHODS`).
    window_size : int
        Sliding window size.
    step : int
        Window step size.
    dt : float
        Physical time step.
    local_detrend : bool
        Remove linear trend within each window.
    n_surrogates : int
        Number of surrogates.  More surrogates give a more precise p-value
        but increase computation time linearly.  1000 is standard.
    seed : int or None
        Random seed for reproducible surrogate generation.
    trend_method : {'kendall', 'linear'}
        Statistic used to quantify the indicator trend.  ``'linear'`` uses
        the OLS slope against physical window time (midpoint index × ``dt``).
    trend_alternative : {'two-sided', 'greater', 'less'}
        Alternative hypothesis for the analytical trend p-value.  The
        analytical p-value itself is not returned by this function, but the
        same setting is used consistently when assessing each trend.
    surrogate_alternative : {'greater', 'less', 'two-sided'}
        Tail of the surrogate null distribution used for the p-value.
    **method_kwargs
        Additional arguments forwarded to the indicator function.

    Returns
    -------
    observed_statistic : float
        Kendall τ or OLS slope of the observed indicator trend.
    p_value : float
        Fraction of surrogate statistics at least as extreme as observed.
    surrogate_statistics : np.ndarray, shape (n_surrogates,)
        Selected trend statistic for each surrogate (null distribution).

    Examples
    --------
    >>> statistic, pval, surr_statistics = significance_test(
    ...     x_processed, method='ac1', window_size=100, n_surrogates=1000
    ... )
    >>> print(f"statistic = {statistic:.3f},  p = {pval:.3f}")
    """
    x = np.asarray(x, dtype=float)
    if surrogate_alternative not in AVAILABLE_ALTERNATIVES:
        raise ValueError(
            f"Unknown surrogate alternative '{surrogate_alternative}'. "
            f"Choose from: {list(AVAILABLE_ALTERNATIVES)}"
        )

    # Observed indicator and selected trend statistic.  Window midpoints are
    # converted to physical time so OLS slopes are not mechanically rescaled
    # by ``step`` and are expressed per unit of ``dt``.
    ind, times = sliding_window_analysis(
        x, method, window_size, step, dt, local_detrend, **method_kwargs
    )
    trend_times = times * dt
    observed = assess_trend(
        ind,
        times=trend_times,
        method=trend_method,
        alternative=trend_alternative,
    )

    # Surrogate null distribution
    surrogates = fourier_surrogates(x, n_surrogates, seed=seed)
    surrogate_statistics = np.empty(n_surrogates)
    for k in range(n_surrogates):
        s_ind, s_times = sliding_window_analysis(
            surrogates[k], method, window_size, step, dt, local_detrend,
            **method_kwargs
        )
        surrogate_statistics[k] = assess_trend(
            s_ind,
            times=s_times * dt,
            method=trend_method,
            alternative=trend_alternative,
        ).statistic

    p_val = surrogate_p_value(
        observed.statistic, surrogate_statistics, surrogate_alternative
    )
    return observed.statistic, p_val, surrogate_statistics
