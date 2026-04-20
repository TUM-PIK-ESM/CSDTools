"""
Statistical significance testing for CSD indicator trends.

A statistically significant CSD signal requires that the observed trend in
an indicator (measured by Kendall τ) is unlikely to have arisen from a
stationary process.  This module provides surrogate-based tests:

1. Generate N Fourier phase surrogates of the observed time series.
2. Compute the same CSD indicator on each surrogate.
3. Compute Kendall τ for each surrogate indicator trend.
4. The p-value is the fraction of surrogates with τ ≥ τ_observed.

References
----------
Dakos et al. (2008). Slowing down as an early warning signal for abrupt
climate change. *PNAS*.

Ben-Yami et al. (2023). Uncertainties in AMOC early warning signals.
"""

import numpy as np
from typing import Tuple

from .surrogates import fourier_surrogates
from .sliding_window import sliding_window_analysis
from ._utils import kendall_tau


def surrogate_p_value(
    observed_tau: float,
    surrogate_taus: np.ndarray,
    alternative: str = "greater",
) -> float:
    """
    Compute a surrogate-based p-value.

    Parameters
    ----------
    observed_tau : float
        Kendall τ of the observed CSD indicator trend.
    surrogate_taus : np.ndarray
        Kendall τ values for each surrogate (null distribution).
    alternative : str
        ``'greater'``  — one-sided test: fraction of surrogates with τ ≥
        τ_observed.  Appropriate when CSD predicts an *increasing* trend.
        ``'two-sided'`` — two-sided test: fraction with |τ| ≥ |τ_observed|.

    Returns
    -------
    float
        p-value in [0, 1].  A small value (e.g., < 0.05) indicates that
        the observed trend is unlikely under the null hypothesis.
    """
    surr = surrogate_taus[np.isfinite(surrogate_taus)]
    if len(surr) == 0:
        return np.nan
    if alternative == "greater":
        return float(np.mean(surr >= observed_tau))
    elif alternative == "two-sided":
        return float(np.mean(np.abs(surr) >= np.abs(observed_tau)))
    else:
        raise ValueError("alternative must be 'greater' or 'two-sided'")


def significance_test(
    x: np.ndarray,
    method: str = "lambda_glsar",
    window_size: int = 100,
    step: int = 1,
    dt: float = 1.0,
    local_detrend: bool = True,
    n_surrogates: int = 1000,
    alternative: str = "greater",
    seed: int = None,
    **method_kwargs,
) -> Tuple[float, float, np.ndarray]:
    """
    Test the statistical significance of a CSD indicator trend.

    Generates ``n_surrogates`` Fourier phase surrogates, computes the same
    sliding-window CSD indicator on each, and builds a null distribution of
    Kendall τ values.  The observed τ is then compared to this distribution.

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
    alternative : str
        ``'greater'`` (one-sided, recommended for CSD) or ``'two-sided'``.
    seed : int or None
        Random seed for reproducible surrogate generation.
    **method_kwargs
        Additional arguments forwarded to the indicator function.

    Returns
    -------
    observed_tau : float
        Kendall τ of the observed indicator trend.
    p_value : float
        Fraction of surrogates with τ ≥ observed_tau (for 'greater').
    surrogate_taus : np.ndarray, shape (n_surrogates,)
        Kendall τ for each surrogate (null distribution).

    Examples
    --------
    >>> tau, pval, surr_taus = significance_test(
    ...     x_processed, method='ac1', window_size=100, n_surrogates=1000
    ... )
    >>> print(f"τ = {tau:.3f},  p = {pval:.3f}")
    """
    x = np.asarray(x, dtype=float)

    # Observed indicator and its Kendall τ
    ind, _ = sliding_window_analysis(
        x, method, window_size, step, dt, local_detrend, **method_kwargs
    )
    observed_tau, _ = kendall_tau(ind)

    # Surrogate null distribution
    surrogates = fourier_surrogates(x, n_surrogates, seed=seed)
    surrogate_taus = np.empty(n_surrogates)
    for k in range(n_surrogates):
        s_ind, _ = sliding_window_analysis(
            surrogates[k], method, window_size, step, dt, local_detrend,
            **method_kwargs
        )
        surrogate_taus[k], _ = kendall_tau(s_ind)

    p_val = surrogate_p_value(observed_tau, surrogate_taus, alternative)
    return observed_tau, p_val, surrogate_taus
