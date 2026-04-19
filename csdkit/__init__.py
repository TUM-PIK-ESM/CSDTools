"""
csdkit — Critical Slowing Down Analysis Toolkit
================================================

A unified toolkit for detecting loss of resilience in dynamical systems
via Critical Slowing Down (CSD) theory.

Quick start
-----------
>>> import numpy as np
>>> from csdkit import analyze_timeseries
>>>
>>> rng = np.random.default_rng(42)
>>> x = rng.standard_normal(500)
>>> result = analyze_timeseries(x, method='lambda_glsar', window_size=100)
>>> print(f"Kendall τ = {result.kendall_tau:.3f}")

With significance testing
-------------------------
>>> result = analyze_timeseries(
...     x, method='ac1', window_size=100,
...     significance=True, n_surrogates=1000,
... )
>>> print(f"p = {result.surrogate_p_value:.3f}")

Available methods
-----------------
+------------------+---------------------------------------------------------+
| ``'ac1'``        | Lag-1 autocorrelation                                   |
| ``'variance'``   | Running variance                                        |
| ``'lambda_glsar'`` | Restoring rate via Generalised Least Squares AR(1)    |
| ``'lambda_ols'`` | Restoring rate via Ordinary Least Squares AR(1)         |
| ``'lambda_acs'`` | Restoring rate via autocorrelation structure fitting    |
| ``'lambda_psd'`` | Restoring rate via power spectrum fitting               |
| ``'lambda_km'``  | Restoring rate via Kramers-Moyal (non-parametric)       |
| ``'rosa'``       | Restoring rate via Ratio Of Spectra Approach            |
+------------------+---------------------------------------------------------+

Module overview
---------------
preprocessing
    Detrending/deseasonalisation before windowing (running mean, linear,
    quadratic, STL, harmonic regression).

indicators
    Scalar CSD estimators operating on a single window of data.

sliding_window
    Apply an indicator in a sliding window over the full time series.

surrogates
    Fourier phase surrogate generation.

significance
    Surrogate-based significance testing of indicator trends.

spatial
    Spatial correlation analysis and distance matrix utilities.

empirical
    Empirical transition detection and recovery rate fitting.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .indicators import AVAILABLE_METHODS, compute_indicator
from .preprocessing import preprocess
from .sliding_window import sliding_window_analysis
from .significance import significance_test
from ._utils import kendall_tau

__version__ = "1.0.0"
__author__ = "CSDTools contributors"

__all__ = [
    "analyze_timeseries",
    "CSDResult",
    "AVAILABLE_METHODS",
    "preprocess",
    "compute_indicator",
    "significance_test",
    "sliding_window_analysis",
]


@dataclasses.dataclass
class CSDResult:
    """
    Container for the result of a CSD time series analysis.

    Attributes
    ----------
    indicator : np.ndarray
        Indicator values at each window position.  Contains NaN for windows
        that could not be computed.
    times : np.ndarray
        Index of the midpoint of each window in the original time series.
    kendall_tau : float
        Kendall τ rank correlation of the indicator against its time index.
        Positive → increasing trend → loss of stability.
    kendall_p_value : float
        Two-sided p-value for the Kendall τ under H₀ of no monotonic trend.
    surrogate_p_value : float or None
        One-sided p-value from Fourier surrogate testing.  Fraction of
        surrogates whose indicator trend is at least as large as observed.
        ``None`` if ``significance=False`` was passed to
        :func:`analyze_timeseries`.
    method : str
        Name of the CSD indicator method used.
    window_size : int
        Sliding window size (samples).
    detrend_method : str or None
        Global preprocessing applied before windowing.
    """

    indicator: np.ndarray
    times: np.ndarray
    kendall_tau: float
    kendall_p_value: float
    surrogate_p_value: Optional[float]
    method: str
    window_size: int
    detrend_method: Optional[str]


def analyze_timeseries(
    x,
    method: str = "lambda_glsar",
    window_size: int = 100,
    step: int = 1,
    detrend: Optional[str] = "running_mean",
    detrend_window: Optional[int] = None,
    local_detrend: bool = True,
    dt: float = 1.0,
    significance: bool = False,
    n_surrogates: int = 1000,
    **method_kwargs,
) -> CSDResult:
    """
    Full CSD analysis pipeline for a 1D time series.

    This is the main user-facing function.  It:

    1. Globally preprocesses the time series (optional detrending).
    2. Computes the chosen CSD indicator in a sliding window.
    3. Measures the monotonic trend via Kendall τ.
    4. Optionally assesses statistical significance via Fourier surrogates.

    Parameters
    ----------
    x : array-like, shape (n,)
        Input time series.
    method : str
        CSD indicator to use.  Choose from :data:`AVAILABLE_METHODS`:

        ``'ac1'``
            Lag-1 autocorrelation (fast, intuitive).
        ``'variance'``
            Running variance.
        ``'lambda_glsar'``
            Restoring rate λ via GLSAR AR(1) fitting.  Recommended default.
        ``'lambda_ols'``
            Restoring rate λ via OLS AR(1) fitting.  Faster than GLSAR.
        ``'lambda_acs'``
            Restoring rate λ via autocorrelation structure fitting.
        ``'lambda_psd'``
            Restoring rate λ via power spectrum fitting.
        ``'lambda_km'``
            Restoring rate λ via Kramers-Moyal (needs ``kramersmoyal``).
        ``'rosa'``
            Restoring rate λ via Ratio Of Spectra Approach (Clarke 2023).

    window_size : int
        Number of samples per sliding window.  A common rule of thumb is
        to use roughly 1/3–1/2 of the total series length, but the optimal
        value depends on the noise level and expected timescale of CSD.
    step : int
        Number of samples to advance between windows.  ``step=1`` gives a
        smooth, densely sampled indicator.  Larger values speed up
        computation at the cost of temporal resolution.
    detrend : str or None
        Global preprocessing applied to the raw series *before* windowing:

        ``'running_mean'`` (default)
            Subtract centred running mean.  Requires ``detrend_window``.
            If not provided, defaults to ``2 × window_size``.
        ``'linear'``
            Remove global linear trend.
        ``'quadratic'``
            Remove global quadratic trend.
        ``'stl'``
            STL seasonal-trend decomposition.  Pass ``period=12`` etc.
            via ``**method_kwargs`` or the ``preprocess_kwargs`` key.
        ``'harmonic'``
            Harmonic regression deseasonalisation.
        ``None``
            No global preprocessing.

    detrend_window : int or None
        Window size for ``'running_mean'`` preprocessing.  Defaults to
        ``2 × window_size`` if not specified.
    local_detrend : bool
        Remove a linear trend *within each window* before computing the
        indicator (default True).  This is an additional local detrending
        step on top of the global one.
    dt : float
        Physical time step in the units desired for λ (e.g., years, months).
        Does not affect AC1 or variance.
    significance : bool
        If ``True``, generate ``n_surrogates`` Fourier phase surrogates and
        compute a one-sided p-value.  Adds computation time ∝ n_surrogates.
    n_surrogates : int
        Number of surrogates for significance testing.  1000 is standard.
    **method_kwargs
        Extra keyword arguments forwarded to the indicator function.
        Examples:

        - ``max_lag=25``  for ``'lambda_acs'``
        - ``bin_number=80``  for ``'lambda_km'``
        - ``eta=0.4``  for ``'rosa'``

        To pass kwargs to the STL/harmonic preprocessor, use the key
        ``preprocess_kwargs={'period': 12}``.

    Returns
    -------
    CSDResult
        Object containing the indicator timeseries, Kendall τ, and
        optionally the surrogate p-value.

    Raises
    ------
    ValueError
        If ``method`` is not in :data:`AVAILABLE_METHODS` or if required
        parameters are missing.

    Examples
    --------
    Basic analysis with default settings:

    >>> result = analyze_timeseries(x, method='lambda_glsar', window_size=100)
    >>> print(result.kendall_tau, result.kendall_p_value)

    With significance testing:

    >>> result = analyze_timeseries(
    ...     x, method='ac1', window_size=100,
    ...     significance=True, n_surrogates=1000,
    ... )
    >>> print(f"p = {result.surrogate_p_value:.3f}")

    Seasonal data (monthly, 12-month period) with STL detrending:

    >>> result = analyze_timeseries(
    ...     x_monthly, method='lambda_glsar', window_size=120,
    ...     detrend='stl', preprocess_kwargs={'period': 12},
    ... )

    Multiple methods (manual loop):

    >>> for m in ['ac1', 'variance', 'lambda_glsar']:
    ...     r = analyze_timeseries(x, method=m, window_size=100)
    ...     print(f"{m:15s}  τ={r.kendall_tau:+.3f}")
    """
    x = np.asarray(x, dtype=float)

    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {AVAILABLE_METHODS}"
        )

    # --- Global preprocessing -------------------------------------------
    preprocess_kwargs = method_kwargs.pop("preprocess_kwargs", {})
    detrend_kw: dict = {}

    if detrend == "running_mean":
        dw = detrend_window if detrend_window is not None else 2 * window_size
        detrend_kw["window"] = dw
    elif detrend in ("stl", "harmonic"):
        detrend_kw.update(preprocess_kwargs)

    x_proc = preprocess(x, method=detrend, **detrend_kw)

    # --- Sliding-window indicator ----------------------------------------
    indicator, times = sliding_window_analysis(
        x_proc, method, window_size, step, dt, local_detrend, **method_kwargs
    )

    # --- Kendall τ trend -------------------------------------------------
    tau, kp = kendall_tau(indicator)

    # --- Significance (optional) ----------------------------------------
    surr_p: Optional[float] = None
    if significance:
        _, surr_p, _ = significance_test(
            x_proc,
            method,
            window_size,
            step,
            dt,
            local_detrend,
            n_surrogates=n_surrogates,
            **method_kwargs,
        )

    return CSDResult(
        indicator=indicator,
        times=times,
        kendall_tau=float(tau),
        kendall_p_value=float(kp),
        surrogate_p_value=surr_p,
        method=method,
        window_size=window_size,
        detrend_method=detrend,
    )
