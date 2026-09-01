"""
csdtools — Critical Slowing Down Analysis Toolkit
================================================

A unified toolkit for detecting loss of resilience in dynamical systems
via Critical Slowing Down (CSD) theory.

Quick start
-----------
>>> import numpy as np
>>> from csdtools import analyze_timeseries
>>>
>>> rng = np.random.default_rng(42)
>>> x = rng.standard_normal(500)
>>> result = analyze_timeseries(x, method='lambda_glsar', window_size=100)
>>> print(f"Kendall τ = {result.trend_statistic:.3f}")

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
    Surrogate-based significance testing of Kendall or linear trends.

trend
    Kendall-rank and ordinary least-squares trend assessment.

spatial
    Spatial correlation analysis and distance matrix utilities.

empirical
    Empirical transition detection and recovery rate fitting.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Optional

import numpy as np

from .indicators import AVAILABLE_METHODS, ND_METHODS, compute_indicator
from .preprocessing import preprocess
from .sliding_window import sliding_window_analysis
from .significance import significance_test
from .trend import (
    AVAILABLE_ALTERNATIVES,
    AVAILABLE_TREND_METHODS,
    TrendAssessment,
    assess_trend,
)
from ._utils import strip_nans
from .plotting import plot_analysis, plot_multi_analysis, plot_significance

__version__ = "1.1.0"   # preliminary bump; subject to change before the 1.1 release
__author__ = "CSDTools contributors"

__all__ = [
    "analyze_timeseries",
    "CSDResult",
    "AVAILABLE_METHODS",
    "ND_METHODS",
    "AVAILABLE_TREND_METHODS",
    "TrendAssessment",
    "assess_trend",
    "preprocess",
    "compute_indicator",
    "significance_test",
    "sliding_window_analysis",
    "plot_analysis",
    "plot_multi_analysis",
    "plot_significance",
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
    surrogate_p_value : float or None
        P-value from Fourier surrogate testing using
        ``surrogate_alternative``.  ``None`` if ``significance=False`` was
        passed to :func:`analyze_timeseries`.
    method : str
        Name of the CSD indicator method used.
    window_size : int
        Sliding window size (samples).
    detrend_method : str or None
        Global preprocessing applied before windowing.
    trend_method : {'kendall', 'linear'}
        Selected trend assessment.
    trend_statistic : float
        Kendall τ or OLS slope, depending on ``trend_method``.
    trend_p_value : float
        Analytical p-value for the selected trend statistic.
    trend_alternative : {'two-sided', 'greater', 'less'}
        Alternative hypothesis used for the analytical p-value.
    surrogate_statistics : np.ndarray or None
        Null distribution of the selected trend statistic.
    surrogate_alternative : {'greater', 'less', 'two-sided'}
        Alternative used for the surrogate p-value.
    trend_intercept : float or None
        OLS intercept for ``trend_method='linear'``.
    trend_standard_error : float or None
        Standard error of the OLS slope for ``trend_method='linear'``.
    trend_r_squared : float or None
        OLS R² for ``trend_method='linear'``.
    dt : float
        Physical sampling interval used to scale linear-trend time.
    """

    indicator: np.ndarray
    times: np.ndarray
    surrogate_p_value: Optional[float]
    method: str
    window_size: int
    detrend_method: Optional[str]
    trend_method: str
    trend_statistic: float
    trend_p_value: float
    trend_alternative: str
    surrogate_statistics: Optional[np.ndarray]
    surrogate_alternative: str
    trend_intercept: Optional[float]
    trend_standard_error: Optional[float]
    trend_r_squared: Optional[float]
    dt: float

    @property
    def linear_slope(self) -> Optional[float]:
        """OLS slope when ``trend_method='linear'``, otherwise ``None``."""
        return self.trend_statistic if self.trend_method == "linear" else None


def analyze_timeseries(
    x,
    method: str = "lambda_glsar",
    window_size: int = 100,
    step: int = 1,
    detrend: Optional[str] = "running_mean",
    detrend_window: Optional[int] = None,
    local_detrend: bool = True,
    dt: float = 1.0,
    exclude_nan_neighbors: bool = False,
    significance: bool = False,
    n_surrogates: int = 1000,
    trend_method: str = "kendall",
    trend_alternative: str = "two-sided",
    surrogate_alternative: str = "greater",
    surrogate_seed: Optional[int] = None,
    **method_kwargs,
) -> CSDResult:
    """
    Full CSD analysis pipeline for a univariate or multivariate time series.

    This is the main user-facing function.  It:

    1. Globally preprocesses the time series (optional detrending).
    2. Computes the chosen CSD indicator in a sliding window.
    3. Measures the trend via Kendall τ or an OLS slope.
    4. Optionally assesses statistical significance via Fourier surrogates.

    Parameters
    ----------
    x : array-like, shape (n,) or (n_dims, n_samples)
        Input time series.  Pass a 1D array for univariate analysis or a 2D
        array of shape ``(n_dims, n_samples)`` for multivariate analysis.
        Multivariate input is only supported by the multivariate methods
        (``'lambda_km'``, ``'lambda_dmd'``, ``'lambda_dmd_floquet'``,
        ``'spatial_corr'``).  Global preprocessing is applied per dimension;
        surrogate significance testing is not yet supported for 2D input.
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
    exclude_nan_neighbors : bool
        Relevant when the input contains NaN values.  If ``False`` (default),
        only the NaN positions themselves are removed and the surrounding
        values are kept.  If ``True``, the values immediately adjacent to
        each NaN are *also* removed.

        **When to use**: set to ``True`` when ``method='ac1'`` and data gaps
        are meaningful.  Without this option, a value just before a gap and
        the value just after it become an apparent lag-1 pair after NaN
        removal, even though they were not actually consecutive.  Removing
        their neighbours prevents such pairs from contaminating the AC1
        estimate.  The option has no effect when the input contains no NaNs.
    significance : bool
        If ``True``, generate ``n_surrogates`` Fourier phase surrogates and
        compute a p-value using ``surrogate_alternative``.  Adds computation
        time ∝ n_surrogates.
    n_surrogates : int
        Number of surrogates for significance testing.  1000 is standard.
    trend_method : {'kendall', 'linear'}
        Trend statistic applied to the sliding-window indicator.  Kendall τ
        is the default.  ``'linear'`` uses the OLS slope against physical
        window time (midpoint index × ``dt``).
    trend_alternative : {'two-sided', 'greater', 'less'}
        Alternative hypothesis for the analytical trend p-value.  The default
        preserves the existing two-sided Kendall test behavior.
    surrogate_alternative : {'greater', 'less', 'two-sided'}
        Tail of the surrogate null distribution used for the surrogate
        p-value.  ``'greater'`` is the CSD-oriented default because the
        package's indicators are generally defined to increase as resilience
        declines.
    surrogate_seed : int or None
        Random seed for reproducible Fourier surrogates.
    **method_kwargs
        Extra keyword arguments forwarded to the indicator function.
        Examples:

        - ``max_lag=25``  for ``'lambda_acs'``
        - ``bin_number=80``  for ``'lambda_km'``
        - ``period=12``  for ``'lambda_dmd_floquet'``
        - ``noise=<series>`` or ``noise_psd=<callable/(ω, values)>``  for ``'rosa'``

        A ``noise`` series passed for ``'rosa'`` must be aligned with (and the
        same length as) ``x`` after NaN handling; it is windowed in sync with
        each analysis window, allowing a time-varying forcing spectrum.

        To pass kwargs to the STL/harmonic preprocessor, use the key
        ``preprocess_kwargs={'period': 12}``.

    Returns
    -------
    CSDResult
        Object containing the indicator timeseries, selected trend assessment,
        and optionally the surrogate p-value.

    Raises
    ------
    ValueError
        If ``method`` is not in :data:`AVAILABLE_METHODS` or if required
        parameters are missing.

    Examples
    --------
    Basic analysis with default settings:

    >>> result = analyze_timeseries(x, method='lambda_glsar', window_size=100)
    >>> print(result.trend_statistic, result.trend_p_value)

    With significance testing:

    >>> result = analyze_timeseries(
    ...     x, method='ac1', window_size=100,
    ...     significance=True, n_surrogates=1000,
    ... )
    >>> print(f"p = {result.surrogate_p_value:.3f}")

    Linear-regression trend assessment:

    >>> result = analyze_timeseries(
    ...     x, method='ac1', window_size=100, trend_method='linear'
    ... )
    >>> print(result.linear_slope, result.trend_p_value)

    Seasonal data (monthly, 12-month period) with STL detrending:

    >>> result = analyze_timeseries(
    ...     x_monthly, method='lambda_glsar', window_size=120,
    ...     detrend='stl', preprocess_kwargs={'period': 12},
    ... )

    Multiple methods (manual loop):

    >>> for m in ['ac1', 'variance', 'lambda_glsar']:
    ...     r = analyze_timeseries(x, method=m, window_size=100)
    ...     print(f"{m:15s}  τ={r.trend_statistic:+.3f}")

    Multivariate KM analysis (x_nd has shape (n_dims, n_samples)):

    >>> result = analyze_timeseries(
    ...     x_nd, method='lambda_km', window_size=300, dt=1.0
    ... )
    >>> print(result.trend_statistic)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim > 2:
        raise ValueError(
            f"Input must be 1D or 2D, got shape {x.shape}."
        )

    multivariate = x.ndim == 2

    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {AVAILABLE_METHODS}"
        )

    if trend_method not in AVAILABLE_TREND_METHODS:
        raise ValueError(
            f"Unknown trend method '{trend_method}'. "
            f"Choose from: {list(AVAILABLE_TREND_METHODS)}"
        )
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

    if multivariate and method not in ND_METHODS:
        raise ValueError(
            f"Multivariate input (shape {x.shape}) is only supported by "
            f"multivariate methods: {sorted(ND_METHODS)}."
        )

    # --- NaN handling ---------------------------------------------------
    if multivariate:
        # Drop time steps where any dimension is non-finite
        valid = np.all(np.isfinite(x), axis=0)
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            warnings.warn(
                f"Input contains {n_dropped} time step(s) with NaN/Inf in at "
                "least one dimension; those steps will be removed.",
                UserWarning,
                stacklevel=2,
            )
            x = x[:, valid]
    else:
        # strip_nans emits a UserWarning if NaNs are present; stacklevel=2
        # makes the warning point at the user's analyze_timeseries() call site.
        x = strip_nans(x, exclude_neighbors=exclude_nan_neighbors, stacklevel=2)

    # --- Global preprocessing -------------------------------------------
    preprocess_kwargs = method_kwargs.pop("preprocess_kwargs", {})
    detrend_kw: dict = {}

    if detrend == "running_mean":
        dw = detrend_window if detrend_window is not None else 2 * window_size
        detrend_kw["window"] = dw
    elif detrend in ("stl", "harmonic"):
        detrend_kw.update(preprocess_kwargs)

    if multivariate:
        if detrend is not None:
            x_proc = np.stack(
                [preprocess(x[i], method=detrend, **detrend_kw)
                 for i in range(x.shape[0])]
            )
        else:
            x_proc = x.copy()
    else:
        x_proc = preprocess(x, method=detrend, **detrend_kw)

    # --- Sliding-window indicator ----------------------------------------
    indicator, times = sliding_window_analysis(
        x_proc, method, window_size, step, dt, local_detrend, **method_kwargs
    )

    # --- Trend assessment -------------------------------------------------
    # Use physical window time so linear slopes are expressed per unit of dt
    # rather than being mechanically rescaled by the selected window step.
    trend = assess_trend(
        indicator,
        times=times * dt,
        method=trend_method,
        alternative=trend_alternative,
    )

    # --- Significance (optional) ----------------------------------------
    surr_p: Optional[float] = None
    surr_statistics: Optional[np.ndarray] = None
    if significance:
        if multivariate:
            warnings.warn(
                "Significance testing via Fourier surrogates is not supported "
                "for multivariate input (nD surrogate generation would need to "
                "preserve the cross-spectral structure). "
                "surrogate_p_value will be None.",
                UserWarning,
                stacklevel=2,
            )
        else:
            _, surr_p, surr_statistics = significance_test(
                x_proc,
                method,
                window_size,
                step,
                dt,
                local_detrend,
                n_surrogates=n_surrogates,
                seed=surrogate_seed,
                trend_method=trend_method,
                trend_alternative=trend_alternative,
                surrogate_alternative=surrogate_alternative,
                **method_kwargs,
            )

    return CSDResult(
        indicator=indicator,
        times=times,
        surrogate_p_value=surr_p,
        method=method,
        window_size=window_size,
        detrend_method=detrend,
        trend_method=trend_method,
        trend_statistic=float(trend.statistic),
        trend_p_value=float(trend.p_value),
        trend_alternative=trend_alternative,
        surrogate_statistics=surr_statistics,
        surrogate_alternative=surrogate_alternative,
        trend_intercept=trend.intercept,
        trend_standard_error=trend.standard_error,
        trend_r_squared=trend.r_squared,
        dt=float(dt),
    )
