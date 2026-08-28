"""
Sliding-window CSD analysis.

The core operation in CSD analysis is to compute an indicator (e.g. AC1,
variance, λ) inside a moving window that steps through the time series.
The resulting indicator timeseries is then tested for a monotonic trend —
a trend toward higher AC1, higher variance, or less-negative λ is evidence
of Critical Slowing Down.
"""

import warnings
import numpy as np
from typing import Tuple

from .indicators import compute_indicator, AVAILABLE_METHODS, RECOMMENDED_MIN_WINDOW, _MIN_WINDOW_REASON

#: Names of ``method_kwargs`` that are *time-aligned auxiliary series* rather
#: than static configuration: they have the same length as the time axis of
#: ``x`` and must be sliced in sync with each window before being handed to the
#: estimator.  Currently only ``'noise'`` (the forcing series for ``rosa``).
_WINDOWED_KWARGS: set = {"noise"}


def sliding_window_analysis(
    x: np.ndarray,
    method: str = "lambda_glsar",
    window_size: int = 100,
    step: int = 1,
    dt: float = 1.0,
    local_detrend: bool = True,
    min_valid_fraction: float = 0.8,
    **method_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a CSD indicator in a sliding window over a time series.

    The window slides from the start to the end of the series in increments
    of ``step``.  The indicator value is assigned to the midpoint index of
    each window.  Windows containing too many NaN values are skipped.

    Parameters
    ----------
    x : np.ndarray, shape (n,) or (n_dims, n)
        Time series, already globally preprocessed if desired.  Pass a 1D
        array for univariate analysis or a 2D array of shape
        ``(n_dims, n_samples)`` for multivariate analysis.  Multivariate
        input is only supported by the multivariate methods (``'lambda_km'``,
        ``'lambda_dmd'``, ``'lambda_dmd_floquet'``, ``'spatial_corr'``);
        windows are sliced along the time axis (axis 1).
    method : str
        CSD indicator method from :data:`~csdtools.indicators.AVAILABLE_METHODS`.
    window_size : int
        Number of samples per window.
    step : int
        Number of samples to advance between consecutive windows.
        ``step=1`` gives the maximum-overlap (smoothest) indicator.
        Larger values reduce computation but coarsen the output.
    dt : float
        Physical time step passed to the indicator function.
    local_detrend : bool
        Remove a linear trend *within each window* before computing the
        indicator.  Recommended when using a global preprocessing step
        that may leave residual slow variations.
    min_valid_fraction : float
        Minimum fraction of non-NaN values required in a window.
        Windows below this threshold are assigned NaN.  For multivariate
        input this fraction is evaluated over all elements of the window.
    **method_kwargs
        Extra keyword arguments forwarded to :func:`~csdtools.indicators.compute_indicator`
        and then to the specific estimator function.  Pass
        ``exclude_nan_neighbors=True`` here to apply NaN-neighbour exclusion
        for every window, for any method.

    Returns
    -------
    indicator : np.ndarray, shape (n_windows,)
        Indicator value for each window.  Contains NaN for skipped windows.
    times : np.ndarray, shape (n_windows,)
        Index of the midpoint of each window in the original array.

    Raises
    ------
    ValueError
        If ``window_size >= n_samples``.

    Examples
    --------
    >>> ind, t = sliding_window_analysis(x, method='ac1', window_size=100)
    >>> ind, t = sliding_window_analysis(x, 'lambda_acs', 150, step=5,
    ...                                  max_lag=25)
    >>> # multivariate: x_nd has shape (n_dims, n_samples)
    >>> ind, t = sliding_window_analysis(x_nd, 'lambda_km', window_size=300)
    """
    x = np.asarray(x, dtype=float)
    multivariate = x.ndim == 2
    n = x.shape[1] if multivariate else len(x)

    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {AVAILABLE_METHODS}"
        )
    if window_size >= n:
        raise ValueError(
            f"window_size ({window_size}) must be smaller than series "
            f"length ({n})."
        )

    # Warn once before the loop if the window is likely too short
    rec_min = RECOMMENDED_MIN_WINDOW.get(method, 0)
    if window_size < rec_min:
        reason = _MIN_WINDOW_REASON.get(method, "")
        warnings.warn(
            f"window_size={window_size} is below the recommended minimum of "
            f"{rec_min} samples for method='{method}'. Results may be "
            f"unreliable or dominated by noise. {reason}",
            UserWarning,
            stacklevel=2,
        )

    # Separate time-aligned auxiliary series (e.g. the ROSA forcing 'noise')
    # from static kwargs: these are sliced in sync with each window below.
    windowed_kwargs = {}
    for key in list(method_kwargs):
        if key in _WINDOWED_KWARGS and method_kwargs[key] is not None:
            arr = np.asarray(method_kwargs.pop(key), dtype=float)
            if arr.shape[-1] != n:
                raise ValueError(
                    f"'{key}' must be a time-aligned series with length {n} "
                    f"(same as the time axis of x); got shape {arr.shape}."
                )
            windowed_kwargs[key] = arr

    starts = np.arange(0, n - window_size + 1, step)
    indicator = np.full(len(starts), np.nan)
    times = starts + window_size // 2  # midpoint of each window

    for k, i in enumerate(starts):
        window = x[:, i : i + window_size] if multivariate else x[i : i + window_size]
        # Skip windows with too many missing values
        if np.mean(np.isfinite(window)) < min_valid_fraction:
            continue
        kw = dict(method_kwargs)
        for key, arr in windowed_kwargs.items():
            kw[key] = arr[..., i : i + window_size]
        indicator[k] = compute_indicator(
            window, method, dt=dt, detrend=local_detrend, **kw
        )

    return indicator, times.astype(float)


def multi_method_analysis(
    x: np.ndarray,
    methods: list,
    window_size: int = 100,
    step: int = 1,
    dt: float = 1.0,
    local_detrend: bool = True,
    **method_kwargs,
) -> dict:
    """
    Run multiple CSD indicator methods over the same sliding window.

    Convenience wrapper that calls :func:`sliding_window_analysis` for each
    method and returns a dictionary of results.

    Parameters
    ----------
    x : np.ndarray
        1D pre-processed time series.
    methods : list of str
        List of method names to compute.
    window_size : int
        Sliding window size.
    step : int
        Window step size.
    dt : float
        Physical time step.
    local_detrend : bool
        Remove linear trend within each window.
    **method_kwargs
        Forwarded to every indicator function.

    Returns
    -------
    dict
        Keys are method names; values are ``(indicator, times)`` tuples as
        returned by :func:`sliding_window_analysis`.

    Examples
    --------
    >>> results = multi_method_analysis(
    ...     x, ['ac1', 'variance', 'lambda_glsar'], window_size=100
    ... )
    >>> for name, (ind, t) in results.items():
    ...     print(name, ind[:5])
    """
    return {
        m: sliding_window_analysis(
            x, m, window_size, step, dt, local_detrend, **method_kwargs
        )
        for m in methods
    }
