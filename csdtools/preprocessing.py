"""
Preprocessing methods for CSD analysis.

Before computing sliding-window CSD indicators, slow non-stationarities
(secular trends, seasonal cycles) must be removed.  These are *not* the
CSD signal itself — they are external forcings that would otherwise
contaminate the indicator.

All functions accept a 1D numpy array and return an array of the same length.
"""

import numpy as np
from typing import Optional

from ._utils import remove_linear_trend, remove_quadratic_trend, running_mean


def detrend_running_mean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Remove a centred running mean from the time series.

    This is the most common preprocessing step in CSD analysis.  The
    running mean tracks slow changes in the baseline level while leaving
    faster fluctuations — which carry the CSD signal — intact.

    The running-mean window should be *larger* than the CSD analysis
    window to avoid inadvertently removing the CSD signal.

    Parameters
    ----------
    x : np.ndarray
        1D input time series.
    window : int
        Width of the running-mean window in samples.

    Returns
    -------
    np.ndarray
        x minus its running mean.
    """
    x = np.asarray(x, dtype=float)
    return x - running_mean(x, window)


def detrend_linear(x: np.ndarray) -> np.ndarray:
    """
    Remove a global linear trend.

    Fits a degree-1 polynomial to the whole time series and subtracts it.
    Appropriate when the series has a single, approximately constant drift.

    Parameters
    ----------
    x : np.ndarray
        1D input time series.

    Returns
    -------
    np.ndarray
        Linearly detrended time series.
    """
    return remove_linear_trend(np.asarray(x, dtype=float))


def detrend_quadratic(x: np.ndarray) -> np.ndarray:
    """
    Remove a global quadratic trend.

    Useful when the baseline accelerates or decelerates monotonically
    (e.g., a forced system with an increasing external parameter).

    Parameters
    ----------
    x : np.ndarray
        1D input time series.

    Returns
    -------
    np.ndarray
        Quadratically detrended time series.
    """
    return remove_quadratic_trend(np.asarray(x, dtype=float))


def detrend_stl(x: np.ndarray, period: int) -> np.ndarray:
    """
    Seasonal-Trend decomposition using LOESS (STL).

    Decomposes the series into trend, seasonal, and residual components
    and returns only the residual.  Ideal for regularly sampled data with
    a known seasonal cycle (e.g., monthly climate observations, period=12).

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    period : int
        Length of the seasonal cycle in samples (e.g., 12 for monthly data).

    Returns
    -------
    np.ndarray
        Residual after trend and seasonal components have been removed.

    Notes
    -----
    Requires statsmodels >= 0.12 (pip install statsmodels).
    """
    from statsmodels.tsa.seasonal import STL

    x = np.asarray(x, dtype=float)
    result = STL(x, period=period).fit()
    return result.resid


def detrend_harmonic(
    x: np.ndarray,
    order: int = 3,
    period: Optional[float] = None,
) -> np.ndarray:
    """
    Remove a seasonal signal using harmonic (Fourier series) regression.

    Fits a model consisting of a constant, a linear trend, and `order`
    pairs of sine/cosine components at integer multiples of the base
    frequency, then returns the residuals.

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    order : int
        Number of harmonic components (default 3 captures the first three
        harmonics of the annual cycle).
    period : float or None
        Period of the base harmonic in samples.  Defaults to len(x)
        if not specified (i.e., a single full-record harmonic fit).

    Returns
    -------
    np.ndarray
        Residuals after harmonic trend removal.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    T = float(period) if period is not None else float(n)
    t = np.arange(n, dtype=float)

    cols = [np.ones(n), t]
    for k in range(1, order + 1):
        omega = 2.0 * np.pi * k / T
        cols.append(np.sin(omega * t))
        cols.append(np.cos(omega * t))

    A = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coeffs


def preprocess(
    x: np.ndarray,
    method: Optional[str] = "running_mean",
    window: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """
    Unified preprocessing interface.

    Parameters
    ----------
    x : np.ndarray
        1D input time series.
    method : str or None
        Preprocessing method:

        ``'running_mean'``
            Subtract centred running mean.  Requires ``window``.
        ``'linear'``
            Remove global linear trend.
        ``'quadratic'``
            Remove global quadratic trend.
        ``'stl'``
            STL seasonal-trend decomposition.  Requires ``period`` kwarg.
        ``'harmonic'``
            Harmonic regression removal.  Optional kwargs: ``order``,
            ``period``.
        ``None``
            No preprocessing (pass through).

    window : int, optional
        Window size for ``'running_mean'``.
    **kwargs
        Additional arguments forwarded to the selected method.

    Returns
    -------
    np.ndarray
        Preprocessed time series of the same length as x.

    Examples
    --------
    >>> x_dt = preprocess(x, method='running_mean', window=200)
    >>> x_dt = preprocess(x, method='stl', period=12)
    >>> x_dt = preprocess(x, method='harmonic', order=2, period=12)
    """
    x = np.asarray(x, dtype=float)

    if method is None:
        return x.copy()
    elif method == "running_mean":
        if window is None:
            raise ValueError("'window' is required for running_mean preprocessing")
        return detrend_running_mean(x, window)
    elif method == "linear":
        return detrend_linear(x)
    elif method == "quadratic":
        return detrend_quadratic(x)
    elif method == "stl":
        return detrend_stl(x, **kwargs)
    elif method == "harmonic":
        return detrend_harmonic(x, **kwargs)
    else:
        raise ValueError(
            f"Unknown preprocessing method: '{method}'. "
            "Choose from: 'running_mean', 'linear', 'quadratic', 'stl', 'harmonic', None."
        )
