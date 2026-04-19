"""
Internal utility functions shared across csdkit modules.
"""

import numpy as np
from typing import Tuple


def validate_timeseries(x: np.ndarray, min_length: int = 10) -> np.ndarray:
    """Ensure input is a finite-valued 1D numpy array of sufficient length."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    if len(x) < min_length:
        raise ValueError(f"Time series too short: {len(x)} < {min_length}")
    return x


def remove_linear_trend(x: np.ndarray) -> np.ndarray:
    """Subtract best-fit degree-1 polynomial from a 1D array."""
    t = np.arange(len(x), dtype=float)
    c = np.polyfit(t, x, 1)
    return x - np.polyval(c, t)


def remove_quadratic_trend(x: np.ndarray) -> np.ndarray:
    """Subtract best-fit degree-2 polynomial from a 1D array."""
    t = np.arange(len(x), dtype=float)
    c = np.polyfit(t, x, 2)
    return x - np.polyval(c, t)


def running_mean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a centred running mean using uniform box-car convolution.

    Edge values are replicated to avoid shrinking the output.

    Parameters
    ----------
    x : np.ndarray
        1D input array.
    window : int
        Width of the averaging window in samples.

    Returns
    -------
    np.ndarray
        Smoothed array of the same length as x.
    """
    half = window // 2
    remainder = window - half - 1
    padded = np.pad(x, (half, remainder), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def kendall_tau(x: np.ndarray) -> Tuple[float, float]:
    """
    Compute Kendall tau rank correlation of x against its own time index.

    Used to quantify the monotonic trend in a CSD indicator timeseries.

    Parameters
    ----------
    x : np.ndarray
        1D indicator timeseries.

    Returns
    -------
    tau : float
        Kendall tau statistic (positive → increasing trend).
    p_value : float
        Two-sided p-value under H0 of no monotonic trend.
    """
    from scipy.stats import kendalltau as _kt

    t = np.arange(len(x))
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return np.nan, np.nan
    tau, pval = _kt(t[mask], x[mask])
    return float(tau), float(pval)
