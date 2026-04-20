"""
Internal utility functions shared across csdtools modules.
"""

import warnings
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


def strip_nans(
    x: np.ndarray,
    exclude_neighbors: bool = False,
    stacklevel: int = 3,
) -> np.ndarray:
    """
    Remove NaN values from a 1D array and emit a warning if any are found.

    Parameters
    ----------
    x : np.ndarray
        Input array, may contain NaN / ±inf.
    exclude_neighbors : bool
        If True, also remove the values immediately *adjacent* to each NaN
        position in the original array.  This is relevant for AC1, where a
        pair (x[t], x[t+1]) that straddles a gap would otherwise appear as a
        valid lag-1 pair after NaN removal.
    stacklevel : int
        Passed to ``warnings.warn`` so the warning points at the user's call
        site rather than this internal function.

    Returns
    -------
    np.ndarray
        A copy of x with NaN (and, optionally, neighbouring) values removed.
        Returns the original array unchanged if no NaNs are present.
    """
    x = np.asarray(x, dtype=float)
    nan_mask = ~np.isfinite(x)
    n_nans = int(nan_mask.sum())

    if n_nans == 0:
        return x.copy()

    valid = ~nan_mask

    if exclude_neighbors:
        # Mark values that form a lag-1 pair with a NaN (left or right)
        neighbor_invalid = np.zeros(len(x), dtype=bool)
        neighbor_invalid[:-1] |= nan_mask[1:]   # x[t] where x[t+1] is NaN
        neighbor_invalid[1:]  |= nan_mask[:-1]  # x[t] where x[t-1] is NaN
        valid &= ~neighbor_invalid

    n_removed = int((~valid).sum())
    n_neighbor = n_removed - n_nans

    neighbor_msg = (
        f"  Additionally, {n_neighbor} neighboring value(s) have been "
        "excluded to avoid lag-1 pairs that straddle a gap."
        if n_neighbor > 0
        else ""
    )
    warnings.warn(
        f"Input time series contains {n_nans} NaN/Inf value(s) "
        f"({100.0 * n_nans / len(x):.1f}% of {len(x)} samples). "
        f"{n_removed} value(s) will be removed and the remaining "
        "values treated as consecutive for analysis." + neighbor_msg,
        UserWarning,
        stacklevel=stacklevel,
    )

    return x[valid]


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
