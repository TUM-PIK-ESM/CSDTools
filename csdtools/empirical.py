"""
Empirical transition detection and recovery rate estimation.

When a real transition *has* occurred in a time series, it is possible to
estimate the *empirical recovery rate* by fitting an exponential recovery
curve to the data following the transition.  Under CSD theory the empirical
recovery rate is directly related to the linear restoring rate λ:

    x(t) ≈ x_eq + A · exp(λ_emp · t)   after the transition

so λ_emp approximates the true λ near the tipping point.

The module provides:
- :func:`detect_transitions` — locate abrupt changes in the series.
- :func:`fit_exponential_recovery` — fit a recovery curve after a transition.
- :func:`empirical_recovery_rate` — pipeline combining both steps.

References
----------
Lana Blaschke's empirical detection methods (lana/empirical_detection.py).
"""

import numpy as np
from typing import Optional, List, Dict, Any
from scipy import optimize, signal
from scipy.stats import ks_2samp


def detect_transitions(
    x: np.ndarray,
    window_size: int = 100,
    percentile: float = 99.0,
) -> np.ndarray:
    """
    Detect abrupt transition points in a time series.

    Uses a Savitzky-Golay filter to estimate the first derivative and then
    flags time steps where the absolute derivative exceeds a high percentile
    of its distribution.

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    window_size : int
        Savitzky-Golay filter window length.  Smaller values are more
        sensitive to rapid changes; larger values are more robust to noise.
    percentile : float
        Threshold percentile of the derivative distribution.  Higher values
        detect only the most abrupt changes.

    Returns
    -------
    np.ndarray of int
        Indices at which transitions are detected.
    """
    x = np.asarray(x, dtype=float)
    # Savitzky-Golay requires an odd window length
    wl = window_size if window_size % 2 == 1 else window_size + 1
    wl = min(wl, len(x) - 2)
    if wl < 5:
        return np.array([], dtype=int)

    deriv = signal.savgol_filter(x, window_length=wl, polyorder=3, deriv=1)
    threshold = np.percentile(np.abs(deriv), percentile)
    return np.where(np.abs(deriv) >= threshold)[0]


def fit_exponential_recovery(
    x: np.ndarray,
    transition_idx: int,
    n_local_min: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Fit an exponential recovery curve after a detected transition.

    After an abrupt drop (transition), many systems recover approximately
    exponentially:

        x(t) = x_baseline + A · exp(λ_emp · t),   t = 0, 1, 2, …

    A negative λ_emp indicates recovery; a value closer to 0 means slower
    recovery, consistent with CSD near a bifurcation.

    The recovery window spans from the transition index to the first local
    minimum that the trajectory reaches.

    Parameters
    ----------
    x : np.ndarray
        Full time series (not just the post-transition part).
    transition_idx : int
        Index of the detected transition (start of recovery window).
    n_local_min : int
        Number of local minima to search in the post-transition segment.
        The recovery is fitted up to the n_local_min-th local minimum.

    Returns
    -------
    dict or None
        If successful, a dictionary with:

        ``'lambda_empirical'``
            Fitted exponential decay rate (negative for recovery).
        ``'amplitude'``
            Amplitude A of the exponential component.
        ``'r_squared'``
            Coefficient of determination of the exponential fit.
        ``'transition_idx'``
            Echo of the input transition index.
        ``'recovery_end_idx'``
            Index in the original series where the recovery window ends.

        Returns ``None`` if the fit fails or the window is too short.
    """
    x = np.asarray(x, dtype=float)
    post = x[transition_idx:]
    if len(post) < 10:
        return None

    # Locate local minima in the post-transition segment
    local_mins = signal.argrelmin(post, order=3)[0]
    if len(local_mins) == 0:
        return None

    # Use up to the n_local_min-th local minimum
    min_idx = local_mins[min(n_local_min - 1, len(local_mins) - 1)]
    recovery = post[: min_idx + 1]
    if len(recovery) < 4:
        return None

    t_rec = np.arange(len(recovery), dtype=float)
    x_baseline = float(np.mean(recovery[-3:]))

    def model(t, A, lam):
        return x_baseline + A * np.exp(lam * t)

    try:
        p0 = [recovery[0] - x_baseline, -0.05]
        popt, _ = optimize.curve_fit(
            model, t_rec, recovery, p0=p0, maxfev=4000
        )
        residuals = recovery - model(t_rec, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((recovery - np.mean(recovery))**2)
        r_sq = 1.0 - ss_res / (ss_tot + 1e-20)

        return {
            "lambda_empirical": float(popt[1]),
            "amplitude": float(popt[0]),
            "r_squared": float(r_sq),
            "transition_idx": int(transition_idx),
            "recovery_end_idx": int(transition_idx + min_idx),
        }
    except Exception:
        return None


def empirical_recovery_rate(
    x: np.ndarray,
    window_size: int = 100,
    transition_percentile: float = 99.0,
    n_local_min: int = 8,
    min_r_squared: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Detect transitions and estimate empirical recovery rates.

    Combined pipeline that first finds candidate transition points and then
    attempts to fit exponential recoveries at each candidate.  Only fits
    meeting the minimum R² criterion are returned.

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    window_size : int
        Savitzky-Golay window for derivative estimation.
    transition_percentile : float
        Percentile threshold for transition detection.
    n_local_min : int
        Number of local minima to search after each transition.
    min_r_squared : float
        Minimum R² for a recovery fit to be included in the output.

    Returns
    -------
    list of dict
        Each element is a result dictionary from
        :func:`fit_exponential_recovery` meeting the R² criterion,
        sorted by ``r_squared`` in descending order.

    Examples
    --------
    >>> results = empirical_recovery_rate(x, window_size=100)
    >>> for r in results:
    ...     print(f"λ_emp = {r['lambda_empirical']:.3f}, R² = {r['r_squared']:.3f}")
    """
    x = np.asarray(x, dtype=float)
    transitions = detect_transitions(x, window_size, transition_percentile)

    results = []
    used = set()
    for idx in transitions:
        # Avoid fitting overlapping recovery windows
        if any(abs(idx - u) < window_size // 2 for u in used):
            continue
        fit = fit_exponential_recovery(x, idx, n_local_min)
        if fit is not None and fit["r_squared"] >= min_r_squared:
            results.append(fit)
            used.add(idx)

    return sorted(results, key=lambda d: d["r_squared"], reverse=True)
