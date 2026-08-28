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
Smith, T., Traxl, D., & Boers, N. (2022). Nature Climate Change, 12, 477–484.
https://doi.org/10.1038/s41558-022-01352-2

-------
checked by Lana on 2026-05-19
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
        For rough derivative estimation. Smaller values are more
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
    ln = x.shape[0]
    half_window = int(window_size / 2)
    # 1. Create rough "derivative"
    av_derivative = [np.nanmean(x[i - half_window : i]) - np.nanmean(x[i : i + half_window]) for i in range(half_window, ln-half_window)] 
    av_derivative = np.array(av_derivative)
    # Savitzky-Golay requires an odd window length
    av_derivative = signal.savgol_filter(av_derivative, window_length=7, polyorder=1, deriv=0)
    av_derivative = np.array([np.nan]*half_window + list(av_derivative) + [np.nan] * half_window)
    # NOTE: this is a rough heuristic that only flags DOWNWARD transitions --
    # a drop makes (past mean - future mean) positive. Zero out everything
    # below the percentile threshold (on a copy, leaving av_derivative intact)
    # so that argrelmax finds peaks among the surviving high-derivative points.
    masked = av_derivative.copy()
    masked[av_derivative < np.nanpercentile(av_derivative, percentile)] = 0
    # Find relative maxima in the masked derivative array
    transition_times = signal.argrelmax(masked, order=1)[0]
    return transition_times


def fit_exponential_recovery(
    draw: np.ndarray,
    ddetr: np.ndarray,
    transition_idx: int,
    n_local_min: int = 8,
    recovery_window: Optional[int] = 3*12,
    fit_baseline: bool = False,
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
    draw : np.ndarray
        Full time series BEFORE detrending and deseasoning.
    ddetr : np.ndarray
        Detrended and deseasoned time series.
    transition_idx : int
        Index of the detected transition (start of recovery window).
    n_local_min : int
        Number of local minima to search in the post-transition segment.
        The recovery is fitted up to the n_local_min-th local minimum.
        n_local_min << recovery window size to ensure there is data left for fitting.
    recovery_window : Optional[int]
        The window size for the recovery period in number of datapoints (default = 36 being 3 years for monthly data)
    fit_baseline : bool
        Recovery model selector.  If ``False`` (default), the legacy behaviour
        is used: the post-minimum segment is de-meaned and an offset-free
        exponential ``A·exp(λt)`` is fitted.  If ``True``, a three-parameter
        model ``c + A·exp(λt)`` is fitted directly (no de-meaning), so the
        equilibrium level ``c`` the system recovers to is estimated explicitly.
        This matches the physical model ``x(t) = x_eq + A·exp(λt)`` in the
        summary above and is the more principled choice, but it is opt-in so
        that existing results remain reproducible.

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

    # Safety: Ensure alignment
    if not (len(ddetr) == len(draw)):
        raise ValueError("detrended and raw must have the same length.")
    
    end_date  = transition_idx + recovery_window
    if end_date >= len(draw):
        return None  # Not enough data after transition for fitting
    prev_date = transition_idx - recovery_window    
    if prev_date < 0:
        return None  # Not enough data before transition for baseline estimation

    # Mask post-transition period for fitting
    fitting     = ddetr[transition_idx:end_date].copy()
    fitting_raw = draw[transition_idx:end_date].copy()


    #Add KS Statistic: we want the system to recover to the old state
    pre_fitting_raw = draw[prev_date:transition_idx]
    try:
        ks, pval = ks_2samp(pre_fitting_raw, fitting_raw)
    except:
        ks, pval = np.nan, np.nan


    # Find the closest local minima to start the fitting from
    armin = np.nanargmin(fitting[:n_local_min])
    fitting_seg = fitting[armin:]

    # See how far the data drops during the transition (to catch spurious transitions)
    raw_drop = fitting_raw[0] - fitting_raw[armin]

    def exp_model(t, A, lam):
        return A * np.exp(lam * t)
    def exp_jacobian(t, a, b):
        # jacobian (partial derivates) for faster and more stable fitting
        jac = np.array([np.exp(b*t), a*t*np.exp(b*t)]).T
        return jac
    def exp_model_baseline(t, A, lam, c):
        return c + A * np.exp(lam * t)

    try:
        trange = np.arange(fitting_seg.shape[0])
        if fit_baseline:
            # Three-parameter recovery to an explicit equilibrium offset c:
            #   x(t) = c + A*exp(lam*t)   (matches x(t) = x_eq + A*exp(lam*t)).
            # No de-meaning; c is estimated directly.
            fitting_min = fitting_seg
            tail_mean = float(np.nanmean(fitting_min[len(fitting_min) // 2:]))
            p0 = [fitting_min[0] - tail_mean, -0.05, tail_mean]  # [A, lam, c]
            bounds = ([-np.inf, -np.inf, -np.inf], [0, 0, np.inf])
            popt, _ = optimize.curve_fit(
                exp_model_baseline, trange, fitting_min, p0=p0, bounds=bounds)
            model_at = exp_model_baseline(trange, *popt)
            baseline = float(popt[2])
        else:
            # Legacy default: de-mean the segment and fit an offset-free
            # exponential A*exp(lam*t).  Kept as the default so existing
            # results remain reproducible (see fit_baseline in the docstring).
            fitting_min = fitting_seg - np.nanmean(fitting_seg)  # make mean 0
            p0 = [fitting_min[0], -0.05]  # Initial guess: [A, lam]
            bounds = ([-np.inf, -np.inf], [0, 0])  # [min A, min lam], [max ...]
            popt, _ = optimize.curve_fit(
                exp_model, trange, fitting_min, p0=p0, jac=exp_jacobian, bounds=bounds)
            model_at = exp_model(trange, *popt)
            baseline = 0.0

        resid = fitting_min - model_at
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((fitting_min - np.mean(fitting_min))**2)
        r_sq = 1.0 - ss_res / (ss_tot + 1e-20)  # for stability if zero variance
        return {
            "lambda_empirical": float(popt[1]),
            "amplitude": float(popt[0]),
            "baseline": baseline,
            "r_squared": float(r_sq),
            "transition_idx": int(transition_idx),
            "fit_start_idx": int(transition_idx + armin),
            "KS": float(ks),
            "KS_pval": float(pval),
            "raw_drop": float(raw_drop),
        }
    except Exception as e:
        print("Curve fitting failed:", e)
        return None


def empirical_recovery_rate(
    ddetr: np.ndarray,
    draw: np.ndarray,
    window_size: int = 100,
    transition_percentile: float = 99.0,
    n_local_min: int = 8,
    min_r_squared: float = -np.inf,
    raw_drop_min: Optional[float] = None,
    fit_baseline: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detect transitions and estimate empirical recovery rates.

    Combined pipeline that first finds candidate transition points and then
    attempts to fit exponential recoveries at each candidate.  Only fits
    meeting the minimum R² criterion are returned.

    Parameters
    ----------
    ddetr: np.ndarray
        1D time series (detrended and deseasoned).
    draw: np.ndarray
        Raw time series (before detrending/deseasoning), needed for fitting and to check the drop magnitude.
    window_size : int
        Savitzky-Golay window for derivative estimation.
    transition_percentile : float
        Percentile threshold for transition detection.
    n_local_min : int
        Number of local minima to search after each transition.
    min_r_squared : float
        Minimum R² for a recovery fit to be included in the output.  Defaults
        to ``-inf``, i.e. no R² filtering (every fit that passes the other
        criteria is returned).  Set a positive value (e.g. 0.5) to discard
        poorly-fitting recoveries.
    fit_baseline : bool
        Forwarded to :func:`fit_exponential_recovery`.  If ``True``, fit the
        three-parameter ``c + A·exp(λt)`` recovery model instead of the default
        de-meaned offset-free exponential.

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
    ddetr = np.asarray(ddetr, dtype=float)
    draw  = np.asarray(draw, dtype=float)
    transitions = detect_transitions(ddetr, window_size, transition_percentile)

    if raw_drop_min is None:
        raw_drop_min = 0.1 * float(draw.std())  # 0.1 of raw std
    results = []
    used = set()
    for idx in transitions:
        # Avoid fitting overlapping recovery windows
        if any(abs(idx - u) < window_size // 2 for u in used):
            continue
        fit = fit_exponential_recovery(draw, ddetr, idx, n_local_min,
                                       fit_baseline=fit_baseline)
        if fit is not None:
            # additional criteria to filter out spurious transitions
            if (fit["raw_drop"] >= raw_drop_min
                    and fit["KS_pval"] > 0.05
                    and fit["lambda_empirical"] > -3
                    and fit["r_squared"] >= min_r_squared):
                results.append(fit)
                used.add(idx)

    return sorted(results, key=lambda d: d["r_squared"], reverse=True)

