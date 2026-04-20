"""
CSD indicator functions.

Each function takes a pre-processed 1D numpy array representing a single
window of data and returns a scalar estimate.

Mathematical background
-----------------------
Near a saddle-node or Hopf bifurcation the linearised dynamics follow an
Ornstein-Uhlenbeck (OU) process:

    dx = λ x dt + σ dW

where λ < 0 is the *linear restoring rate* (negative for stability) and σ
is the noise amplitude.  In discrete time with step Δt:

    x(t+Δt) = φ x(t) + ε,    φ = exp(λΔt)

Critical Slowing Down (CSD) predicts λ → 0 (φ → 1) as the system approaches
the bifurcation.  This manifests as:

    - Increasing lag-1 autocorrelation (AC1 → 1)
    - Increasing variance (Var = σ²/(1 − φ²) → ∞)
    - Decreasing |λ| estimated by any of the methods below

Sign convention
---------------
λ is negative for a stable system and approaches 0 from below near a
tipping point.  Functions that return λ therefore return a value that
*increases* (becomes less negative) as resilience is lost — consistent
with the AC1 and variance indicators.
"""

import warnings
import numpy as np
from typing import Optional

from ._utils import remove_linear_trend


# ---------------------------------------------------------------------------
# Per-method window-size recommendations
# ---------------------------------------------------------------------------

#: Minimum window size (samples) below which a UserWarning is issued.
#: These are practical thresholds based on statistical reliability, not hard
#: limits — the functions will still attempt a computation below them.
RECOMMENDED_MIN_WINDOW: dict = {
    "ac1":          30,   # need ~30 lag-1 pairs for a stable correlation
    "variance":     20,   # variance estimate needs a few dozen samples
    "lambda_glsar": 50,   # GLS iteration needs enough data to converge
    "lambda_ols":   30,   # Yule-Walker estimator stabilises around 30
    "lambda_acs":   50,   # ACF must be estimated reliably at multiple lags
    "lambda_psd":   30,   # periodogram needs adequate frequency resolution
    "lambda_km":    100,  # non-parametric KM needs many samples; 200+ ideal
    "rosa":         30,   # spectral ratio needs adequate frequency resolution
}

#: Human-readable explanation of why each method needs a minimum window.
_MIN_WINDOW_REASON: dict = {
    "ac1":
        "AC1 is estimated from lag-1 pairs; fewer than ~30 samples yields "
        "an unreliable Pearson correlation.",
    "variance":
        "The sample variance has high relative uncertainty below ~20 samples.",
    "lambda_glsar":
        "GLSAR iterative fitting may not converge reliably below ~50 samples.",
    "lambda_ols":
        "The Yule-Walker AR(1) estimate stabilises around ~30 samples.",
    "lambda_acs":
        "Fitting the autocorrelation structure requires reliable ACF estimates "
        "at multiple lags; ~50 samples is the practical minimum.",
    "lambda_psd":
        "The periodogram has poor frequency resolution below ~30 samples, "
        "making spectral fitting unreliable.",
    "lambda_km":
        "Non-parametric Kramers-Moyal estimation requires kernel density "
        "estimation; ~200 samples is strongly recommended, 100 is the minimum.",
    "rosa":
        "The spectral ratio method needs adequate frequency resolution "
        "in both bands; ~30 samples is the practical minimum.",
}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _prep(x: np.ndarray, detrend: bool = True) -> np.ndarray:
    """Convert to float, strip any residual NaNs, optionally detrend."""
    x = np.asarray(x, dtype=float).copy()
    # Safety strip: upstream should have removed NaNs, but guard anyway
    finite = np.isfinite(x)
    if not finite.all():
        x = x[finite]
    if len(x) == 0:
        return x
    if detrend:
        x = remove_linear_trend(x)
    return x


# ---------------------------------------------------------------------------
# Direct CSD indicators
# ---------------------------------------------------------------------------

def estimate_ac1(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Lag-1 autocorrelation coefficient (AC1).

    For an AR(1) process x(t) = φ·x(t−1) + σ·ξ(t), the expected value of
    AC1 equals φ = exp(λΔt).  As λ → 0, AC1 → 1.

    **CSD signature**: AC1 increases toward 1.

    Parameters
    ----------
    x : array-like, shape (n,)
        Pre-processed time series window.  NaN values are stripped by
        :func:`compute_indicator` before this function is called.
        Pass ``exclude_nan_neighbors=True`` to :func:`compute_indicator`
        (or to :func:`~csdtools.analyze_timeseries`) to also remove values
        that were adjacent to NaN positions in the original series.
    dt : float
        Time step (not used by this estimator; present for API consistency).
    detrend : bool
        Remove linear trend within window before computing.

    Returns
    -------
    float
        Pearson correlation coefficient between lag-0 and lag-1 values.
    """
    x = _prep(x, detrend)
    if len(x) < 4:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def estimate_variance(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Sample variance after linear detrending.

    For a stable OU process: Var = σ²/(1 − φ²) → ∞ as φ → 1.

    **CSD signature**: variance increases.

    Parameters
    ----------
    x : array-like, shape (n,)
        Pre-processed time series window.
    dt : float
        Time step (unused).
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Sample variance (ddof=1).
    """
    x = _prep(x, detrend)
    return float(np.var(x, ddof=1))


# ---------------------------------------------------------------------------
# Lambda estimators — parametric
# ---------------------------------------------------------------------------

def estimate_lambda_glsar(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Restoring rate via Generalised Least Squares AR(1) (GLSAR).

    Fits the regression x(t) = φ·x(t−1) + ε(t) using Generalised Least
    Squares with an AR(1) noise model and computes λ = log(φ)/Δt.

    GLS is preferred over plain OLS because the regression residuals are
    themselves autocorrelated; ignoring this leads to underestimated φ.
    The iterative GLSAR procedure corrects for this bias.

    **CSD signature**: λ increases toward 0.

    Parameters
    ----------
    x : array-like
        Pre-processed window (≥ 30 samples recommended).
    dt : float
        Physical time step (determines the units of the returned λ).
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Continuous-time restoring rate λ (negative for stable systems).

    References
    ----------
    Held & Kleinen (2004); Dakos et al. (2008); Boers (2021).
    """
    import statsmodels.api as sm

    x = _prep(x, detrend)
    if len(x) < 6 or np.var(x) == 0:
        return np.nan
    try:
        model = sm.GLSAR(x[1:], x[:-1], rho=1)
        result = model.iterative_fit(maxiter=20, rtol=1e-4)
        phi = float(result.params[0])
        if phi <= 0 or not np.isfinite(phi):
            return np.nan
        return np.log(phi) / dt
    except Exception:
        return np.nan


def estimate_lambda_ols(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Restoring rate via Ordinary Least Squares AR(1) (OLS).

    Uses the Yule-Walker estimator φ = Cov(x[t], x[t−1]) / Var(x[t−1])
    and converts to λ = log(φ)/Δt.

    Faster than GLSAR and asymptotically equivalent, but slightly more
    biased in short windows.  Useful as a quick, cheap estimate.

    **CSD signature**: λ increases toward 0.

    Parameters
    ----------
    x : array-like
        Pre-processed window.
    dt : float
        Physical time step.
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Continuous-time restoring rate λ.
    """
    x = _prep(x, detrend)
    if len(x) < 4 or np.var(x) == 0:
        return np.nan
    try:
        cv = np.cov(x[1:], x[:-1])
        phi = cv[0, 1] / (np.var(x[:-1]) + 1e-20)
        if phi <= 0 or not np.isfinite(phi):
            return np.nan
        return np.log(phi) / dt
    except Exception:
        return np.nan


def estimate_lambda_acs(
    x: np.ndarray,
    dt: float = 1.0,
    max_lag: int = 20,
    detrend: bool = True,
) -> float:
    """
    Restoring rate by fitting the theoretical autocorrelation structure (ACS).

    For an OU process the autocorrelation at lag τ is:

        C(τ) = exp(λ τ)

    Minimises the squared difference between the empirical ACF and this
    theoretical form over lags 1 … max_lag.  Using multiple lags makes
    the estimate more robust than a single-lag estimator.

    **CSD signature**: λ increases toward 0.

    Parameters
    ----------
    x : array-like
        Pre-processed window.
    dt : float
        Physical time step.
    max_lag : int
        Maximum lag (samples) used in the fitting.
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Continuous-time restoring rate λ.

    References
    ----------
    Morr & Boers (2024).
    """
    from scipy import optimize
    from statsmodels.tsa.stattools import acf as _acf

    x = _prep(x, detrend)
    nlags = min(max_lag, max(2, len(x) // 4))
    try:
        obs = _acf(x, nlags=nlags, fft=True)[1:]
        lags = np.arange(1, nlags + 1, dtype=float) * dt

        def cost(p):
            return np.sum((obs - np.exp(p[0] * lags)) ** 2)

        res = optimize.minimize(
            cost, x0=[-0.1], bounds=[(-50.0, -1e-8)], method="L-BFGS-B"
        )
        return float(res.x[0]) if res.success else np.nan
    except Exception:
        return np.nan


def estimate_lambda_psd(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Restoring rate by fitting the theoretical power spectral density (PSD).

    For a continuous-time OU process with restoring rate λ and noise σ, the
    one-sided power spectral density is:

        S(f) = 2σ² / (λ² + (2πf)²)

    Fits this to the periodogram of the window in log-space (which is
    equivalent to minimising the mean-square error of the log-spectrum, and
    is more robust to the wide dynamic range typical of spectral estimates).

    **CSD signature**: λ increases toward 0.

    Parameters
    ----------
    x : array-like
        Pre-processed window (≥ 16 samples recommended).
    dt : float
        Physical time step.
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Continuous-time restoring rate λ (negative).

    References
    ----------
    Morr & Boers (2024); Morr et al. (2024).
    """
    from scipy import optimize, signal

    x = _prep(x, detrend)
    if len(x) < 8:
        return np.nan
    try:
        freqs, psd = signal.periodogram(x, fs=1.0 / dt)
        freqs, psd = freqs[1:], psd[1:]   # exclude DC
        if not np.any(psd > 0):
            return np.nan

        def cost(p):
            lam_abs = np.abs(p[0]) + 1e-10
            sig2 = np.abs(p[1]) + 1e-20
            theo = 2.0 * sig2 / (lam_abs**2 + (2.0 * np.pi * freqs) ** 2)
            return np.sum((np.log(psd + 1e-20) - np.log(theo)) ** 2)

        res = optimize.minimize(
            cost, x0=[0.1, float(np.var(x))], method="Nelder-Mead"
        )
        return -float(np.abs(res.x[0]))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Lambda estimators — non-parametric
# ---------------------------------------------------------------------------

def estimate_lambda_km(
    x: np.ndarray,
    dt: float = 1.0,
    bin_number: int = 50,
    bw: Optional[float] = None,
    center_fraction: float = 0.5,
    detrend: bool = True,
) -> float:
    """
    Restoring rate via non-parametric Kramers-Moyal (KM) estimation.

    Estimates the 1st Kramers-Moyal coefficient (drift) D¹(x) from the data
    using a kernel density approach.  For an OU process D¹(x) = λ·x, so the
    slope of D¹ vs x gives λ directly without assuming a specific noise model.

    **CSD signature**: λ increases toward 0.

    Parameters
    ----------
    x : array-like
        Pre-processed window.  Non-parametric estimation requires more data
        than the parametric methods; ≥ 200 samples is recommended.
    dt : float
        Physical time step (used to normalise the KM coefficients).
    bin_number : int
        Number of bins for the KM estimator.
    bw : float or None
        Kernel bandwidth.  If None, estimated automatically from the data.
    center_fraction : float
        Fraction of central bins used for the linear fit of D¹ vs x.
        Edge bins can be noisy due to sparse sampling.
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Continuous-time restoring rate λ.

    Notes
    -----
    Requires the ``kramersmoyal`` package::

        pip install kramersmoyal

    References
    ----------
    Morr & Boers (2024); Morr et al. (2024).
    """
    x = _prep(x, detrend)
    if len(x) < 50:
        return np.nan
    try:
        from kramersmoyal import km  # type: ignore

        powers = np.array([[1], [2]])
        kmc, edges = km(x.reshape(-1, 1), bins=np.array([bin_number]),
                        powers=powers, bw=bw)
        km1 = kmc[0].flatten() / dt   # drift normalised by time step
        centres = (edges[0][:-1] + edges[0][1:]) / 2.0

        n = len(centres)
        i0 = int(n * (1.0 - center_fraction) / 2.0)
        i1 = n - i0
        cx = centres[i0:i1]
        cy = km1[i0:i1]
        mask = np.isfinite(cy)
        if mask.sum() < 3:
            return np.nan
        lam = np.polyfit(cx[mask], cy[mask], 1)[0]
        return float(lam)
    except ImportError:
        warnings.warn(
            "The 'kramersmoyal' package is not installed; 'lambda_km' is unavailable. "
            "Install with: pip install kramersmoyal"
        )
        return np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Spectral ratio estimator (ROSA)
# ---------------------------------------------------------------------------

def estimate_rosa(
    x: np.ndarray,
    dt: float = 1.0,
    eta: float = 0.5,
    detrend: bool = True,
) -> float:
    """
    Restoring rate via the Ratio Of Spectra Approach (ROSA).

    For a continuous-time OU process, the power spectral density is:

        S(f) ∝ 1 / (λ² + (2πf)²)

    The ratio of spectral power in a low-frequency band to a high-frequency
    band is therefore a deterministic function of λ.  Inverting this
    relationship yields a direct estimate of λ without needing to fit the
    full spectrum.

    The split between low and high bands is controlled by ``eta``:
    frequencies below eta × f_Nyquist are the "low" band.

    **CSD signature**: λ increases toward 0 (ratio of low/high power grows).

    Parameters
    ----------
    x : array-like
        Pre-processed window.
    dt : float
        Physical time step.
    eta : float in (0, 1)
        Fractional frequency split point.  eta=0.5 places the boundary at
        the mid-point of the available frequency range.
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Continuous-time restoring rate λ (negative).

    References
    ----------
    Clarke et al. (2023). Seeking more robust early warning signals for
    climate tipping points: the ratio of spectra method.
    """
    from scipy import optimize, signal

    x = _prep(x, detrend)
    if len(x) < 8:
        return np.nan
    try:
        freqs, psd = signal.periodogram(x, fs=1.0 / dt)
        freqs, psd = freqs[1:], psd[1:]   # exclude DC
        f_split = eta * freqs[-1]
        low_mask = freqs <= f_split
        high_mask = freqs > f_split
        if low_mask.sum() == 0 or high_mask.sum() == 0:
            return np.nan

        S_low = np.mean(psd[low_mask])
        S_high = np.mean(psd[high_mask]) + 1e-20
        observed_ratio = S_low / S_high

        # Mean representative frequencies for each band
        f_l = np.mean(freqs[low_mask])
        f_h = np.mean(freqs[high_mask])

        def theo_ratio(lam_abs: float) -> float:
            """Theoretical S_low/S_high for an OU process."""
            return (lam_abs**2 + (2.0 * np.pi * f_h) ** 2) / (
                lam_abs**2 + (2.0 * np.pi * f_l) ** 2
            )

        def cost(p: np.ndarray) -> float:
            return (theo_ratio(np.abs(p[0]) + 1e-10) - observed_ratio) ** 2

        res = optimize.minimize(cost, x0=[0.1], method="Nelder-Mead")
        return -float(np.abs(res.x[0]))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Registry and unified interface
# ---------------------------------------------------------------------------

#: Maps method name → (function, bool: uses_dt).
#: The bool is informational; all functions accept dt for API uniformity.
INDICATOR_REGISTRY: dict = {
    "ac1":          (estimate_ac1,          False),
    "variance":     (estimate_variance,     False),
    "lambda_glsar": (estimate_lambda_glsar, True),
    "lambda_ols":   (estimate_lambda_ols,   True),
    "lambda_acs":   (estimate_lambda_acs,   True),
    "lambda_psd":   (estimate_lambda_psd,   True),
    "lambda_km":    (estimate_lambda_km,    True),
    "rosa":         (estimate_rosa,         True),
}

#: List of valid method names for ``analyze_timeseries``.
AVAILABLE_METHODS: list = list(INDICATOR_REGISTRY.keys())


def compute_indicator(
    x: np.ndarray,
    method: str,
    dt: float = 1.0,
    detrend: bool = True,
    exclude_nan_neighbors: bool = False,
    **kwargs,
) -> float:
    """
    Compute a single CSD indicator value on a 1D array.

    This is the unified entry point for all eight indicator methods.
    NaN handling (including optional neighbour exclusion) is performed here,
    so individual estimator functions always receive clean arrays.

    Parameters
    ----------
    x : array-like
        1D time series or pre-extracted window.  May contain NaN values;
        they are removed before the estimator is called.
    method : str
        One of :data:`AVAILABLE_METHODS`.
    dt : float
        Physical time step.
    detrend : bool
        Remove linear trend within x before computing.
    exclude_nan_neighbors : bool
        If ``True``, also remove the values immediately adjacent to each NaN
        position before calling the estimator.  Applies to **all** methods.

        This matters most for ``'ac1'``, where a pair (x[t], x[t+1]) that
        straddles a data gap would otherwise appear as a valid lag-1 pair
        after plain NaN removal.  For spectral and regression methods it
        removes a small number of additional samples near gap edges, which
        can slightly improve stationarity within the window.

        When using :func:`~csdtools.analyze_timeseries`, NaN neighbours are
        already removed from the full series before windowing, so this
        parameter is only meaningful for direct calls to ``compute_indicator``
        or :func:`~csdtools.sliding_window.sliding_window_analysis` on data
        that still contains NaNs.
    **kwargs
        Additional arguments forwarded to the specific estimator function.

    Returns
    -------
    float
        Indicator value.  Returns ``np.nan`` if computation fails.

    Examples
    --------
    >>> val = compute_indicator(window, 'lambda_glsar', dt=1.0)
    >>> val = compute_indicator(window, 'lambda_acs', dt=1.0, max_lag=30)
    >>> val = compute_indicator(window_with_nans, 'ac1',
    ...                         exclude_nan_neighbors=True)
    """
    if method not in INDICATOR_REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. Available: {AVAILABLE_METHODS}"
        )

    x = np.asarray(x, dtype=float)
    nan_mask = ~np.isfinite(x)
    if nan_mask.any():
        valid = ~nan_mask
        if exclude_nan_neighbors:
            nb = np.zeros(len(x), dtype=bool)
            nb[:-1] |= nan_mask[1:]   # left neighbour of a NaN
            nb[1:]  |= nan_mask[:-1]  # right neighbour of a NaN
            valid &= ~nb
        x = x[valid]

    func, _ = INDICATOR_REGISTRY[method]
    return func(x, dt=dt, detrend=detrend, **kwargs)
