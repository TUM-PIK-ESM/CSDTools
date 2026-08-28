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

    - Increasing lag-1 autocorrelation (AC1 = φ → 1)
    - Increasing variance (Var = σ²/2|λ| → ∞)
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
    "skewness":     30,   # third moment needs enough samples to be reliable
    "lambda_ac1":   30,   # needs ~30 lag-1 pairs for a stable Pearson correlation
    "lambda_var":   30,   # OLS residual noise estimate needs ~30 samples
    "lambda_glsar": 50,   # GLS iteration needs enough data to converge
    "lambda_ols":   30,   # Yule-Walker estimator stabilises around 30
    "lambda_acs":   100,   # ACF must be estimated reliably at multiple lags
    "lambda_psd":   100,   # periodogram needs adequate frequency resolution
    "lambda_km":    100,  # non-parametric KM needs many samples; 200+ ideal
    "rosa":         30,   # spectral ratio needs adequate frequency resolution
    "lambda_dmd":   50,   # DMD needs enough snapshots to resolve the operator
    "lambda_dmd_floquet": 48,  # needs several periods; default period=12 -> ~4 cycles
    "spatial_corr": 30,   # Pearson correlation needs enough samples per cell
}

#: Human-readable explanation of why each method needs a minimum window.
_MIN_WINDOW_REASON: dict = {
    "ac1":
        "AC1 is estimated from lag-1 pairs; fewer than ~30 samples yields "
        "an unreliable Pearson correlation.",
    "variance":
        "The sample variance has high relative uncertainty below ~20 samples.",
    "skewness":
        "The third standardised moment has high relative uncertainty below "
        "~30 samples.",
    "lambda_ac1":
        "AC1-based λ uses a single Pearson correlation; fewer than ~30 lag-1 "
        "pairs yields an unreliable estimate.",
    "lambda_var":
        "Variance-based λ requires an OLS residual noise estimate; fewer than "
        "~30 samples makes both the noise and variance estimates unstable.",
    "lambda_glsar":
        "GLSAR iterative fitting may not converge reliably below ~50 samples.",
    "lambda_ols":
        "The Yule-Walker AR(1) estimate stabilises around ~30 samples.",
    "lambda_acs":
        "Fitting the autocorrelation structure requires reliable ACF estimates "
        "at multiple lags; ~100 samples is the practical minimum.",
    "lambda_psd":
        "The periodogram has poor frequency resolution below ~100 samples, "
        "making spectral fitting unreliable.",
    "lambda_km":
        "Non-parametric Kramers-Moyal estimation requires kernel density "
        "estimation; ~200 samples is strongly recommended and 100 is the "
        "recommended minimum (the estimator itself returns NaN below a hard "
        "floor of 50 samples).",
    "rosa":
        "ROSA fits the response-to-forcing spectral ratio, which needs "
        "adequate periodogram frequency resolution; ~30 samples is the "
        "practical minimum.",
    "lambda_dmd":
        "DMD requires more snapshots than state dimensions to solve the "
        "linear operator; ~50 samples is the practical minimum.",
    "lambda_dmd_floquet":
        "Floquet DMD fits a period-T map, so the window must span several "
        "periods; use at least ~(num_cycles+1)*period samples (~48 for the "
        "default period=12).",
    "spatial_corr":
        "Pearson correlation between center cell and neighbors requires sufficient samples "
        "for a stable estimate; ~30 samples per window is the practical minimum.",
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
        Empirical lag-1 autocorrelation.

    References
    ----------
    Held, H. & Kleinen, T. (2004). Geophysical Research Letters, 31, L23207.
    https://doi.org/10.1029/2004GL020972

    Dakos, V., et al. (2008). PNAS, 105(38), 14308-14312.
    https://doi.org/10.1073/pnas.0802430105

    -------
    checked by Lana on 2026-05-19
    """
    import statsmodels.api as sm

    x = _prep(x, detrend)
    if len(x) < 4:
        return np.nan
    return sm.tsa.acf(x,nlags=1)[1]


def estimate_variance(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Sample variance after linear detrending.

    For a stable OU process: Var = σ²/2|λ| → ∞ as λ → 0.

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

    References
    ----------
    Held, H. & Kleinen, T. (2004). Geophysical Research Letters, 31, L23207.
    https://doi.org/10.1029/2004GL020972

    Dakos, V., et al. (2008). PNAS, 105(38), 14308-14312.
    https://doi.org/10.1073/pnas.0802430105

    -------
    checked by Lana on 2026-05-19
    """
    x = _prep(x, detrend)
    return float(np.var(x, ddof=1))


def estimate_skewness(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Sample skewness (adjusted Fisher-Pearson standardised moment coefficient).

    Near a fold bifurcation the effective potential well becomes asymmetric,
    causing the distribution of fluctuations to skew.  The magnitude of
    skewness increases as the system approaches the bifurcation; the sign
    depends on whether the bifurcation is approached from above or below.

    Uses the bias-corrected (adjusted) estimator G1, equivalent to
    ``scipy.stats.skew(x, bias=False)``.

    **CSD signature**: |skewness| increases.

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
        Adjusted skewness coefficient.
    """
    x = _prep(x, detrend)
    n = len(x)
    if n < 4:
        return np.nan
    xbar = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0.0:
        return np.nan
    m3 = np.mean((x - xbar) ** 3)
    return float(np.sqrt(n * (n - 1)) / (n - 2) * m3 / s ** 3)


# ---------------------------------------------------------------------------
# Lambda estimators — parametric
# ---------------------------------------------------------------------------

def estimate_a_glsar(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Difference-regression slope from Generalised Least Squares AR(1) (GLSAR).

    Fits the differenced regression ``dx = a·x(t) + ε(t)`` (with
    ``dx = x(t+1) − x(t)``) using Generalised Least Squares with an AR(1)
    noise model, and returns the slope ``a``.  This slope equals ``φ − 1``,
    where ``φ`` is the AR(1) coefficient of ``x(t+1) = φ·x(t) + ε``; the
    continuous-time restoring rate follows as ``λ = log(a + 1)/Δt`` (see
    :func:`estimate_lambda_glsar`).

    GLS is preferred over plain OLS because the regression residuals are
    themselves autocorrelated; ignoring this biases the slope.  The iterative
    GLSAR procedure corrects for this.

    **CSD signature**: ``a`` (= φ − 1) is negative for a stable system and
    increases toward 0 as resilience is lost.

    Parameters
    ----------
    x : array-like
        Pre-processed window (≥ 30 samples recommended).
    dt : float
        Time step (unused).
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Difference-regression slope ``a = φ − 1`` (negative for stable
        systems, in ``(−1, 0)``; increases toward 0 near a bifurcation).
    """
    import statsmodels.api as sm

    x = _prep(x, detrend)
    # Guard on the raw data column *before* add_constant, otherwise np.var
    # would include the constant column and never be zero.
    if len(x) < 6 or np.var(x) == 0:
        return np.nan
    dx = x[1:] - x[:-1]
    x = sm.add_constant(x)
    try:
        model = sm.GLSAR(endog=dx, exog=x[:-1], rho=1)
        result = model.iterative_fit(maxiter=10, rtol=1e-4)
        phi = float(result.params[1])
        return phi
    except Exception:
        return np.nan

def estimate_lambda_glsar(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Restoring rate via Generalised Least Squares AR(1) (GLSAR).

    Fits the regression dx = φ·x·dt + ε(t) using Generalised Least
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
    Boers, N. (2021). Observation-based early-warning signals for a collapse
    of the Atlantic Meridional Overturning Circulation.
    Nature Climate Change, 11, 680-688.
    https://doi.org/10.1038/s41558-021-01097-4

    -------
    checked by Lana on 2026-05-19
    """
    # Delegate the GLSAR slope estimation to estimate_a_glsar and convert
    # the returned slope (phi) to a continuous-time restoring rate λ.
    try:
        phi = estimate_a_glsar(x, detrend=detrend)
        if phi is None or np.isnan(phi) or phi <= -1 or not np.isfinite(phi):
            return np.nan
        return np.log(phi + 1) / dt
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
    
    -------
    @Maya, @Andreas, I did have some OLS code (which I never really used though) but its not that, so I thought I best do not change anything here
    (Andreas): This is exactly the same as lambda computed from the ac1 right?
    """
    x = _prep(x, detrend)
    if len(x) < 4 or np.var(x) == 0:
        return np.nan
    try:
        cv = np.cov(x[1:], x[:-1])
        # np.cov uses ddof=1; match it in the denominator variance so phi is
        # not inflated by a factor n/(n-1).
        phi = cv[0, 1] / (np.var(x[:-1], ddof=1) + 1e-20)
        if phi <= 0 or not np.isfinite(phi):
            return np.nan
        return np.log(phi) / dt
    except Exception:
        return np.nan


def estimate_lambda_ac1(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Restoring rate via lag-1 autocorrelation (AC1).

    Estimates the AR(1) coefficient φ as the Pearson correlation between
    successive pairs x[t] and x[t+1], then converts to a continuous-time
    restoring rate:

        λ = log(φ) / Δt

    As CSD develops, φ → 1 and λ → 0 from below.

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
        Continuous-time restoring rate λ (negative for stable systems).

    References
    ----------
    Smith, T. & Boers, N. (2023). Reliability of vegetation resilience
    estimates depends on biomass density. Nature Ecology & Evolution,
    7(11), 1799-1808.
    https://doi.org/10.1038/s41559-023-02194-7
    """
    x = _prep(x, detrend)
    if len(x) < 4:
        return np.nan
    try:
        a, b = x[:-1], x[1:]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 3:
            return np.nan
        phi = float(np.corrcoef(a[valid], b[valid])[0, 1])
        if not np.isfinite(phi) or phi <= 0.0:
            return np.nan
        return float(np.log(phi) / dt)
    except Exception:
        return np.nan


def estimate_lambda_var(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
) -> float:
    """
    Restoring rate via the variance-to-noise ratio.

    For a stable OU process with restoring rate λ, noise amplitude σ, and
    stationary variance V = σ²/(2|λ|), the relationship between the
    per-step noise variance σ²_noise and V gives:

        λ = (1/2) · log(1 − σ²_noise / V) / Δt

    σ²_noise is estimated from OLS residuals of the regression
    dx = λ_step · x[t] + ε, where dx = x[t+1] − x[t].

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
        Continuous-time restoring rate λ (negative for stable systems).

    References
    ----------
    Smith, T. & Boers, N. (2023). Reliability of vegetation resilience
    estimates depends on biomass density. Nature Ecology & Evolution,
    7(11), 1799-1808.
    https://doi.org/10.1038/s41559-023-02194-7
    """
    from scipy.stats import linregress as _linregress

    x = _prep(x, detrend)
    if len(x) < 4:
        return np.nan
    try:
        dx = x[1:] - x[:-1]
        reg = _linregress(x[:-1], dx)
        lam_step, intercept = reg[0], reg[1]
        diff = dx - (lam_step * x[:-1] + intercept)
        sigma_noise = float(np.std(diff))
        var = float(np.var(x))
        if var == 0.0 or sigma_noise ** 2 >= var:
            return np.nan
        return float(0.5 * np.log(1.0 - sigma_noise ** 2 / var) / dt)
    except Exception:
        return np.nan


def estimate_lambda_acs(
    x: np.ndarray,
    dt: float = 1.0,
    max_lag: int = 3,
    detrend: bool = True,
) -> float:
    """
    Restoring rate by fitting the theoretical autocorrelation structure (ACS).

    Fits a two-mode theoretical ACS to the empirical ACF over lags 1…max_lag.
    The two-mode ACS is the autocorrelation of a scalar observable from a 2D
    OU process with two decay rates λ and θ (both positive):

        General case (|λ − θ| ≥ 0.01):
            C(τ) = (θ·exp(−λτ) − λ·exp(−θτ)) / (θ − λ)

        Degenerate case (λ ≈ θ):
            C(τ) = exp(−λτ) · (1 + λτ)

    The dominant (slowest-decaying) mode min(λ, θ) is returned as a negative
    restoring rate, consistent with the sign convention of all other estimators.
    As CSD develops, this value increases toward 0.

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
        Continuous-time restoring rate λ (negative for stable systems).

    References
    ----------
    Morr, A. & Boers, N. (2024). Physical Review X, 14, 021037.
    https://doi.org/10.1103/PhysRevX.14.021037
    """
    from scipy import optimize
    from statsmodels.tsa.stattools import acf as _acf

    x = _prep(x, detrend)
    nlags = min(max_lag, max(2, len(x) // 4))
    try:
        obs = _acf(x, nlags=nlags, fft=True)[1:]
        lags = np.arange(1, nlags + 1, dtype=float) * dt

        def cost(p):
            lam   = min(abs(p[0]), 5.0)
            theta = min(abs(p[1]), 5.0)
            if abs(lam - theta) < 0.01:
                theo = np.exp(-lam * lags) * (1.0 + lam * lags)
            else:
                theo = (theta * np.exp(-lam * lags) - lam * np.exp(-theta * lags)) / (theta - lam)
            return float(np.sum((obs - theo) ** 2))

        res = optimize.minimize(
            cost, x0=[0.1, 0.5], method="Nelder-Mead",
            options={"maxiter": 100000},
        )
        lam   = min(abs(res.x[0]), 5.0)
        theta = min(abs(res.x[1]), 5.0)
        # Return the dominant eigenvalue as a negative restoring rate
        return -float(min(lam, theta))
    except Exception:
        return np.nan


def estimate_lambda_psd(
    x: np.ndarray,
    dt: float = 1.0,
    band_size: int = 10,
    detrend: bool = True,
) -> float:
    """
    Restoring rate by fitting the theoretical power spectral density (PSD).

    Fits a two-mode discrete-time PSD model to the band-averaged log-periodogram.
    The theoretical PSD is the exact one-sided spectrum of a scalar observable
    from a discrete-time AR(2) process with poles at exp(−λ) and exp(−θ)
    (λ, θ > 0) driven by white noise of amplitude κ:

        General case (|λ − θ| ≥ 0.01):
            S(f) = κ²/[2λθ(λ²−θ²)] · [λ(1−e^{−2θ})/(1+e^{−2θ}−2e^{−θ}cos(2πf))
                                       − θ(1−e^{−2λ})/(1+e^{−2λ}−2e^{−λ}cos(2πf))]

        Degenerate case (λ ≈ θ):
            S(f) = κ²/(4λ³) · [−λ − cos(2πf)sinh(λ) + cosh(λ)(λcos(2πf)+sinh(λ))]
                              / (cos(2πf) − cosh(λ))²

    Observed and theoretical spectra are band-averaged (``band_size`` bins per
    band) before log-transformation and comparison, which reduces the noise of
    individual periodogram ordinates.

    The dominant (slowest-decaying) mode min(λ, θ) is returned as a negative
    restoring rate in per-unit-time units.

    **CSD signature**: λ increases toward 0.

    Parameters
    ----------
    x : array-like
        Pre-processed window (≥ 16 samples recommended).
    dt : float
        Physical time step.  λ and θ are fitted in per-sample units and
        divided by dt to convert to per-unit-time.
    band_size : int
        Number of frequency bins averaged together before fitting.
        Default 10 matches the reference implementation.
    detrend : bool
        Remove linear trend within window.

    Returns
    -------
    float
        Continuous-time restoring rate λ (negative for stable systems).

    References
    ----------
    Morr, A. & Boers, N. (2024). Physical Review X, 14, 021037.
    https://doi.org/10.1103/PhysRevX.14.021037
    """
    from scipy import optimize

    x = _prep(x, detrend)
    n = len(x)
    if n < 8:
        return np.nan
    try:
        # One-sided periodogram: rfft, exclude DC, take positive half
        psd = (np.abs(np.fft.rfft(x)[1:round(n / 2)]) ** 2) / n

        # Band-average: groups of band_size consecutive bins
        def _band_avg(arr):
            n_bands = round(len(arr) / band_size - 1)
            return np.array([np.mean(arr[i * band_size:(i + 1) * band_size])
                             for i in range(n_bands)])

        obs = np.log(_band_avg(psd))
        if len(obs) == 0 or not np.all(np.isfinite(obs)):
            return np.nan

        # cos(2πf) for each periodogram frequency (cycles per sample)
        cos_arr = np.cos(2.0 * np.pi * np.fft.fftfreq(n)[1:round(n / 2)])

        def _theo_psd(lam, theta, kappa):
            el = np.exp(lam)
            et = np.exp(theta)
            c  = cos_arr
            if abs(lam - theta) < 0.01:
                theo = (kappa ** 2 / (4.0 * lam ** 3) *
                        (-lam - c * np.sinh(lam) +
                         np.cosh(lam) * (lam * c + np.sinh(lam))) /
                        (c - np.cosh(lam)) ** 2)
            else:
                theo = (kappa ** 2 / (2.0 * lam * theta * (lam ** 2 - theta ** 2)) *
                        (lam * (1.0 - et ** -2) / (1.0 + et ** -2 - 2.0 * et ** -1 * c) -
                         theta * (1.0 - el ** -2) / (1.0 + el ** -2 - 2.0 * el ** -1 * c)))
            return np.log(_band_avg(theo))

        def cost(p):
            lam   = min(abs(p[0]), 5.0)
            theta = min(abs(p[1]), 5.0)
            kappa = abs(p[2])
            theo  = _theo_psd(lam, theta, kappa)
            if len(theo) != len(obs) or not np.all(np.isfinite(theo)):
                return 1e20
            return float(np.sum(np.abs(obs - theo) ** 2))

        res = optimize.minimize(
            cost, x0=[0.1, 0.5, float(np.std(x))],
            method="Nelder-Mead",
            options={"maxiter": 100000, "disp": False},
        )
        lam   = min(abs(res.x[0]), 5.0)
        theta = min(abs(res.x[1]), 5.0)
        # λ and θ are per-sample rates; divide by dt to give per-unit-time
        return -float(min(lam, theta)) / dt
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

    Supports both univariate (1D) and multivariate (nD) time series.

    **1D**: estimates the drift D¹(x) = λ·x by linear regression on the first
    KM coefficient.  The slope gives λ directly without assuming a parametric
    noise model.

    **nD**: estimates the full drift Jacobian B (so that D¹(x) ≈ B·x) and
    returns the dominant (least-negative) real eigenvalue of B.  This is the
    eigenvalue closest to 0 and the one expected to increase toward 0 as CSD
    develops.  Each variable is normalised by its standard deviation before
    the KM estimation to improve numerical conditioning.  Input must have
    shape ``(n_dims, n_samples)`` with each row being one variable.

    **CSD signature**: λ increases toward 0.

    Parameters
    ----------
    x : array-like, shape (n,) or (n_dims, n_samples)
        Time series window.  For univariate analysis pass a 1D array of
        length n.  For multivariate analysis pass a 2D array where each
        row is one variable and each column is one time step; n_dims and
        n_samples are inferred from the array shape.  Non-parametric
        estimation requires more data than the parametric methods: the hard
        floor below which NaN is returned is 50 samples, the recommended
        minimum (see :data:`RECOMMENDED_MIN_WINDOW`) is 100, and ≥ 200 is
        ideal.
    dt : float
        Physical time step (used to normalise the KM coefficients).
    bin_number : int
        Number of bins per dimension for the KM estimator.
    bw : float or None
        Kernel bandwidth.  If None, estimated automatically from the data.
    center_fraction : float
        Fraction of central bins used for the linear fit of D¹ vs x (1D) or
        for the Jacobian regression (nD).  Edge bins can be noisy due to
        sparse sampling.
    detrend : bool
        Remove linear trend within window (applied per dimension for nD input).

    Returns
    -------
    float
        Continuous-time restoring rate λ (negative for stable systems,
        increases toward 0 as CSD develops).

    Notes
    -----
    Requires the ``kramersmoyal`` package::

        pip install kramersmoyal

    References
    ----------
    Morr, A., Riechers, K., Gorjão, L. R., & Boers, N. (2024).
    Anticipating critical transitions in multidimensional systems driven by
    time- and state-dependent noise. Physical Review Research, 6, 033251.
    https://doi.org/10.1103/PhysRevResearch.6.033251
    """
    x = np.asarray(x, dtype=float).copy()

    if x.ndim == 1:
        # ------------------------------------------------------------------ 1D
        x = _prep(x, detrend)
        if len(x) < 50:   # hard floor; RECOMMENDED_MIN_WINDOW['lambda_km']=100
            return np.nan
        try:
            from kramersmoyal import km  # type: ignore

            # Request the 0th power (density) explicitly and read the drift from
            # kmc[1]. This is robust across kramersmoyal versions: 0.4 auto-
            # prepends power 0 and trims it back off (so drift lands at kmc[0]),
            # whereas >=0.4.1 always keeps the density row at index 0 (drift at
            # kmc[1]). Passing 0 explicitly makes both return [density, drift,
            # diffusion] in the same order, so the indexing no longer depends on
            # the installed version.
            powers = np.array([[0], [1], [2]])
            kmc, edges = km(x.reshape(-1, 1), bins=np.array([bin_number]),
                            powers=powers, bw=bw)
            km1 = kmc[1].flatten() / dt   # drift normalised by time step
            centres = (edges[0][:-1] + edges[0][1:]) / 2.0

            n_bins = len(centres)
            i0 = int(n_bins * (1.0 - center_fraction) / 2.0)
            i1 = n_bins - i0
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

    elif x.ndim == 2:
        # ------------------------------------------------------------------ nD
        # x shape: (n_dims, n_samples)
        import itertools

        n_dims, n_samples = x.shape
        if n_samples < 50:
            return np.nan
        try:
            from kramersmoyal import km  # type: ignore

            # Detrend each dimension while keeping its mean (needed for KM
            # to locate the stable state correctly)
            if detrend:
                for i in range(n_dims):
                    center_i = float(np.mean(x[i]))
                    x[i] = remove_linear_trend(x[i]) + center_i

            # Normalise each dimension by its std (whitening) so that the
            # shared bandwidth parameter is meaningful across dimensions
            stds = np.array([np.std(x[i]) for i in range(n_dims)])
            x_norm = np.array(
                [(x[i] - np.mean(x[i])) / stds[i] + np.mean(x[i])
                 for i in range(n_dims)]
            )
            center = np.array([float(np.mean(x_norm[i])) for i in range(n_dims)])

            # Powers: zeroth moment + unit vector for each dimension
            powers = np.array(
                [[0] * n_dims]
                + [[0] * i + [1] + [0] * (n_dims - 1 - i) for i in range(n_dims)]
            )
            bins = (np.ones(n_dims) * bin_number).astype(int)
            bw_used = n_dims * 14.0 / bin_number if bw is None else bw

            kmc, edges = km(
                np.transpose(x_norm), bw=bw_used, bins=bins, powers=powers
            )
            # kmc[0] is the zeroth moment; kmc[1:] are the first-order drift
            # coefficients for each dimension
            km1 = [kmc[i + 1] / dt for i in range(n_dims)]

            # Keep only the central fraction of bins around the stable state
            for i in range(n_dims):
                center_idx = np.searchsorted(edges[i], center[i])
                half = round(bin_number * center_fraction / 2)
                left = max(0, center_idx - half)
                right = min(bin_number - 1, center_idx + half)
                min_bins = round(bin_number * center_fraction)
                left = min(min_bins, left)
                right = max(min_bins, right)
                edges[i] = edges[i][left:right]
                km1 = list(km1)
                for j in range(n_dims):
                    km1[j] = km1[j].take(indices=range(left, right), axis=i)

            # Build OLS design matrix over the grid of retained bin centres
            mesh = np.meshgrid(*edges, indexing="ij")
            idx_iter = list(
                itertools.product(*[range(len(mesh[j])) for j in range(n_dims)])
            )
            X_mat = np.array(
                [[1.0] + [mesh[i][t] for i in range(n_dims)] for t in idx_iter]
            )
            Y_mat = np.array(
                [[km1[i][t] for i in range(n_dims)] for t in idx_iter]
            )

            # OLS: B_full = (X'X)^{-1} X' Y, then drop intercept row
            XtX = X_mat.T @ X_mat
            B = np.linalg.inv(XtX) @ X_mat.T @ Y_mat
            B = B[1:, :]  # (n_dims, n_dims) Jacobian of drift

            # Dominant eigenvalue (least negative = closest to 0 = CSD signal)
            eigvals = np.real(np.linalg.eigvals(B))
            return float(np.max(eigvals))

        except ImportError:
            warnings.warn(
                "The 'kramersmoyal' package is not installed; 'lambda_km' is unavailable. "
                "Install with: pip install kramersmoyal"
            )
            return np.nan
        except Exception:
            return np.nan

    else:
        warnings.warn(
            f"estimate_lambda_km: input must be 1D or 2D (got ndim={x.ndim}); "
            "returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan


# ---------------------------------------------------------------------------
# Spectral ratio estimator (ROSA)
# ---------------------------------------------------------------------------

def estimate_rosa(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
    noise: Optional[np.ndarray] = None,
    noise_psd=None,
) -> float:
    """
    Restoring rate via the Ratio Of Spectra Approach (ROSA).

    For a first-order (OU) system driven by a stochastic forcing ξ,

        dx/dt = -λ x + ξ(t),

    the response and forcing power spectra are related by the transfer
    function of the system:

        S_xx(ω) = S_ξξ(ω) / (ω² + λ²),   ω = 2πf.

    ROSA therefore estimates λ from the **ratio** of the response spectrum to
    the forcing spectrum, R(ω) = S_xx(ω) / S_ξξ(ω) ∝ 1 / (ω² + λ²), which —
    unlike a plain fit to S_xx — does **not** assume the forcing is white.
    This is the "beyond white noise" property of the method: as long as the
    forcing spectrum is supplied (or estimated), coloured/red forcing is
    handled correctly.

    The forcing spectrum S_ξξ is taken from (in order of precedence):

    * ``noise`` — a forcing/noise **time series**.  Its periodogram is computed
      and interpolated onto the window's frequency grid.  In the sliding-window
      pipeline pass a full series aligned with ``x``; it is windowed in sync
      with ``x`` (see :data:`~csdtools.sliding_window._WINDOWED_KWARGS`), so the
      forcing may vary in time.
    * ``noise_psd`` — an explicit forcing PSD, either a callable ``S(ω)`` of the
      angular frequency ω, or a ``(omegas, values)`` pair that is interpolated
      onto the window's ω grid.  Treated as stationary (same for every window).
    * neither — a **white** forcing (``S_ξξ ≡ const``), i.e. a plain Lorentzian
      fit to S_xx.

    A free amplitude is fitted alongside λ, so an overall scale mismatch
    between a ``noise`` proxy and the true forcing does not bias λ (which is
    set by the corner frequency ω = λ of the Lorentzian).

    **CSD signature**: λ increases toward 0.

    Parameters
    ----------
    x : array-like
        Pre-processed window.
    dt : float
        Physical time step.  Frequencies are ``2π`` times the periodogram
        frequencies in per-unit-time units.
    detrend : bool
        Remove linear trend within window (applied to ``x`` and, if given, to
        the ``noise`` window before their periodograms).
    noise : array-like or None
        Forcing/noise time series (see above).  Mutually exclusive with
        ``noise_psd``.
    noise_psd : callable or (omegas, values) or None
        Forcing power spectral density as a function of **angular frequency**
        ω = 2πf.  A callable is evaluated at the window's ω grid; a
        ``(omegas, values)`` pair is linearly interpolated onto it.  Mutually
        exclusive with ``noise``.

    Returns
    -------
    float
        Continuous-time restoring rate λ (negative for stable systems).

    References
    ----------
    Clarke, J., Huntingford, C., Ritchie, P. D. L., & Cox, P. M. (2023).
    Seeking more robust early warning signals for climate tipping points:
    the ratio of spectra method (ROSA).
    Environmental Research Letters, 18(3), 035006.
    https://doi.org/10.1088/1748-9326/acbc8d
    """
    from scipy import optimize, signal

    if noise is not None and noise_psd is not None:
        raise ValueError("Pass only one of 'noise' or 'noise_psd', not both.")

    x = _prep(x, detrend)
    if len(x) < 8:
        return np.nan
    try:
        # One-sided response periodogram, excluding DC.
        freqs, S_xx = signal.periodogram(x, fs=1.0 / dt)
        freqs, S_xx = freqs[1:], S_xx[1:]
        if len(freqs) < 3:
            return np.nan
        omega = 2.0 * np.pi * freqs

        # --- Forcing spectrum S_ff on the window's frequency grid ---
        if noise is not None:
            nz = _prep(np.asarray(noise, dtype=float), detrend)
            if len(nz) < 8:
                return np.nan
            f_n, S_n = signal.periodogram(nz, fs=1.0 / dt)
            # Grids coincide when the noise window matches x in length;
            # interpolate defensively so mismatched lengths still work.
            S_ff = np.interp(freqs, f_n[1:], S_n[1:])
        elif noise_psd is not None:
            if callable(noise_psd):
                S_ff = np.asarray(noise_psd(omega), dtype=float)
            else:
                om_ref, val_ref = noise_psd
                S_ff = np.interp(
                    omega, np.asarray(om_ref, dtype=float),
                    np.asarray(val_ref, dtype=float),
                )
            if S_ff.shape != omega.shape:
                return np.nan
        else:
            S_ff = np.ones_like(S_xx)   # white forcing

        # Response-to-forcing spectral ratio, keeping usable points only.
        valid = (np.isfinite(S_ff) & (S_ff > 0.0)
                 & np.isfinite(S_xx) & (S_xx > 0.0))
        if valid.sum() < 3:
            return np.nan
        R = S_xx[valid] / S_ff[valid]
        w = omega[valid]

        # Fit R(ω) = A / (ω² + λ²) in log space (robust to periodogram noise).
        def log_model(w_, log_A, lam):
            return log_A - np.log(w_ ** 2 + lam ** 2)

        log_A0 = float(np.log(np.median(R)) + np.log(np.median(w) ** 2 + 1.0))
        popt, _ = optimize.curve_fit(
            log_model, w, np.log(R),
            p0=[log_A0, 0.1 / dt],
            bounds=([-np.inf, 0.0], [np.inf, np.inf]),
            maxfev=10000,
        )
        lam = float(popt[1])
        if not np.isfinite(lam):
            return np.nan
        return -lam
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# DMD-based indicator
# ---------------------------------------------------------------------------

def estimate_lambda_dmd(
    x: np.ndarray,
    dt: float = 1.0,
    embedding_dim: int = 1,
    tau: int = 1,
    dmd_rank: int = -1,
    detrend: bool = True,
) -> float:
    """
    Dominant DMD eigenvalue magnitude via Dynamic Mode Decomposition.

    Fits a linear operator A such that A·X ≈ Y (where columns of X and Y are
    consecutive state snapshots) using DMD and returns the magnitude of the
    dominant eigenvalue.  For a stable system near a fixed point the dominant
    eigenvalue has magnitude < 1; as the system approaches a bifurcation the
    magnitude increases toward 1.

    Both 1D and nD inputs are handled uniformly: a 1D signal is treated as a
    single-row state.  For each dimension, time-delay (Hankel) embedding with
    lag ``tau`` and depth ``embedding_dim`` is applied, and the resulting rows
    are stacked to form the state matrix.  With ``embedding_dim=1`` the raw
    snapshots are used directly.  For 1D signals a higher ``embedding_dim``
    (e.g. 3–10) lifts the scalar series into a richer state space and usually
    gives more stable eigenvalue estimates.

    **CSD signature**: dominant eigenvalue magnitude increases toward 1.

    Parameters
    ----------
    x : array-like, shape (n,) or (n_dims, n_samples)
        Time series window.  1D arrays are treated as a single-row state
        (n_dims=1).  For 2D arrays each row is one variable and each column
        is one time step; n_dims and n_samples are inferred from the shape.
    dt : float
        Time step (unused; present for API consistency).
    embedding_dim : int
        Number of time-delay lags used to construct the state vector.
        ``embedding_dim=1`` uses raw snapshots without additional embedding.
    tau : int
        Lag in samples between successive embedded copies of the state.
    dmd_rank : int
        SVD rank truncation for DMD.  ``-1`` keeps all singular values;
        smaller values filter noise at the cost of information.
    detrend : bool
        Remove a linear trend within the window before fitting DMD
        (applied per row for nD input).

    Returns
    -------
    float
        Magnitude of the dominant DMD eigenvalue.  In (0, 1) for stable
        systems; increases toward 1 as resilience is lost.

    Notes
    -----
    Unlike the other ``lambda_*`` estimators, this returns the **magnitude of
    the dominant discrete-time eigenvalue** — a dimensionless number in
    ``(0, 1)`` for a stable system that grows toward 1 — **not** a
    continuous-time restoring rate.  It is therefore not on the same scale as
    ``lambda_glsar`` etc., is not converted via ``log(·)/Δt``, and the ``dt``
    argument does not affect it.  Only the sign/direction of its trend
    (increasing = destabilising) is comparable across methods.

    Requires the ``pydmd`` package::

        pip install pydmd

    References
    ----------
    Smith, T., Morr, A., Bookhagen, B., & Boers, N. (2026).
    Predicting instabilities in transient landforms and interconnected ecosystems.
    Nature Communications, 17(1), 1316.
    https://doi.org/10.1038/s41467-026-68944-w
    """
    x = np.asarray(x, dtype=float).copy()

    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim != 2:
        warnings.warn(
            f"estimate_lambda_dmd: input must be 1D or 2D (got ndim={x.ndim}); "
            "returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan

    n_dims, n_samples = x.shape

    if detrend:
        for i in range(n_dims):
            x[i] = remove_linear_trend(x[i])

    # Number of columns in the Hankel-embedded matrix
    n_cols = n_samples - (embedding_dim - 1) * tau
    if n_cols < 3:
        return np.nan

    try:
        from pydmd import DMD  # type: ignore

        # Build time-delay Hankel embedding and stack across all dimensions.
        # For dimension i and lag k, row k of H_i is x[i, k*tau : k*tau + n_cols].
        # Stacking all (n_dims * embedding_dim) rows gives the full state matrix.
        rows = []
        for i in range(n_dims):
            for k in range(embedding_dim):
                rows.append(x[i, k * tau : k * tau + n_cols])
        H = np.vstack(rows)  # (n_dims * embedding_dim, n_cols)

        dmd = DMD(svd_rank=dmd_rank)
        dmd.fit(X=H[:, :-1], Y=H[:, 1:])

        magnitudes = np.abs(dmd.eigs)
        if len(magnitudes) == 0:
            return np.nan
        return float(np.max(magnitudes))

    except ImportError:
        warnings.warn(
            "The 'pydmd' package is not installed; 'lambda_dmd' is unavailable. "
            "Install with: pip install pydmd",
            UserWarning,
        )
        return np.nan
    except Exception:
        return np.nan


def estimate_lambda_dmd_floquet(
    x: np.ndarray,
    dt: float = 1.0,
    period: int = 12,
    embedding_dim: int = 1,
    tau: int = 1,
    dmd_rank: int = -1,
    mean_center: bool = True,
    standardize: bool = True,
    detrend: bool = True,
) -> float:
    """
    Dominant Floquet-multiplier magnitude via spatiotemporal DMD.

    Seasonal / spatially-extended counterpart of :func:`estimate_lambda_dmd`.
    Instead of a one-step operator it fits the *period-T monodromy operator* M
    that maps the (embedded) system state at time t to its state one full
    period later, t + ``period``:

        M · s(t) ≈ s(t + T)

    The eigenvalues of M are the **Floquet multipliers**; the dominant one has
    magnitude < 1 for a stable limit cycle and crosses 1 as resilience is lost.
    Fitting the period-T map (rather than the one-step map) makes the estimator
    appropriate for strongly seasonal / periodically-forced data, where a
    one-step DMD would be dominated by the seasonal cycle itself.

    The state at each time step is the full (multivariate) snapshot — for
    spatial data, the flattened field of all cells — optionally lifted by
    time-delay embedding of depth ``embedding_dim`` and lag ``tau``.  Snapshots
    are mean-centred and/or standardised per feature before the fit (SVD-based
    DMD cannot handle differing offsets/scales across features).

    Both 1D and nD inputs are accepted: a 1D signal is treated as a single
    feature.  Input of shape ``(n_features, n_samples)`` has one row per
    variable / spatial cell and one column per time step.

    **CSD signature**: dominant Floquet-multiplier magnitude increases toward 1.

    Parameters
    ----------
    x : array-like, shape (n,) or (n_features, n_samples)
        Time series window.  For nD input each row is one variable / spatial
        cell and each column one time step.
    dt : float
        Time step (unused; present for API consistency).
    period : int
        Fundamental period T in samples (e.g. 12 for monthly data with an
        annual cycle, 365 for daily data).  The DMD operator maps each state
        to the state ``period`` samples later.
    embedding_dim : int
        Depth of the time-delay embedding of the (spatial) snapshots.
        ``embedding_dim=1`` uses the raw snapshots.
    tau : int
        Lag in samples between successive embedded copies of the state.
    dmd_rank : int
        SVD rank truncation for DMD.  ``-1`` keeps all singular values;
        smaller values filter noise (recommended for high-dimensional fields).
    mean_center : bool
        Subtract each feature's mean (over the snapshot columns) before the
        fit.  Recommended for collections of series with different means.
    standardize : bool
        Divide each feature by its standard deviation before the fit.
        Recommended for collections of series with different scales.
    detrend : bool
        Remove a linear trend within the window before embedding (applied per
        row for nD input).  Smith et al. (2026) instead rely on
        ``standardize=True``; ``detrend`` is provided for consistency with the
        other estimators and may be disabled.

    Returns
    -------
    float
        Magnitude of the dominant Floquet multiplier.  In (0, 1) for a stable
        limit cycle; increases toward 1 as resilience is lost.

    Notes
    -----
    Like :func:`estimate_lambda_dmd`, this returns an eigenvalue **magnitude**,
    not a continuous-time rate, and ``dt`` has no effect.

    The window must be long enough to contain the period shift and the
    embedding lags — at least ``period + (embedding_dim - 1) * tau + 2``
    samples — and in practice several periods (``≈ (num_cycles + 1) * period``)
    for a stable fit; the effective number of cycles is set implicitly by the
    window size.

    Requires the ``pydmd`` package::

        pip install pydmd

    References
    ----------
    Smith, T., Morr, A., Bookhagen, B., & Boers, N. (2026).
    Predicting instabilities in transient landforms and interconnected
    ecosystems. Nature Communications, 17(1), 1316.
    https://doi.org/10.1038/s41467-026-68944-w
    """
    x = np.asarray(x, dtype=float).copy()

    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim != 2:
        warnings.warn(
            f"estimate_lambda_dmd_floquet: input must be 1D or 2D (got "
            f"ndim={x.ndim}); returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan

    n_features, n_samples = x.shape
    T = int(period)
    idx_offset = (embedding_dim - 1) * tau

    if T < 1:
        return np.nan
    if detrend:
        for i in range(n_features):
            x[i] = remove_linear_trend(x[i])

    # Number of X/Y snapshot pairs that fit in the window. The youngest map of
    # the oldest X state sits at idx_offset; its Y counterpart is T steps ahead,
    # so the last usable youngest-Y index is n_samples - 1.
    num_snapshots = n_samples - T - idx_offset
    if num_snapshots < 2:
        return np.nan

    try:
        from pydmd import DMD  # type: ignore

        n_embedded = n_features * embedding_dim
        X = np.empty((n_embedded, num_snapshots))
        Y = np.empty((n_embedded, num_snapshots))
        for j in range(num_snapshots):
            xi = idx_offset + j
            yi = xi + T
            # Embedded state = youngest map first, then lagged copies.
            X[:, j] = np.concatenate(
                [x[:, xi - k * tau] for k in range(embedding_dim)]
            )
            Y[:, j] = np.concatenate(
                [x[:, yi - k * tau] for k in range(embedding_dim)]
            )

        # Per-feature mean-centre / standardise (stats from X, applied to both).
        if mean_center or standardize:
            mean_X = np.nanmean(X, axis=1, keepdims=True)
        if standardize:
            tmp = X - mean_X if mean_center else X
            std_X = np.nanstd(tmp, axis=1, keepdims=True)
            std_X[std_X == 0] = 1.0
        if mean_center:
            X = X - mean_X
            Y = Y - mean_X
        if standardize:
            X = X / std_X
            Y = Y / std_X

        # SVD-based DMD cannot handle NaN/Inf; fill defensively.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        dmd = DMD(svd_rank=dmd_rank)
        dmd.fit(X=X, Y=Y)

        magnitudes = np.abs(dmd.eigs)
        if len(magnitudes) == 0:
            return np.nan
        return float(np.max(magnitudes))

    except ImportError:
        warnings.warn(
            "The 'pydmd' package is not installed; 'lambda_dmd_floquet' is "
            "unavailable. Install with: pip install pydmd",
            UserWarning,
        )
        return np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Multivariate spatial correlation indicator
# ---------------------------------------------------------------------------

def estimate_spatial_corr(
    x: np.ndarray,
    dt: float = 1.0,
    detrend: bool = True,
    distance_matrix: Optional[np.ndarray] = None,
    min_dist: float = 0.0,
    max_dist: Optional[float] = None,
    center_idx: Optional[int] = None,
) -> float:
    """
    Mean spatial correlation (CSD indicator).

    For a spatially extended system approaching a critical transition the
    correlation length diverges: neighboring locations become increasingly
    correlated. The *mean spatial correlation* is defined as the average 
    Pearson correlation between the center grid cell (center_idx) and
    all other cells within a user-defined "neighborhood" 
    (min_dist < d_j <= max_dist).

    **CSD signature**: spatial correlation increases.

    Parameters
    ----------
    x : np.ndarray, shape (n_cells, n_samples)
        Multi-cell time series window.  Each row is one spatial cell.
    dt : float
        Time step (unused; present for API consistency).
    detrend : bool
        Remove a linear trend within each cell's window before computing
        correlations.
    distance_matrix : np.ndarray, shape (n_cells,)
        Symmetric pairwise distance matrix (e.g. in km).
        If None, all grid cells are considered neighbors of the center cell.
    min_dist : float
        Exclude cell from neighbors if closer than this distance.
    max_dist : float or None
        Exclude cell from neighbors farther than this distance.
    center_idx : int or None
        Index of the central grid cell.  Defaults to ``n_cells // 2`` if
        not given.

    Returns
    -------
    float
        Mean spatial Pearson correlation.

    Notes
    -----
    Pass ``distance_matrix=D`` via ``**method_kwargs`` to
    :func:`~csdtools.analyze_timeseries` or
    :func:`~csdtools.sliding_window.sliding_window_analysis`.

    References
    ----------
    Blaschke, L. L., et al. (2024). Earth's Future, 12.
    https://doi.org/10.1029/2023EF004040
    """

    x = np.asarray(x, dtype=float).copy()
    if detrend:
        for i in range(x.shape[0]):
            x[i] = remove_linear_trend(x[i])

    from .spatial import spatial_correlation
    return spatial_correlation(
        x, distance_matrix, min_dist=min_dist, max_dist=max_dist, center_idx=center_idx
    )


# ---------------------------------------------------------------------------
# Registry and unified interface
# ---------------------------------------------------------------------------

#: Maps method name → (function, bool: uses_dt).
#: The bool is informational; all functions accept dt for API uniformity.
INDICATOR_REGISTRY: dict = {
    "ac1":           (estimate_ac1,           False),
    "variance":      (estimate_variance,      False),
    "skewness":      (estimate_skewness,      False),
    "lambda_ac1":    (estimate_lambda_ac1,    True),
    "lambda_var":    (estimate_lambda_var,    True),
    "lambda_glsar":  (estimate_lambda_glsar,  True),
    "lambda_ols":    (estimate_lambda_ols,    True),
    "lambda_acs":    (estimate_lambda_acs,    True),
    "lambda_psd":    (estimate_lambda_psd,    True),
    "lambda_km":     (estimate_lambda_km,     True),
    "rosa":          (estimate_rosa,          True),
    "lambda_dmd":    (estimate_lambda_dmd,    False),
    "lambda_dmd_floquet": (estimate_lambda_dmd_floquet, False),
    "spatial_corr":  (estimate_spatial_corr,  False),
}

#: Methods that accept a 2D ``(n_dims, n_samples)`` input array.
#: All other methods expect a 1D array.  Add new multivariate indicators here.
ND_METHODS: set = {"lambda_km", "lambda_dmd", "lambda_dmd_floquet", "spatial_corr"}

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

    # NaN removal is only applicable to 1D input; for 2D arrays (e.g. nD KM)
    # the estimator handles its own data validation.
    if x.ndim == 1:
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
