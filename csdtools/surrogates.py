"""
Surrogate time series generation for significance testing.

The standard null hypothesis in CSD analysis is that the observed trend in
an indicator arises by chance from a *stationary* process — i.e., one whose
statistical properties do not change over time.

Fourier phase surrogates are the most widely used tool for testing this
hypothesis.  Each surrogate has exactly the same power spectrum (and
therefore the same linear autocorrelation function) as the original data,
but with the phase information randomised.  Any monotonic trend in a
surrogate indicator therefore represents the trend that a stationary process
with the same spectral shape could produce by chance.

References
----------
Theiler et al. (1992). Testing for nonlinearity in time series: the method
of surrogate data. *Physica D*, 58, 77-94.

Dakos et al. (2008). Slowing down as an early warning signal for abrupt
climate change. *PNAS*, 105, 14308-14312.
"""

import numpy as np


def fourier_surrogates(x: np.ndarray, n: int, seed: int = None) -> np.ndarray:
    """
    Generate n Fourier phase surrogates of a time series.

    Each surrogate is produced by:

    1. Computing the real FFT of x.
    2. Preserving the amplitudes |F(k)| exactly.
    3. Assigning independent uniform random phases φ_k ∈ [0, 2π).
    4. Inverting the FFT to obtain a real-valued surrogate.

    The resulting surrogates are guaranteed to have the same power spectrum
    as x (up to the randomness of the phases), so any trend observed in the
    original indicator but absent in the surrogate distribution is evidence
    against the null hypothesis of stationarity.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        1D time series to generate surrogates from.
    n : int
        Number of surrogates to generate.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, T)
        Array of surrogate time series.  Row k is the k-th surrogate.

    Examples
    --------
    >>> surrogates = fourier_surrogates(x, n=1000)
    >>> surrogates.shape
    (1000, 500)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    T = len(x)
    spec = np.fft.rfft(x)
    amplitudes = np.abs(spec)
    n_freqs = len(spec)

    surrogates = np.empty((n, T))
    for k in range(n):
        phases = rng.uniform(0.0, 2.0 * np.pi, n_freqs)
        new_spec = amplitudes * np.exp(1j * phases)
        surrogates[k] = np.fft.irfft(new_spec, n=T)

    return surrogates
