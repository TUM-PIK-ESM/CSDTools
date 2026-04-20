"""
Spatial correlation analysis for CSD.

In spatially extended systems approaching a critical transition, the
correlation length diverges: nearby locations become increasingly correlated.
This makes spatially averaged correlation a useful CSD indicator that is
complementary to the single-point methods in indicators.py.

The key quantity computed here is the *distance-weighted mean spatial
correlation*: the average Pearson correlation between all pairs of spatial
cells, weighted by the inverse of their pairwise distance.  This emphasises
correlation between distant cells, which is expected to grow most during CSD.

References
----------
Blaschke et al. (2024). Spatial Correlation Increase in Single-Sensor
Satellite Data Reveals Loss of Amazon Rainforest Resilience.

Rietkerk et al. (2021). Evasion of tipping in complex systems through
spatial heterogeneity. *Science*.
"""

import numpy as np
from typing import Optional, Tuple


def spatial_correlation(
    data: np.ndarray,
    distance_matrix: np.ndarray,
    min_dist: float = 0.0,
    max_dist: Optional[float] = None,
) -> float:
    """
    Distance-weighted mean spatial correlation between cells.

    For each pair of cells (i, j) at distance d_{ij}, computes the Pearson
    correlation between their time series, then averages with weight
    w_{ij} = 1/d_{ij}.  Pairs closer than ``min_dist`` are excluded to
    avoid trivial autocorrelation at zero lag distance.

    Parameters
    ----------
    data : np.ndarray, shape (n_cells, n_time)
        Time series for each spatial cell along the first axis.
    distance_matrix : np.ndarray, shape (n_cells, n_cells)
        Symmetric pairwise distance matrix (e.g., in km).
    min_dist : float
        Exclude pairs with distance < min_dist.
    max_dist : float or None
        Exclude pairs with distance > max_dist (focus on local correlation).

    Returns
    -------
    float
        Distance-weighted mean spatial correlation.

    Notes
    -----
    For a single sliding window, pass the window slice of your data array
    (shape: n_cells × window_size) to this function.
    """
    n_cells = data.shape[0]
    # Pearson correlation matrix across all cells
    corr = np.corrcoef(data)   # shape (n_cells, n_cells)

    total_w = 0.0
    weighted_sum = 0.0
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            d = distance_matrix[i, j]
            if d <= min_dist:
                continue
            if max_dist is not None and d > max_dist:
                continue
            w = 1.0 / (d + 1e-12)
            weighted_sum += w * corr[i, j]
            total_w += w

    if total_w == 0.0:
        return np.nan
    return float(weighted_sum / total_w)


def sliding_spatial_correlation(
    data: np.ndarray,
    distance_matrix: np.ndarray,
    window_size: int,
    step: int = 1,
    min_dist: float = 0.0,
    max_dist: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sliding-window spatial correlation.

    Applies :func:`spatial_correlation` inside a sliding window along the
    time axis.  Returns the spatial correlation timeseries and the
    corresponding window midpoint indices.

    Parameters
    ----------
    data : np.ndarray, shape (n_cells, n_time)
        Multi-cell time series.
    distance_matrix : np.ndarray, shape (n_cells, n_cells)
        Pairwise cell distances.
    window_size : int
        Number of time steps per window.
    step : int
        Step between consecutive windows.
    min_dist : float
        Minimum distance threshold for cell pairs.
    max_dist : float or None
        Maximum distance threshold for cell pairs.

    Returns
    -------
    spc : np.ndarray
        Spatial correlation value at each window position.
    times : np.ndarray
        Midpoint time index of each window.

    Examples
    --------
    >>> spc, t = sliding_spatial_correlation(
    ...     data, dist_matrix, window_size=120, step=1
    ... )
    """
    n_time = data.shape[1]
    starts = np.arange(0, n_time - window_size + 1, step)
    spc = np.full(len(starts), np.nan)
    times = starts + window_size // 2

    for k, i in enumerate(starts):
        window = data[:, i : i + window_size]
        spc[k] = spatial_correlation(
            window, distance_matrix, min_dist=min_dist, max_dist=max_dist
        )

    return spc, times.astype(float)


def compute_distance_matrix(
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Compute a great-circle distance matrix between geographic locations.

    Uses the WGS-84 ellipsoid model (via pyproj) for accurate distances.

    Parameters
    ----------
    lats : np.ndarray, shape (n,)
        Latitudes in decimal degrees.
    lons : np.ndarray, shape (n,)
        Longitudes in decimal degrees.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric matrix of pairwise great-circle distances in kilometres.

    Notes
    -----
    Requires pyproj::

        pip install pyproj

    Examples
    --------
    >>> D = compute_distance_matrix(lats, lons)
    >>> D.shape
    (n_cells, n_cells)
    """
    try:
        from pyproj import Geod  # type: ignore
    except ImportError:
        raise ImportError(
            "pyproj is required for compute_distance_matrix. "
            "Install with: pip install pyproj"
        )

    geod = Geod(ellps="WGS84")
    n = len(lats)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _, _, d_m = geod.inv(lons[i], lats[i], lons[j], lats[j])
            dist[i, j] = d_m / 1000.0   # metres → kilometres
            dist[j, i] = dist[i, j]

    return dist
