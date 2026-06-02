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
Blaschke, L. L., et al. (2024). Earth's Future, 12.
https://doi.org/10.1029/2023EF004040

Rietkerk, M., et al. (2021). Science, 374(6564), eabj0359.
https://doi.org/10.1126/science.abj0359


-------
checked by Lana on 2026-05-19
"""

import numpy as np
from typing import Optional


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
    # Pearson correlation matrix: shape (n_cells, n_cells)
    corr = np.corrcoef(data)
    dist = np.asarray(distance_matrix, dtype=float)

    # Build mask: pairs strictly beyond min_dist, within max_dist, off-diagonal
    mask = dist > min_dist
    if max_dist is not None:
        mask &= dist <= max_dist
    np.fill_diagonal(mask, False)

    if not mask.any():
        return np.nan

    # Distance-weighted mean: w_ij = 1/d_ij
    weights = np.divide(1.0, dist, where=mask, out=np.zeros_like(dist))
    total_weight = weights.sum()
    if total_weight == 0.0:
        return np.nan
    return float(np.sum(corr * weights) / total_weight)
    



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
