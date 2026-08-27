"""
Spatial correlation analysis for CSD.

In spatially extended systems approaching a critical transition, the
correlation length diverges: nearby locations become increasingly correlated.
This makes spatially averaged correlation a useful CSD indicator that is
complementary to the single-point methods in indicators.py.

The key quantity computed here is the *mean spatial correlation*: 
the average Pearson correlation between the center grid cell (center_idx)
and all other cells within a user-defined "neighborhood" (min_dist < d_j <= max_dist).

References
----------
Blaschke et al. (2024). Spatial Correlation Increase in Single-Sensor
Satellite Data Reveals Loss of Amazon Rainforest Resilience.

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
    center_idx: Optional[int] = None,
) -> float:
    """
    Mean spatial correlation between cells.

    For the central grid cell (i = center_idx, defaulting to n_cells//2),
    calculated the Pearson correlation between its time series and all other
    grid cells, then averages over all neighbors for which
    min_dist < d_j <= max_dist.
    The distance matrix contains the distances d_j of all grid cells to cell i (including j=i).

    Parameters
    ----------
    data : np.ndarray, shape (n_cells, n_time)
        Time series for each spatial cell along the first axis.
    distance_matrix : np.ndarray, shape (n_cells)
        Distance matrix (e.g., in km).
        All distances are set to 1 except for the j==i entry, which is 0.
    min_dist : float
        Exclude cells from neighbors with distance < min_dist.
    max_dist : float or None
        Exclude cells from neighbors with distance > max_dist (focus on local correlation).
    center_idx : int or None
        Index of the central grid cell.  Defaults to ``n_cells // 2`` if
        not given.

    Returns
    -------
    float
            Mean spatial correlation of cell i with its neighbors

    Notes
    -----
    For a single sliding window, pass the window slice of your data array
    (shape: n_cells × window_size) to this function.
    """
    n_cells = data.shape[0]
    i = center_idx if center_idx is not None else n_cells // 2  # central cell index
    if not (-n_cells <= i < n_cells):
        raise ValueError(
            f"center_idx={i} is out of bounds for {n_cells} grid cells "
            f"(data has shape {data.shape})."
        )
    # create mask from distance matrix
    if distance_matrix is None:
        mask = np.ones(n_cells, dtype=bool)
        mask[i] = False  # exclude self-pair
    elif distance_matrix.shape != (n_cells,):
        raise ValueError(
            f"Distance matrix shape {distance_matrix.shape} does not match "
            f"data shape {data.shape}."
        )
    else:
        dist = np.asarray(distance_matrix, dtype=float)     
        mask = dist > min_dist                                                                                                          
        if max_dist is not None:
            mask &= (dist <= max_dist)

    # Pearson correlation matrix across all cells
    corr = np.corrcoef(data)[i, :]   # shape (1, n_cells)
    return np.nanmean(corr, where=mask, axis=0)


def spatial_correlation_2d(
    data: np.ndarray,
    distance_matrix: np.ndarray,
    min_dist: float = 0.0,
    max_dist: Optional[float] = None,
) -> float:
    """
    Mean spatial correlation between cells.

    For each pair of cells (i, j) at distance d_{ij}, computes the Pearson
    correlation between their time series, then averages for each cell the correlation
    with its neighbors of min_dist < d_{ij} <= max_dist.

    Parameters
    ----------
    data : np.ndarray, shape (n_cells, n_time)
        Time series for each spatial cell along the first axis.
    distance_matrix : np.ndarray, shape (n_cells, n_cells)
        Symmetric pairwise distance matrix (e.g., in km).
        If None, all pairs are included (except self-pairs).
    min_dist : float
        Exclude pairs with distance < min_dist.
    max_dist : float or None
        Exclude pairs with distance > max_dist (focus on local correlation).

    Returns
    -------
    np.ndarray, shape (n_cells,):
            Mean spatial correlation.

    Notes
    -----
    For a single sliding window, pass the window slice of your data array
    (shape: n_cells × window_size) to this function.
    """
    n_cells = data.shape[0]
    # create mask from distance matrix
    if distance_matrix is None:
        mask = ~np.eye(n_cells, dtype=bool)
    elif distance_matrix.shape != (n_cells, n_cells):
        raise ValueError(
            f"Distance matrix shape {distance_matrix.shape} does not match "
            f"data shape {data.shape}."
        )
    else:
        dist = np.asarray(distance_matrix, dtype=float)     
        mask = dist > min_dist                                                                                                          
        if max_dist is not None:
            mask &= (dist <= max_dist)
        np.fill_diagonal(mask, False)  

    # Pearson correlation matrix across all cells
    corr = np.corrcoef(data, rowvar=True)   # shape (n_cells, n_cells)
    return np.nanmean(corr, where=mask, axis=1)
    



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
