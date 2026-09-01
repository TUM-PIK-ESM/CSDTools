"""
Multivariate CSD analysis example.

Demonstrates how to apply all three nD-capable indicators to a 2D synthetic
time series with Critical Slowing Down:

    lambda_km    — non-parametric Kramers-Moyal drift Jacobian
    lambda_dmd   — Dynamic Mode Decomposition dominant eigenvalue magnitude
    spatial_corr — Mean spatial correlation

The synthetic data is a 2D coupled AR(1) process whose dominant eigenvalue
increases linearly from 0.3 to 0.95, mimicking CSD as a 2D system approaches
a tipping point.

Input array shape throughout: (n_dims=2, n_samples) — rows are variables,
columns are time steps.

Run from the repository root::

    python -m csdtools.examples.multivariate_analysis
"""

import numpy as np
import matplotlib.pyplot as plt

from csdtools import analyze_timeseries
from csdtools.plotting import plot_indicator_on_ax

# ──────────────────────────────────────────────────────────────────────────────
# 1. Generate synthetic 2D data with CSD
# ──────────────────────────────────────────────────────────────────────────────
# Coupled 2D AR(1) process:
#
#   x(t+1) = A(t) @ x(t) + σ ε(t),   ε ~ N(0, I)
#
# A(t) = [[φ(t), 0.1],   with φ(t) increasing from 0.3 → 0.95.
#          [0.1,  0.5]]
#
# The dominant eigenvalue of A(t) follows φ(t) closely, so all three nD
# indicators should detect an increasing trend.

rng   = np.random.default_rng(42)
T     = 1500
sigma = 0.3

phi = np.linspace(0.3, 0.95, T)
x   = np.zeros((2, T))            # shape: (n_dims=2, n_samples=1500)

for t in range(1, T):
    A = np.array([[phi[t], 0.1],
                  [0.1,   0.5]])
    x[:, t] = A @ x[:, t - 1] + sigma * rng.standard_normal(2)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Analysis parameters
# ──────────────────────────────────────────────────────────────────────────────
WINDOW    = 300
DETREND_W = 600   # running-mean bandwidth; rule of thumb: ≥ 2 × window_size

# For spatial_corr: distances from the center cell (CENTER, defaulting to
# n_cells // 2 = 1) to every cell, with 0 at the center's own index.
# Both variables are treated as spatial locations 1 km apart.
CENTER = 1
D = np.array([1.0, 0.0])

# ──────────────────────────────────────────────────────────────────────────────
# 3. Run all three nD indicators
# ──────────────────────────────────────────────────────────────────────────────
common = dict(
    window_size=WINDOW,
    detrend="running_mean",
    detrend_window=DETREND_W,
    local_detrend=True,
)

print("Running multivariate CSD analysis …\n")
print(f"{'Method':<15} {'Kendall τ':>10} {'p (Kendall)':>12}")
print("-" * 40)

r_km = analyze_timeseries(x, method="lambda_km",  **common)
r_dmd = analyze_timeseries(x, method="lambda_dmd", embedding_dim=3, **common)
r_sc  = analyze_timeseries(
    x, method="spatial_corr", distance_matrix=D, center_idx=CENTER, **common
)

results = {"lambda_km": r_km, "lambda_dmd": r_dmd, "spatial_corr": r_sc}

for name, r in results.items():
    print(f"{name:<15} {r.trend_statistic:>+10.3f} {r.trend_p_value:>12.4f}")

print("-" * 40)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Plot: 2D time series + all three indicators
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    4, 1, figsize=(11, 10), sharex=True,
    gridspec_kw={"height_ratios": [2, 1.2, 1.2, 1.2]},
)
fig.suptitle(
    "Multivariate CSD — 2D coupled AR(1) with φ: 0.3 → 0.95",
    fontsize=11, fontweight="semibold",
)

# --- Time series panel ---
# Center cell (CENTER) highlighted; its neighbor plotted in grey.
t_idx = np.arange(T)
for i, d in enumerate(D):
    if i != CENTER:
        axes[0].plot(t_idx, x[i], color="grey", lw=0.7,
                 label=f"$x_{i}(t)$ (neighbor)", alpha=0.7)
axes[0].plot(t_idx, x[CENTER], color="#1565C0", lw=0.9,
             label=f"$x_{CENTER}(t)$ (center)")
axes[0].set_ylabel("State", fontsize=9)
axes[0].legend(fontsize=8, loc="upper left")
axes[0].tick_params(labelbottom=False)

# --- Indicator panels ---
indicator_cfg = [
    (r_km,  "#795548", "λ  (Kramers-Moyal)"),
    (r_dmd, "#546E7A", "DMD eigenvalue\nmagnitude"),
    (r_sc,  "#00838F", "Spatial\ncorrelation"),
]

for ax, (r, color, ylabel) in zip(axes[1:], indicator_cfg):
    plot_indicator_on_ax(
        ax, r,
        show_trend=True,
        show_significance=True,
        color=color,
        stats_loc="upper left",
    )
    ax.set_ylabel(ylabel, fontsize=8)

axes[-1].set_xlabel("Sample index", fontsize=9)

plt.tight_layout()
fig.savefig("csd_multivariate.png", dpi=150, bbox_inches="tight")
print("\nFigure saved to csd_multivariate.png")

plt.show()
