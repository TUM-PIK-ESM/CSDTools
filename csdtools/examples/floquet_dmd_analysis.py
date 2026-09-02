"""
Floquet DMD (``lambda_dmd_floquet``) example — seasonal / periodically forced data.

``lambda_dmd_floquet`` fits the *period-T monodromy operator* that maps the
(embedded) system state at time t to its state one full period later, t + T.
Its dominant eigenvalue is the leading **Floquet multiplier**: its magnitude is
< 1 for a stable limit cycle and crosses 1 as the cycle loses stability.  This
is the natural CSD indicator for strongly seasonal / periodically forced
systems, where a one-step DMD (``lambda_dmd``) would be dominated by the
seasonal cycle itself.

Synthetic data
--------------
A textbook Floquet setting: a **parametrically forced** (Mathieu-type) linear
field.  Each of ``n_cells`` spatial cells follows a scalar map whose coefficient
is periodic with period ``P`` (the "seasonal forcing"),

    x_i(t+1) = a(t) · x_i(t) + σ ε_i(t),      a(t) = g(t) · (1 + m·cos(2π t / P)),

with independent noise per cell.  The *monodromy multiplier* over one period is
ρ(t) ≈ ∏ a ≈ g(t)^P, and we let it ramp from 0.30 → 0.95 — i.e. the periodic
orbit gradually loses stability.  ``lambda_dmd_floquet`` should recover this
ramp (Floquet magnitude → 1), while a stationary control (constant ρ = 0.30)
should show no trend.

Note there is *no* deterministic seasonal signal added to the observable: a
purely deterministic cycle would contribute a trivial Floquet multiplier of 1
that masks the signal.  Here the periodicity lives in the *dynamics* (the
coefficient a(t)), which is exactly what Floquet analysis is designed for.

Input array shape throughout: (n_cells, n_samples) — rows are cells, columns
are time steps.

Run from the repository root::

    python -m csdtools.examples.floquet_dmd_analysis
"""

import numpy as np
import matplotlib.pyplot as plt

from csdtools import analyze_timeseries
from csdtools.plotting import plot_indicator_on_ax

# ──────────────────────────────────────────────────────────────────────────────
# 1. Generate a parametrically forced field with a ramping Floquet multiplier
# ──────────────────────────────────────────────────────────────────────────────
rng     = np.random.default_rng(7)
P       = 12        # forcing period (e.g. 12 months → annual cycle)
N       = 1800      # samples
N_CELLS = 6         # spatial cells sharing the same (forced) dynamics
SIGMA   = 0.3       # per-step noise amplitude
M       = 0.15      # parametric-forcing amplitude


def parametric_field(rho_series, seed):
    """Field whose per-period monodromy multiplier follows ``rho_series``."""
    r = np.random.default_rng(seed)
    t = np.arange(N)
    g = rho_series ** (1.0 / P)                 # per-step geometric coefficient
    a = g * (1.0 + M * np.cos(2.0 * np.pi * t / P))   # periodic (parametric) forcing
    x = np.zeros((N_CELLS, N))
    for k in range(1, N):
        x[:, k] = a[k] * x[:, k - 1] + SIGMA * r.standard_normal(N_CELLS)
    return x


rho_csd  = np.linspace(0.30, 0.95, N)           # ramps toward 1 → CSD
rho_ctrl = np.full(N, 0.30)                      # constant → stationary control

x_csd  = parametric_field(rho_csd,  seed=7)
x_ctrl = parametric_field(rho_ctrl, seed=8)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Analysis parameters
# ──────────────────────────────────────────────────────────────────────────────
# The window must span several periods for a stable monodromy fit; 120 samples
# is ten forcing periods.  dmd_rank=2 keeps the leading coherent mode(s) and
# filters the noisy tail of the spectrum.  Global/local linear detrending is
# switched off: the dynamics are zero-mean with no trend, and per-feature
# standardisation inside the estimator handles normalisation.
common = dict(
    method="lambda_dmd_floquet",
    window_size=120,
    step=6,
    period=P,
    dmd_rank=2,
    standardize=True,
    detrend=None,          # no global preprocessing
    local_detrend=False,   # no per-window linear detrend (see note above)
)

print("Running Floquet DMD analysis …\n")
r_csd  = analyze_timeseries(x_csd,  **common)
r_ctrl = analyze_timeseries(x_ctrl, **common)

print(f"{'Series':<22} {'Kendall τ':>10} {'p (Kendall)':>13}")
print("-" * 47)
for label, result in (
    ("CSD (ρ: 0.30→0.95)", r_csd),
    ("Control (ρ=0.30)", r_ctrl),
):
    trend = result.trends["kendall"]
    print(f"{label:<22} {trend.statistic:>+10.3f} {trend.p_value:>13.2e}")
print("-" * 47)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Plot: field + Floquet indicator (with ground-truth ρ) + control
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 1, figsize=(11, 8), sharex=True,
    gridspec_kw={"height_ratios": [1.6, 1.3, 1.3]},
)
fig.suptitle(
    "Floquet DMD — parametrically forced field, monodromy multiplier ρ: 0.30 → 0.95",
    fontsize=11, fontweight="semibold",
)

# --- Field panel: a couple of cells ---
t_idx = np.arange(N)
axes[0].plot(t_idx, x_csd[0], color="#1565C0", lw=0.6, label="cell 0")
axes[0].plot(t_idx, x_csd[1], color="grey", lw=0.6, alpha=0.7, label="cell 1")
axes[0].set_ylabel("State", fontsize=9)
axes[0].legend(fontsize=8, loc="upper left")
axes[0].tick_params(labelbottom=False)

# --- CSD indicator panel with ground-truth ρ overlay ---
plot_indicator_on_ax(
    axes[1], r_csd, show_trend=True, show_significance=True,
    color="#263238", stats_loc="upper left",
)
axes[1].plot(
    r_csd.times, rho_csd[r_csd.times.astype(int)],
    color="#D32F2F", lw=1.4, ls=":", label="true ρ(t)",
)
axes[1].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
axes[1].set_ylabel("|Floquet mult.|\n(CSD)", fontsize=8)
axes[1].legend(fontsize=8, loc="lower right")

# --- Control indicator panel ---
plot_indicator_on_ax(
    axes[2], r_ctrl, show_trend=True, show_significance=True,
    color="#607D8B", stats_loc="upper left",
)
axes[2].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
axes[2].set_ylabel("|Floquet mult.|\n(control)", fontsize=8)
axes[2].set_xlabel("Sample index", fontsize=9)

plt.tight_layout()
fig.savefig("csd_floquet_dmd.png", dpi=150, bbox_inches="tight")
print("\nFigure saved to csd_floquet_dmd.png")

plt.show()
