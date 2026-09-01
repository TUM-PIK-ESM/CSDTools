"""
ROSA (``rosa``) example — avoiding false positives under coloured forcing.

The Ratio Of Spectra Approach (Clarke et al. 2023) estimates the restoring rate
λ from the ratio of the *response* spectrum to the *forcing* spectrum,

    R(ω) = S_xx(ω) / S_ξξ(ω) ∝ 1 / (ω² + λ²).

Because the forcing spectrum is divided out, ROSA is robust to **coloured
(non-white) forcing** — its "beyond white noise" property.  This example makes
the point with the classic failure mode of naive CSD indicators: a system whose
resilience is **constant**, but whose forcing noise becomes increasingly *red*
over time.  The growing low-frequency power masquerades as Critical Slowing
Down, so a naive (white-forcing) fit reports a strong — but entirely spurious —
loss of resilience.  Supplying the (time-varying) forcing spectrum removes the
artefact.

Synthetic data
--------------
A single stable Ornstein-Uhlenbeck process with a **constant** restoring rate
λ = 0.20, driven by an AR(1) forcing whose coefficient a(t) ramps from 0 (white)
to 0.9 (strongly red):

    ξ(t) = a(t)·ξ(t-1) + w(t),      a(t): 0 → 0.9
    x(t) = x(t-1) - λ·x(t-1)·dt + ξ(t)·dt

There is **no** real CSD here — the correct answer is "no trend".

Two analyses are compared:
  * ``rosa`` with no forcing supplied      → white-forcing assumption (naive)
  * ``rosa`` with ``noise=ξ``              → the true forcing, windowed in sync
                                             with each analysis window

Run from the repository root::

    python -m csdtools.examples.rosa_analysis
"""

import numpy as np
import matplotlib.pyplot as plt

from csdtools import analyze_timeseries
from csdtools.plotting import plot_indicator_on_ax

# ──────────────────────────────────────────────────────────────────────────────
# 1. Stable system (constant λ) driven by forcing that reddens over time
# ──────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(5)
DT  = 1.0
N   = 4000
LAM_TRUE = 0.20                       # constant restoring rate → NO real CSD

a  = np.linspace(0.0, 0.9, N)         # forcing AR(1) coefficient: white → red
xi = np.zeros(N)
for t in range(1, N):
    xi[t] = a[t] * xi[t - 1] + rng.standard_normal()

x = np.zeros(N)
for t in range(1, N):
    x[t] = x[t - 1] - LAM_TRUE * x[t - 1] * DT + xi[t] * DT

# ──────────────────────────────────────────────────────────────────────────────
# 2. Analyse: naive (white) ROSA vs forcing-aware ROSA
# ──────────────────────────────────────────────────────────────────────────────
common = dict(method="rosa", window_size=400, step=10, detrend=None)

print("Running ROSA analysis (constant λ, reddening forcing) …\n")
r_white = analyze_timeseries(x, **common)             # white-forcing assumption
r_noise = analyze_timeseries(x, **common, noise=xi)   # true forcing, windowed

print(f"{'Analysis':<22} {'Kendall τ':>10} {'p (Kendall)':>13}  interpretation")
print("-" * 70)
print(f"{'white (naive)':<22} {r_white.trend_statistic:>+10.3f} {r_white.trend_p_value:>13.2e}"
      f"  SPURIOUS CSD (false alarm)")
print(f"{'forcing-aware (noise)':<22} {r_noise.trend_statistic:>+10.3f} {r_noise.trend_p_value:>13.2e}"
      f"  no false trend (correct)")
print("-" * 70)
print(f"Ground truth: λ = -{LAM_TRUE:.2f} (constant) — resilience does NOT change.\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Plot: series, forcing redness, and the two ROSA indicators
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    4, 1, figsize=(11, 9.5), sharex=True,
    gridspec_kw={"height_ratios": [1.6, 1.0, 1.3, 1.3]},
)
fig.suptitle(
    "ROSA — constant resilience, reddening forcing: naive fit gives a false CSD signal",
    fontsize=11, fontweight="semibold",
)

t_idx = np.arange(N)

# --- Time series ---
axes[0].plot(t_idx, x, color="#1565C0", lw=0.5)
axes[0].set_ylabel("State x(t)", fontsize=9)
axes[0].tick_params(labelbottom=False)

# --- The confound: forcing AR(1) coefficient a(t) ---
axes[1].plot(t_idx, a, color="#8E24AA", lw=1.4)
axes[1].set_ylabel("Forcing redness\na(t)", fontsize=8)
axes[1].tick_params(labelbottom=False)

# --- Naive (white) ROSA: spurious upward trend ---
plot_indicator_on_ax(
    axes[2], r_white, show_trend=True, show_significance=True,
    color="#C62828", stats_loc="upper left",
)
axes[2].axhline(-LAM_TRUE, color="k", lw=0.8, ls="--", alpha=0.6,
                label=f"true λ = -{LAM_TRUE:.2f}")
axes[2].set_ylabel("λ  (ROSA, white)\n— false alarm", fontsize=8)
axes[2].legend(fontsize=8, loc="lower right")

# --- Forcing-aware ROSA: no false trend ---
plot_indicator_on_ax(
    axes[3], r_noise, show_trend=True, show_significance=True,
    color="#2E7D32", stats_loc="upper left",
)
axes[3].axhline(-LAM_TRUE, color="k", lw=0.8, ls="--", alpha=0.6,
                label=f"true λ = -{LAM_TRUE:.2f}")
axes[3].set_ylabel("λ  (ROSA, forcing)\n— correct", fontsize=8)
axes[3].legend(fontsize=8, loc="lower right")
axes[3].set_xlabel("Sample index", fontsize=9)

plt.tight_layout()
fig.savefig("csd_rosa.png", dpi=150, bbox_inches="tight")
print("Figure saved to csd_rosa.png")

plt.show()
