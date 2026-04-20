"""
Significance testing example.

Demonstrates the surrogate-based significance test and compares the null
distribution of a CSD series against a stationary one, using the csdtools
plotting utilities throughout.

Run from the repository root::

    python -m csdtools.examples.significance_testing
"""

import numpy as np
import matplotlib.pyplot as plt

from csdtools import analyze_timeseries, plot_analysis, plot_multi_analysis, plot_significance
from csdtools.plotting import plot_indicator_on_ax

# ──────────────────────────────────────────────────────────────────────────────
# 1. Synthetic data: series WITH and WITHOUT CSD
# ──────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(1)
T = 800
sigma = 0.5

# CSD series: φ increases from 0.2 to 0.9
phi_csd = np.linspace(0.2, 0.9, T)
x_csd = np.zeros(T)
for t in range(1, T):
    x_csd[t] = phi_csd[t] * x_csd[t - 1] + sigma * rng.standard_normal()

# Stationary series: constant φ = 0.5
x_stat = np.zeros(T)
for t in range(1, T):
    x_stat[t] = 0.5 * x_stat[t - 1] + sigma * rng.standard_normal()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Analyse both series with significance testing
# ──────────────────────────────────────────────────────────────────────────────
print("Running analyses with surrogate significance tests (50 surrogates) …")

WINDOW   = 150
DETREND_W = 300

r_csd = analyze_timeseries(
    x_csd, method="lambda_glsar", window_size=WINDOW,
    detrend="running_mean", detrend_window=DETREND_W,
    significance=True, n_surrogates=50,
)
r_stat = analyze_timeseries(
    x_stat, method="lambda_glsar", window_size=WINDOW,
    detrend="running_mean", detrend_window=DETREND_W,
    significance=True, n_surrogates=50,
)

print(f"CSD series   : τ = {r_csd.kendall_tau:+.3f},  p = {r_csd.surrogate_p_value:.4f}")
print(f"Stationary   : τ = {r_stat.kendall_tau:+.3f},  p = {r_stat.surrogate_p_value:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 3a. Side-by-side single-method plots (time series + indicator)
# ──────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 7),
                            gridspec_kw={"height_ratios": [1.6, 2.2]})

# CSD column
axes1[0, 0].plot(x_csd, color="#263238", lw=0.7)
axes1[0, 0].set_ylabel("x(t)", fontsize=9)
axes1[0, 0].set_title("CSD series  (φ: 0.2 → 0.9)", fontsize=10)
axes1[0, 0].tick_params(labelbottom=False)

plot_indicator_on_ax(axes1[1, 0], r_csd, show_trend=True, show_significance=True,
                     stats_loc="upper left")
axes1[1, 0].set_xlabel("Sample index", fontsize=9)

# Stationary column
axes1[0, 1].plot(x_stat, color="#263238", lw=0.7)
axes1[0, 1].set_title("Stationary series  (φ = 0.5)", fontsize=10)
axes1[0, 1].tick_params(labelbottom=False)

plot_indicator_on_ax(axes1[1, 1], r_stat, show_trend=True, show_significance=True,
                     stats_loc="upper left")
axes1[1, 1].set_xlabel("Sample index", fontsize=9)

fig1.suptitle("λ (GLSAR) indicator: CSD vs stationary", fontsize=11, fontweight="semibold")
plt.tight_layout()
fig1.savefig("csd_sig_indicators.png", dpi=150)
print("\nFigure saved to csd_sig_indicators.png")

# ──────────────────────────────────────────────────────────────────────────────
# 3b. Side-by-side surrogate null distributions
# ──────────────────────────────────────────────────────────────────────────────
fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

plot_significance(r_csd,  ax=ax_l, title="CSD series — surrogate null distribution")
plot_significance(r_stat, ax=ax_r, title="Stationary series — surrogate null distribution")

fig2.savefig("csd_sig_nulldists.png", dpi=150)
print("Figure saved to csd_sig_nulldists.png")

# ──────────────────────────────────────────────────────────────────────────────
# 3c. Multi-method comparison on the CSD series
# ──────────────────────────────────────────────────────────────────────────────
print("\nRunning multi-method analysis on CSD series …")

multi_methods = ["ac1", "variance", "lambda_glsar", "lambda_ols"]
multi_results = {
    m: analyze_timeseries(
        x_csd, method=m, window_size=WINDOW,
        detrend="running_mean", detrend_window=DETREND_W,
    )
    for m in multi_methods
}

fig3, axes3 = plot_multi_analysis(
    x_csd,
    multi_results,
    x_label="Sample index",
    ts_label="x(t)",
    title="CSD series — multiple indicators",
    show_trend=True,
    show_significance=True,
)
fig3.savefig("csd_sig_multi.png", dpi=150)
print("Figure saved to csd_sig_multi.png")

plt.show()
