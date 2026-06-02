"""
Basic CSD analysis example.

Demonstrates how to:
1. Generate a synthetic time series with decreasing restoring rate (CSD).
2. Run the analysis with all available methods.
3. Inspect the Kendall τ trend.
4. Test statistical significance via Fourier surrogates.
5. Visualise results with the csdtools plotting utilities.

Run from the repository root::

    python -m csdtools.examples.basic_analysis
"""

import numpy as np
import matplotlib.pyplot as plt

from csdtools import analyze_timeseries, plot_analysis, plot_multi_analysis, plot_significance

# ──────────────────────────────────────────────────────────────────────────────
# 1. Generate synthetic data
# ──────────────────────────────────────────────────────────────────────────────
# AR(1) process with φ increasing linearly from 0.3 to 0.95,
# mimicking CSD as a system approaches a tipping point.

rng = np.random.default_rng(0)
T = 1200
sigma = 0.3

phi = np.linspace(0.3, 0.95, T)
x = np.zeros(T)
for t in range(1, T):
    x[t] = phi[t] * x[t - 1] + sigma * rng.standard_normal()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Run all methods
# ──────────────────────────────────────────────────────────────────────────────
# lambda_km (Kramers-Moyal) is non-parametric and needs more data per window.
WINDOW      = 200
WINDOW_300  = 300   # lambda_acs, lambda_psd, lambda_km benefit from more data
DETREND_W   = 500

WINDOW_SIZES = {
    "lambda_acs": WINDOW_300,
    "lambda_psd": WINDOW_300,
    "lambda_km":  WINDOW_300,
}

methods = ["ac1", "variance", "skewness",
           "lambda_ac1", "lambda_var",
           "lambda_glsar", "lambda_ols",
           "lambda_acs", "lambda_psd", "rosa", "lambda_km"]

results = {}

print("=" * 60)
print(f"{'Method':<20} {'Kendall τ':>10} {'p (Kendall)':>12}")
print("-" * 60)

for m in methods:
    ws = WINDOW_SIZES.get(m, WINDOW)
    r = analyze_timeseries(
        x,
        method=m,
        window_size=ws,
        detrend="running_mean",
        detrend_window=DETREND_W,
        local_detrend=True,
    )
    results[m] = r
    print(f"{m:<20} {r.kendall_tau:>+10.3f} {r.kendall_p_value:>12.4f}")

print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Significance test for lambda_glsar
# ──────────────────────────────────────────────────────────────────────────────
print("\nRunning surrogate significance test for 'lambda_glsar' …")

r_sig = analyze_timeseries(
    x,
    method="lambda_glsar",
    window_size=WINDOW,
    detrend="running_mean",
    detrend_window=DETREND_W,
    significance=True,
    n_surrogates=500,
)

print(f"  Observed Kendall τ : {r_sig.kendall_tau:+.3f}")
print(f"  Surrogate p-value  : {r_sig.surrogate_p_value:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4a. Multi-method plot: time series + all indicators stacked
# ──────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plot_multi_analysis(
    x,
    results,
    x_label="Sample index",
    ts_label="x(t)",
    title="CSD simulation — all methods compared",
    show_trend=True,
    show_significance=True,
)
fig1.savefig("csd_basic_multi.png", dpi=150, bbox_inches="tight")
print("\nFigure saved to csd_basic_multi.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4b. Single-method plot: λ (GLSAR) with surrogate significance
# ──────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plot_analysis(
    x,
    r_sig,
    x_label="Sample index",
    ts_label="x(t)",
    title="CSD simulation — λ (GLSAR) with significance",
    show_trend=True,
    show_significance=True,
)
fig2.savefig("csd_basic_single.png", dpi=150, bbox_inches="tight")
print("Figure saved to csd_basic_single.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4c. Surrogate null distribution for lambda_glsar
# ──────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plot_significance(r_sig)
fig3.savefig("csd_basic_significance.png", dpi=150, bbox_inches="tight")
print("Figure saved to csd_basic_significance.png")

plt.show()
