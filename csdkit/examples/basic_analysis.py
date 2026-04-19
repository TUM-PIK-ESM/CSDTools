"""
Basic CSD analysis example.

Demonstrates how to:
1. Generate a synthetic time series with decreasing restoring rate (CSD).
2. Run the analysis with several methods.
3. Inspect the Kendall τ trend.
4. Test statistical significance via Fourier surrogates.

Run from the repository root::

    python -m csdkit.examples.basic_analysis
"""

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1. Generate synthetic data
# ──────────────────────────────────────────────────────────────────────────────
# Simulate an AR(1) process with φ increasing linearly from 0.3 to 0.95,
# mimicking the CSD that occurs as a system approaches a tipping point.

rng = np.random.default_rng(0)
T = 1000
sigma = 0.3

# φ increases over time → restoring rate λ = log(φ) increases toward 0
phi = np.linspace(0.3, 0.95, T)

x = np.zeros(T)
for t in range(1, T):
    x[t] = phi[t] * x[t - 1] + sigma * rng.standard_normal()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Run the analysis pipeline with different methods
# ──────────────────────────────────────────────────────────────────────────────
from csdkit import analyze_timeseries

methods = ["ac1", "variance", "lambda_glsar", "lambda_ols", "lambda_psd", "rosa"]
results = {}

print("=" * 60)
print(f"{'Method':<20} {'Kendall τ':>10} {'p (Kendall)':>12}")
print("-" * 60)

for m in methods:
    r = analyze_timeseries(
        x,
        method=m,
        window_size=200,   # 200-sample sliding window
        detrend="running_mean",
        detrend_window=400,
        local_detrend=True,
    )
    results[m] = r
    print(f"{m:<20} {r.kendall_tau:>+10.3f} {r.kendall_p_value:>12.4f}")

print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Significance test for the best method
# ──────────────────────────────────────────────────────────────────────────────
print("\nRunning surrogate significance test for 'lambda_glsar' …")

r_sig = analyze_timeseries(
    x,
    method="lambda_glsar",
    window_size=200,
    detrend="running_mean",
    detrend_window=400,
    significance=True,
    n_surrogates=500,
)

print(f"  Observed Kendall τ : {r_sig.kendall_tau:+.3f}")
print(f"  Surrogate p-value  : {r_sig.surrogate_p_value:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Plot
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axes[0].plot(x, lw=0.7, color="steelblue")
axes[0].set_ylabel("x(t)")
axes[0].set_title("Synthetic time series with increasing φ (CSD simulation)")

for ax, m in zip(axes[1:], ["ac1", "variance", "lambda_glsar"]):
    r = results[m]
    ax.plot(r.times, r.indicator, lw=1.0, color="darkorange")
    ax.set_ylabel(m)
    ax.set_title(f"{m}  (τ = {r.kendall_tau:+.3f})")

axes[-1].set_xlabel("Time index")
plt.tight_layout()
plt.savefig("csd_basic_example.png", dpi=150)
print("\nFigure saved to csd_basic_example.png")
plt.show()
