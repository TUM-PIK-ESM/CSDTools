"""
Significance testing example.

Demonstrates the surrogate-based significance test and compares
the null distribution against the observed Kendall τ.

Run from the repository root::

    python -m csdkit.examples.significance_testing
"""

import numpy as np
import matplotlib.pyplot as plt

from csdkit import analyze_timeseries
from csdkit.significance import significance_test
from csdkit.preprocessing import preprocess

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data: series WITH and WITHOUT CSD
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
phi_stat = 0.5
x_stat = np.zeros(T)
for t in range(1, T):
    x_stat[t] = phi_stat * x_stat[t - 1] + sigma * rng.standard_normal()

# ──────────────────────────────────────────────────────────────────────────────
# Preprocess
# ──────────────────────────────────────────────────────────────────────────────
x_csd_proc  = preprocess(x_csd,  method="running_mean", window=300)
x_stat_proc = preprocess(x_stat, method="running_mean", window=300)

# ──────────────────────────────────────────────────────────────────────────────
# Significance tests
# ──────────────────────────────────────────────────────────────────────────────
print("Running surrogate tests (500 surrogates each) …")

tau_csd, p_csd, surr_taus_csd = significance_test(
    x_csd_proc, method="lambda_glsar", window_size=150,
    n_surrogates=500, seed=0
)
tau_stat, p_stat, surr_taus_stat = significance_test(
    x_stat_proc, method="lambda_glsar", window_size=150,
    n_surrogates=500, seed=0
)

print(f"CSD series   : τ = {tau_csd:+.3f},  p = {p_csd:.4f}")
print(f"Stationary   : τ = {tau_stat:+.3f},  p = {p_stat:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Plot null distributions
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, surr_taus, tau_obs, p, title in [
    (axes[0], surr_taus_csd,  tau_csd,  p_csd,  "CSD series"),
    (axes[1], surr_taus_stat, tau_stat, p_stat, "Stationary series"),
]:
    ax.hist(surr_taus[np.isfinite(surr_taus)], bins=40,
            color="lightsteelblue", edgecolor="white", label="Surrogate τ")
    ax.axvline(tau_obs, color="tomato", lw=2.5,
               label=f"Observed τ = {tau_obs:+.3f}\np = {p:.4f}")
    ax.set_xlabel("Kendall τ")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("csd_significance_example.png", dpi=150)
print("Figure saved to csd_significance_example.png")
plt.show()
