"""
Kendall-versus-linear trend assessment example.

Runs one CSD indicator analysis with both Kendall's tau and an ordinary
least-squares slope. Each statistic is tested against its own null distribution
while sharing the expensive surrogate-indicator computation.

Run from the repository root::

    python -m csdtools.examples.significance_testing
"""

import matplotlib.pyplot as plt
import numpy as np

from csdtools import analyze_timeseries, plot_analysis, plot_significance


# 1. Synthetic CSD data: AR(1) persistence increases toward one.
rng = np.random.default_rng(1)
n_samples = 800
phi = np.linspace(0.2, 0.9, n_samples)
x = np.zeros(n_samples)
for t in range(1, n_samples):
    x[t] = phi[t] * x[t - 1] + 0.5 * rng.standard_normal()


# 2. Assess the same lambda indicator with two alternative trend methods.
common = dict(
    method="lambda_glsar",
    window_size=150,
    detrend="running_mean",
    detrend_window=300,
    significance=True,
    n_surrogates=50,
    surrogate_alternative="greater",
    surrogate_seed=42,
)

print("Running shared Kendall and linear surrogate significance tests …")
result = analyze_timeseries(
    x,
    trend_methods=("kendall", "linear"),
    trend_alternative="two-sided",
    **common,
)

kendall = result.trends["kendall"]
linear = result.trends["linear"]

print(
    f"Kendall: tau={kendall.statistic:+.3f}, "
    f"analytical p={kendall.p_value:.4g}, "
    f"surrogate p={kendall.surrogate_p_value:.4f}"
)
print(
    f"Linear : slope={linear.statistic:+.4e} per sample, "
    f"R^2={linear.details['r_squared']:.3f}, "
    f"analytical p={linear.p_value:.4g}, "
    f"surrogate p={linear.surrogate_p_value:.4f}"
)


# 3. The one indicator panel displays both stored assessments.
fig1, _ = plot_analysis(
    x,
    result,
    x_label="Sample index",
    ts_label="x(t)",
    title="One CSD indicator with two stored trend assessments",
    show_trend=True,
    show_significance=True,
)
fig1.savefig("csd_sig_trend_comparison.png", dpi=150, bbox_inches="tight")
print("Figure saved to csd_sig_trend_comparison.png")


# 4. Each method has a null distribution in its own statistic's units.
fig2, (ax_k, ax_l) = plt.subplots(
    1, 2, figsize=(13, 4), constrained_layout=True
)
plot_significance(
    result,
    trend_method="kendall",
    ax=ax_k,
    title="Kendall tau surrogate null distribution",
)
plot_significance(
    result,
    trend_method="linear",
    ax=ax_l,
    title="OLS-slope surrogate null distribution",
)
fig2.savefig("csd_sig_nulldists.png", dpi=150, bbox_inches="tight")
print("Figure saved to csd_sig_nulldists.png")

plt.show()
