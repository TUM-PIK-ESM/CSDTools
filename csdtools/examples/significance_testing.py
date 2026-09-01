"""
Kendall-versus-linear trend assessment example.

Runs the same CSD indicator analysis twice: once using Kendall's tau and once
using an ordinary least-squares slope.  Each selected statistic is also tested
against its own Fourier-surrogate null distribution.

Run from the repository root::

    python -m csdtools.examples.significance_testing
"""

import matplotlib.pyplot as plt
import numpy as np

from csdtools import analyze_timeseries, plot_multi_analysis, plot_significance


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

print("Running Kendall and linear surrogate significance tests …")
r_kendall = analyze_timeseries(
    x,
    trend_method="kendall",
    trend_alternative="two-sided",
    **common,
)
r_linear = analyze_timeseries(
    x,
    trend_method="linear",
    trend_alternative="two-sided",
    **common,
)

print(
    f"Kendall: tau={r_kendall.trend_statistic:+.3f}, "
    f"analytical p={r_kendall.trend_p_value:.4g}, "
    f"surrogate p={r_kendall.surrogate_p_value:.4f}"
)
print(
    f"Linear : slope={r_linear.trend_statistic:+.4e} per sample, "
    f"R^2={r_linear.trend_r_squared:.3f}, "
    f"analytical p={r_linear.trend_p_value:.4g}, "
    f"surrogate p={r_linear.surrogate_p_value:.4f}"
)


# 3. The indicator is identical; its annotation reflects the selected trend.
fig1, _ = plot_multi_analysis(
    x,
    {
        "lambda (Kendall trend)": r_kendall,
        "lambda (linear trend)": r_linear,
    },
    x_label="Sample index",
    ts_label="x(t)",
    title="One CSD indicator, two trend assessments",
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
    r_kendall,
    ax=ax_k,
    title="Kendall tau surrogate null distribution",
)
plot_significance(
    r_linear,
    ax=ax_l,
    title="OLS-slope surrogate null distribution",
)
fig2.savefig("csd_sig_nulldists.png", dpi=150, bbox_inches="tight")
print("Figure saved to csd_sig_nulldists.png")

plt.show()
