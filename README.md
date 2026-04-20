# CSDTools — Critical Slowing Down Analysis Toolkit

A unified Python toolkit for detecting loss of resilience in dynamical systems
using **Critical Slowing Down (CSD)** theory.  The package consolidates
several methods that were previously scattered across different files in this
repository into a single, consistent API.

---

## Table of Contents

1. [Background: What is CSD?](#1-background-what-is-csd)
2. [Mathematical Framework](#2-mathematical-framework)
3. [Available Methods](#3-available-methods)
   - [Lag-1 Autocorrelation (AC1)](#lag-1-autocorrelation-ac1)
   - [Variance](#variance)
   - [Lambda via GLSAR](#lambda-via-glsar)
   - [Lambda via OLS](#lambda-via-ols)
   - [Lambda via ACS](#lambda-via-acs)
   - [Lambda via PSD](#lambda-via-psd)
   - [Lambda via Kramers-Moyal](#lambda-via-kramers-moyal)
   - [ROSA (Ratio of Spectra)](#rosa-ratio-of-spectra)
   - [Spatial Correlation](#spatial-correlation)
   - [Empirical Recovery Rate](#empirical-recovery-rate)
4. [Installation](#4-installation)
5. [Workflow](#5-workflow)
   - [Step 1 — Load your data](#step-1--load-your-data)
   - [Step 2 — Choose a preprocessing method](#step-2--choose-a-preprocessing-method)
   - [Step 3 — Choose an indicator method](#step-3--choose-an-indicator-method)
   - [Step 4 — Run the analysis](#step-4--run-the-analysis)
   - [Step 5 — Visualise the results](#step-5--visualise-the-results)
   - [Step 6 — Assess statistical significance](#step-6--assess-statistical-significance)
   - [Step 7 — Compare multiple methods](#step-7--compare-multiple-methods)
   - [Handling missing data (NaNs)](#handling-missing-data-nans)
6. [Full API Reference](#6-full-api-reference)
7. [Preprocessing Guide](#7-preprocessing-guide)
8. [Significance Testing](#8-significance-testing)
9. [Plotting](#9-plotting)
10. [Examples](#10-examples)
11. [References](#11-references)

---

## 1. Background: What is CSD?

Many natural and engineered systems can transition abruptly between
qualitatively different states — *tipping points* — when an external
parameter crosses a critical threshold.  Examples include:

- The Atlantic Meridional Overturning Circulation (AMOC) collapsing.
- Amazon rainforest dieback due to warming and deforestation.
- Arctic sea-ice loss transitioning to a seasonally ice-free state.
- Ecological regime shifts (e.g., from coral reef to algae-dominated).

Near such a tipping point, the dynamical system **slows down**: it takes
longer to return to equilibrium after being disturbed.  This phenomenon is
called **Critical Slowing Down** and it produces measurable statistical
fingerprints in observed time series *before* the transition occurs —
making it a potential early warning signal.

---

## 2. Mathematical Framework

### The OU process near a bifurcation

Close to a saddle-node bifurcation, the dynamics of a system variable *x*
around its equilibrium can be linearised to an **Ornstein-Uhlenbeck (OU)**
process:

```
dx = λ x dt + σ dW
```

where:
- **λ < 0** is the *linear restoring rate* (stability parameter).
- **σ > 0** is the noise amplitude.
- **W** is a Wiener process (white noise).

As the system approaches the bifurcation, **λ → 0** from below.  This is
Critical Slowing Down.

### Discrete-time equivalent

In discrete time with step Δt the OU process becomes an AR(1) model:

```
x(t + Δt) = φ x(t) + ε(t),     φ = exp(λ Δt)
```

For a stable system **φ ∈ (0, 1)**.  At the tipping point **φ → 1**.

### Observable consequences of CSD

| Indicator | Formula | CSD behaviour |
|-----------|---------|---------------|
| AC1 | `⟨x(t)x(t+Δt)⟩ / Var(x)` ≈ φ | Increases toward 1 |
| Variance | `σ²/(1 − φ²)` | Increases toward ∞ |
| Restoring rate | `λ = log(φ)/Δt` | Increases toward 0 |

All methods in csdtools either estimate **λ** directly or track **AC1** and
**variance** as proxies.

---

## 3. Available Methods

### Lag-1 Autocorrelation (AC1)

**Method name**: `'ac1'`

The simplest CSD indicator.  For an AR(1) process the expected lag-1
autocorrelation equals φ = exp(λΔt).  As λ → 0, AC1 → 1.

**Pros**: Fast, easy to interpret, widely established.  
**Cons**: Only uses one lag; does not directly estimate λ.

```python
result = analyze_timeseries(x, method='ac1', window_size=100)
```

---

### Variance

**Method name**: `'variance'`

The steady-state variance of the OU process is σ²/(1 − φ²).  As φ → 1,
variance diverges.

**Pros**: Fast, complementary to AC1.  
**Cons**: Sensitive to changes in noise amplitude σ that are not caused by CSD.

```python
result = analyze_timeseries(x, method='variance', window_size=100)
```

---

### Lambda via GLSAR

**Method name**: `'lambda_glsar'`  *(recommended default)*

Fits the AR(1) regression `x(t) = φ x(t−1) + ε(t)` using **Generalised
Least Squares** with an AR(1) residual covariance structure (statsmodels
`GLSAR`), then computes `λ = log(φ)/Δt`.

The key advantage of GLS over plain OLS is that it accounts for the fact
that the regression residuals ε(t) are themselves autocorrelated.  Ignoring
this leads to a downward bias in φ.

**Pros**: Accurate, well-established in the CSD literature.  
**Cons**: Slightly slower than OLS; requires ≥ 30 samples per window.

```python
result = analyze_timeseries(x, method='lambda_glsar', dt=1.0, window_size=100)
```

**References**: Held & Kleinen (2004); Dakos et al. (2008); Boers (2021).

---

### Lambda via OLS

**Method name**: `'lambda_ols'`

Uses the Yule-Walker estimator (least-squares regression) to estimate φ:

```
φ̂ = Cov(x[t], x[t−1]) / Var(x[t−1])
```

then `λ = log(φ̂)/Δt`.

**Pros**: Very fast.  
**Cons**: Slightly biased downward in short windows relative to GLSAR.

```python
result = analyze_timeseries(x, method='lambda_ols', window_size=100)
```

---

### Lambda via ACS

**Method name**: `'lambda_acs'`

Fits the **theoretical autocorrelation structure** of the OU process to the
observed empirical ACF.  For an OU process, `C(τ) = exp(λ τ)`, so fitting
this to multiple observed lags simultaneously gives a robust estimate of λ.

The number of lags used is controlled by the `max_lag` parameter.

**Pros**: Uses more information (multiple lags) than a single-lag estimator;
robust to moderate noise.  
**Cons**: Requires solving a nonlinear optimisation; ≥ 300 samples recommended.

```python
result = analyze_timeseries(x, method='lambda_acs', window_size=300,
                            max_lag=20)
```

**References**: Morr & Boers (2024).

---

### Lambda via PSD

**Method name**: `'lambda_psd'`

Fits the **theoretical power spectral density** of the OU process:

```
S(f) = 2σ² / (λ² + (2πf)²)
```

to the periodogram of each window using log-space least squares.  Fitting
in log-space is robust to the heavy-tailed distribution of periodogram
ordinates.

**Pros**: Uses all available frequency information; independent of
time-domain autocorrelation estimates.  
**Cons**: Sensitive to spectral leakage; ≥ 300 samples recommended for reliable fits.

```python
result = analyze_timeseries(x, method='lambda_psd', window_size=300)
```

**References**: Morr & Boers (2024); Morr et al. (2024).

---

### Lambda via Kramers-Moyal

**Method name**: `'lambda_km'`

Uses the **Kramers-Moyal expansion** to estimate the drift coefficient
non-parametrically.  For an OU process the first Kramers-Moyal coefficient
(drift) is `D¹(x) = λ x`, so the slope of D¹ vs x gives λ directly.

This method makes *no assumption* about the functional form of the noise
beyond the Markov property, making it valuable when the underlying process
may deviate from the simple OU model.

**Pros**: Non-parametric; detects nonlinear effects.  
**Cons**: Requires substantially more data (≥ 200 samples recommended);
requires the `kramersmoyal` package.

```python
result = analyze_timeseries(x, method='lambda_km', dt=1.0,
                            window_size=300, bin_number=50)
```

**References**: Morr & Boers (2024); Morr et al. (2024).

---

### ROSA (Ratio of Spectra)

**Method name**: `'rosa'`

Introduced by Clarke et al. (2023) as a more robust spectral EWS indicator.
The **Ratio Of Spectra Approach** computes the ratio of spectral power in a
low-frequency band to a high-frequency band:

```
R = ⟨S(f < η f_Nyq)⟩ / ⟨S(f > η f_Nyq)⟩
```

For an OU process this ratio is a deterministic function of λ:

```
R(λ) = (λ² + (2π f_high)²) / (λ² + (2π f_low)²)
```

Inverting this gives an estimate of λ.  The frequency split parameter
`eta` (default 0.5) controls the boundary between the two bands.

**Pros**: More robust to certain types of non-stationarity than full-spectrum
fitting; computationally efficient.  
**Cons**: Assumes an OU-like spectrum; performance depends on `eta`.

```python
result = analyze_timeseries(x, method='rosa', window_size=100, eta=0.5)
```

**References**: Clarke et al. (2023).

---

### Spatial Correlation

Found in `csdtools.spatial`.  Not a time-series indicator in the same sense as
the above — instead, it quantifies the **spatial correlation length**
increase predicted by CSD in spatially extended systems.

```python
from csdtools.spatial import sliding_spatial_correlation, compute_distance_matrix

D = compute_distance_matrix(lats, lons)   # pairwise km distances
spc, times = sliding_spatial_correlation(
    data,           # shape (n_cells, n_time)
    D,
    window_size=120,
)
```

**References**: Blaschke et al. (2024).

---

### Empirical Recovery Rate

Found in `csdtools.empirical`.  For time series where transitions *have
already occurred*, fits an exponential recovery curve after each transition
to estimate the empirical recovery rate λ_emp.

```python
from csdtools.empirical import empirical_recovery_rate

results = empirical_recovery_rate(x, window_size=100)
for r in results:
    print(r['lambda_empirical'], r['r_squared'])
```

---

## 4. Installation

Clone the repository and install:

```bash
git clone <repo-url>
cd CSDTools
pip install .
```

This installs all required dependencies (`numpy`, `scipy`, `statsmodels`,
`matplotlib`) automatically.

Optional extras for specific methods:

```bash
pip install ".[km]"          # adds kramersmoyal  (method='lambda_km')
pip install ".[spatial]"     # adds pyproj        (spatial.compute_distance_matrix)
pip install ".[km,spatial]"  # both
```

After installation, `import csdtools` works from any directory.

---

## 5. Workflow

This section walks through a complete analysis from raw data to publication-ready
figures.  Each step corresponds to a function in csdtools.

### Step 1 — Load your data

csdtools works with any 1-D NumPy array of evenly-sampled observations.  Load
your data however is most convenient and convert it to an array:

```python
import numpy as np

# From a text file
x = np.loadtxt("my_data.csv", delimiter=",")

# From a NetCDF file via xarray
import xarray as xr
ds = xr.open_dataset("my_data.nc")
x = ds["variable"].values   # shape (T,)

# Or generate synthetic data for testing
rng = np.random.default_rng(42)
T = 1000
phi = np.linspace(0.3, 0.95, T)
x = np.zeros(T)
for t in range(1, T):
    x[t] = phi[t] * x[t - 1] + 0.3 * rng.standard_normal()
```

**Missing values**: NaN and Inf values are stripped automatically.  A warning
is emitted with the count and percentage removed.  See
[Handling missing data](#handling-missing-data-nans) below.

---

### Step 2 — Choose a preprocessing method

Raw observations typically contain a slow trend (seasonal cycle, long-term
mean drift) that must be removed before computing CSD indicators.  Pass the
`detrend` and `detrend_window` arguments to `analyze_timeseries`:

| `detrend=` | Description | Typical use |
|---|---|---|
| `'running_mean'` | Subtract a centred running mean | Default; handles smooth trends |
| `'linear'` | Remove a global linear regression | When trend is approximately linear |
| `'quadratic'` | Remove a global quadratic polynomial | Slow nonlinear drift |
| `'stl'` | STL seasonal decomposition | Seasonal data (requires `period`) |
| `'harmonic'` | Remove fitted sine/cosine harmonics | Known seasonal frequency |

The `detrend_window` controls the bandwidth of the running mean (in samples).
A common rule of thumb: set it to roughly half the length of the series, or
to the dominant cycle length if a periodic signal is present.

```python
# Remove a running mean with a 400-sample bandwidth
result = analyze_timeseries(x, method='lambda_glsar',
                            detrend='running_mean', detrend_window=400)
```

Set `local_detrend=True` to additionally remove a linear trend *within each
sliding window* (reduces spurious trends in the indicator time series for
highly non-stationary data).

---

### Step 3 — Choose an indicator method

| `method=` | What it measures | When to use |
|---|---|---|
| `'ac1'` | Lag-1 autocorrelation | Quick first look; widely cited |
| `'variance'` | Window variance | Complementary to AC1 |
| `'lambda_glsar'` | Restoring rate via GLS | **Recommended default** |
| `'lambda_ols'` | Restoring rate via Yule-Walker | Faster alternative to GLSAR |
| `'lambda_acs'` | Restoring rate via ACF fit | Robust with multiple lags; ≥ 300 pts/window |
| `'lambda_psd'` | Restoring rate via PSD fit | Full spectral information; ≥ 300 pts/window |
| `'lambda_km'` | Restoring rate via Kramers-Moyal | Non-parametric; ≥ 300 pts/window |
| `'rosa'` | Ratio of spectral bands | Robust to some non-stationarities |

Start with `lambda_glsar` unless you have a specific reason to prefer another.
Use multiple methods together (Step 7) to assess robustness.

---

### Step 4 — Run the analysis

```python
from csdtools import analyze_timeseries

result = analyze_timeseries(
    x,
    method        = 'lambda_glsar',   # indicator method
    window_size   = 200,               # samples per sliding window
    step          = 1,                 # window step (1 = maximum overlap)
    detrend       = 'running_mean',    # global detrending method
    detrend_window= 400,               # running-mean bandwidth (samples)
    local_detrend = True,              # also detrend within each window
    dt            = 1.0,               # sampling interval (any consistent unit)
    significance  = False,             # set True to compute surrogate p-value
)

print(f"Kendall τ  = {result.kendall_tau:+.3f}")
print(f"Kendall p  = {result.kendall_p_value:.4f}")
```

The returned `CSDResult` object contains:

| Attribute | Description |
|---|---|
| `result.indicator` | Array of indicator values, one per window |
| `result.times` | Array of window-centre indices |
| `result.kendall_tau` | Kendall τ trend of the indicator |
| `result.kendall_p_value` | Kendall p-value (Mann-Kendall test) |
| `result.surrogate_p_value` | Surrogate p-value (`None` if `significance=False`) |
| `result.surrogate_taus` | Array of surrogate τ values (`None` if not run) |
| `result.method` | Method name string |
| `result.window_size` | Window size used |
| `result.detrend_method` | Detrending method name |

**Window-size guidance**: A warning is emitted automatically if `window_size`
is below the recommended minimum for the chosen method (e.g., 30 for GLSAR,
200 for Kramers-Moyal).

---

### Step 5 — Visualise the results

```python
from csdtools import plot_analysis

fig, axes = plot_analysis(
    x,
    result,
    title          = "My CSD analysis",
    show_trend     = True,        # overlay Theil-Sen trend line
    show_significance = True,     # annotate τ and p-value on plot
)
fig.savefig("csd_result.png", dpi=150)
```

`plot_analysis` returns a `(fig, axes)` pair where `axes[0]` is the time
series panel and `axes[1]` is the indicator panel — you can customise them
further with standard matplotlib calls after the function returns.

Pass a `time_axis` array to label the x-axis with real dates or physical
units instead of sample indices:

```python
import pandas as pd
times = pd.date_range("1990-01", periods=len(x), freq="ME")
fig, axes = plot_analysis(x, result, time_axis=times.to_numpy())
```

---

### Step 6 — Assess statistical significance

Surrogate significance testing asks: "Could this Kendall τ arise by chance
from a stationary process with the same power spectrum?"

```python
result = analyze_timeseries(
    x,
    method       = 'lambda_glsar',
    window_size  = 200,
    detrend      = 'running_mean',
    detrend_window = 400,
    significance = True,          # enable surrogate test
    n_surrogates = 500,           # 500 is a good default (1000 for publication)
)

print(f"Surrogate p-value = {result.surrogate_p_value:.4f}")
# p < 0.05 → significant trend unlikely to arise from a stationary process
```

Visualise the null distribution:

```python
from csdtools import plot_significance

fig, ax = plot_significance(result)
fig.savefig("null_dist.png", dpi=150)
```

The plot shows a histogram of surrogate Kendall τ values with the observed τ
marked and the 5 % rejection region shaded in red.

---

### Step 7 — Compare multiple methods

Robustness across methods strengthens the evidence for CSD:

```python
from csdtools import analyze_timeseries, plot_multi_analysis

# lambda_acs, lambda_psd, and lambda_km need more data per window
WINDOW_SIZES = {'lambda_acs': 300, 'lambda_psd': 300, 'lambda_km': 300}

methods = ['ac1', 'variance', 'lambda_glsar', 'lambda_ols',
           'lambda_acs', 'lambda_psd', 'rosa', 'lambda_km']

results = {
    m: analyze_timeseries(
        x, method=m,
        window_size=WINDOW_SIZES.get(m, 200),
        detrend='running_mean', detrend_window=400,
    )
    for m in methods
}

fig, axes = plot_multi_analysis(
    x, results,
    title      = "Multi-method CSD comparison",
    show_trend = True,
)
fig.savefig("csd_multi.png", dpi=150)
```

Each indicator is plotted in its own colour-coded panel below the original
time series, with a Theil-Sen trend line overlay and τ annotation.

---

### Handling missing data (NaNs)

By default, NaN and Inf values are silently removed and the remaining samples
are treated as a contiguous sequence:

```python
import numpy as np
x_with_nans = x.copy()
x_with_nans[200:210] = np.nan   # introduce a gap

result = analyze_timeseries(x_with_nans, method='lambda_glsar', window_size=200)
# UserWarning: Input time series contains 10 NaN/Inf value(s) (1.0 % of 1000).
#              They have been removed; remaining samples are treated as consecutive.
```

**Excluding NaN neighbours**: In some datasets a few corrupted samples
(telemetry dropout, sensor glitch) can distort lag-1 autocorrelations because
the samples immediately before and after the gap are now artificially
adjacent.  Pass `exclude_nan_neighbors=True` to also remove those neighbours:

```python
result = analyze_timeseries(
    x_with_nans,
    method               = 'lambda_glsar',
    window_size          = 200,
    exclude_nan_neighbors= True,   # also removes x[199] and x[210]
)
```

This is applied at the window level inside `compute_indicator` and is safe to
combine with all methods.

---

## 6. Full API Reference

### `analyze_timeseries`

```python
from csdtools import analyze_timeseries

result = analyze_timeseries(
    x,                    # 1D array-like, required
    method='lambda_glsar',  # str, default 'lambda_glsar'
    window_size=100,        # int
    step=1,                 # int; step=1 → maximum overlap
    detrend='running_mean', # str or None
    detrend_window=None,    # int; defaults to 2×window_size
    local_detrend=True,     # bool
    dt=1.0,                 # float; physical time step
    significance=False,     # bool
    n_surrogates=1000,      # int
    **method_kwargs,        # forwarded to indicator function
)
```

Returns a `CSDResult` with fields:

| Field | Type | Description |
|-------|------|-------------|
| `indicator` | `np.ndarray` | Indicator timeseries |
| `times` | `np.ndarray` | Window midpoint indices |
| `kendall_tau` | `float` | Kendall τ of the trend |
| `kendall_p_value` | `float` | p-value for Kendall τ |
| `surrogate_p_value` | `float` or `None` | Surrogate p-value |
| `method` | `str` | Method used |
| `window_size` | `int` | Window size used |
| `detrend_method` | `str` or `None` | Preprocessing used |

---

### `compute_indicator`

Compute a single indicator value on a pre-extracted window:

```python
from csdtools import compute_indicator

val = compute_indicator(window, method='lambda_glsar', dt=1.0)
```

---

### `sliding_window_analysis`

Low-level sliding window function, returns indicator array and times:

```python
from csdtools.sliding_window import sliding_window_analysis

indicator, times = sliding_window_analysis(
    x_preprocessed,
    method='ac1',
    window_size=100,
    step=1,
)
```

---

### `significance_test`

Run significance testing separately (e.g., for a previously computed
preprocessed series):

```python
from csdtools.significance import significance_test

tau, p_val, surr_taus = significance_test(
    x_processed,
    method='lambda_glsar',
    window_size=100,
    n_surrogates=1000,
)
```

---

### `preprocess`

Apply global preprocessing independently of the full pipeline:

```python
from csdtools.preprocessing import preprocess

x_dt = preprocess(x, method='running_mean', window=200)
x_dt = preprocess(x, method='stl', period=12)
x_dt = preprocess(x, method='harmonic', order=3, period=12)
```

---

## 7. Preprocessing Guide

Choosing the right preprocessing is as important as choosing the right
indicator.  The goal is to remove slow external forcing that is not
caused by CSD, without removing the CSD signal itself.

| Method | When to use |
|--------|-------------|
| `'running_mean'` | Most time series; the default choice.  Use a window 2–5× larger than the CSD analysis window. |
| `'linear'` | When the series has a single, approximately constant drift (e.g., long-term climate trend). |
| `'quadratic'` | When the forcing accelerates or decelerates. |
| `'stl'` | Regularly sampled seasonal data (monthly, daily) with a known cycle length. |
| `'harmonic'` | Similar to STL but with explicit harmonic components; useful for irregular grids. |
| `None` | Only when the series is already stationary / pre-processed. |

**Rule of thumb for running_mean window size**: use `detrend_window ≥ 2 ×
window_size`.  If the window is too small, part of the CSD signal itself
will be removed; if too large, the preprocessing is insufficient.

---

## 8. Significance Testing

A positive Kendall τ (increasing indicator trend) does *not* by itself
confirm CSD — even a stationary process can produce a trend by chance in a
finite sample.

The standard approach is to compare the observed τ against the **null
distribution** of τ values obtained from Fourier phase surrogates:

```python
result = analyze_timeseries(
    x, method='ac1', window_size=100,
    significance=True, n_surrogates=1000,
)

# Low p-value → the trend is unlikely to have arisen by chance
print(f"p = {result.surrogate_p_value:.4f}")
```

The surrogate p-value is the **fraction of surrogates** whose Kendall τ is
≥ the observed τ (one-sided test, appropriate for CSD which predicts an
*increase*).

**Interpretation**:
- `p < 0.05`: the CSD trend is statistically significant at the 5% level.
- `p ≥ 0.05`: the trend could plausibly arise from a stationary process.

**Caveats (from Ben-Yami et al. 2023)**:
- The test assumes the null hypothesis is a stationary process with the
  same power spectrum as the original.  If the observed series has
  pre-existing trends for other reasons (e.g., forced warming), the test
  may be misleading.
- 1000 surrogates gives a precision of ~0.001 on the p-value.  For
  exploratory work 200–500 may suffice.

---

## 9. Plotting

The `csdtools.plotting` module (also importable directly from `csdtools`)
provides three ready-made figure layouts.

---

### `plot_analysis` — single indicator

Two-panel figure: the raw time series in the top panel and the chosen CSD
indicator below it.  A dashed linear trend line and a stats box showing
Kendall τ and p-values are overlaid on the indicator panel by default.

```python
from csdtools import analyze_timeseries, plot_analysis

result = analyze_timeseries(x, method='lambda_glsar', window_size=100,
                            significance=True, n_surrogates=1000)

fig, axes = plot_analysis(
    x,
    result,
    time_axis=years,       # optional physical time axis (e.g. np.arange(1850, 2020))
    x_label='Year',
    ts_label='AMOC index',
    title='AMOC resilience — λ (GLSAR)',
    show_trend=True,       # dashed OLS trend line on indicator
    show_significance=True # Kendall τ / p-value annotation box
)
plt.show()
```

**Customisation**:

| Parameter | What it controls |
|-----------|-----------------|
| `ts_color` | Time series line colour |
| `indicator_color` | Indicator line colour |
| `stats_loc` | Box position: `'upper left'` / `'upper right'` / `'lower left'` / `'lower right'` |
| `ts_kwargs` / `indicator_kwargs` | Any extra `ax.plot` keyword arguments |
| `height_ratios` | Relative heights of the two panels |

---

### `plot_multi_analysis` — multiple indicators stacked

One panel per indicator, all sharing the same x-axis, with the raw time
series at the top.  Each method gets a distinct colour from the built-in
palette.

```python
from csdtools import analyze_timeseries, plot_multi_analysis

methods = ['ac1', 'variance', 'lambda_glsar', 'lambda_ols', 'rosa']
results = {m: analyze_timeseries(x, method=m, window_size=100)
           for m in methods}

fig, axes = plot_multi_analysis(
    x,
    results,             # dict {label: CSDResult}  or  list of CSDResult
    time_axis=years,
    x_label='Year',
    title='Multi-method CSD comparison',
)
plt.show()
```

The `results` argument can also be a plain **list** of `CSDResult` objects,
in which case the method names are used as panel labels automatically.

---

### `plot_significance` — surrogate null distribution

Histogram of the Fourier surrogate Kendall τ null distribution with the
observed τ marked as a vertical line.  The rejection region (surrogates with
τ ≥ observed) is shaded.

Requires `significance=True` in `analyze_timeseries` so the surrogate array
is stored in `result.surrogate_taus`.

```python
result = analyze_timeseries(x, method='ac1', window_size=100,
                            significance=True, n_surrogates=1000)

fig, ax = plot_significance(result)
plt.show()
```

---

### `plot_indicator_on_ax` — embed in a custom layout

Draw a single indicator panel onto an existing `matplotlib` axes object.
Useful when building larger custom figures.

```python
import matplotlib.pyplot as plt
from csdtools.plotting import plot_indicator_on_ax

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
plot_indicator_on_ax(axes[0], result_ac1, time_axis=years)
plot_indicator_on_ax(axes[1], result_var,  time_axis=years)
plot_indicator_on_ax(axes[2], result_glsar, time_axis=years)
plt.tight_layout()
```

---

### Method colour palette

Each method has a consistent default colour across all plot types:

| Method | Colour |
|--------|--------|
| `ac1` | Blue |
| `variance` | Green |
| `lambda_glsar` | Red |
| `lambda_ols` | Orange |
| `lambda_acs` | Purple |
| `lambda_psd` | Teal |
| `lambda_km` | Brown |
| `rosa` | Pink |

Override any colour via the `indicator_color` parameter or the
`csdtools.plotting.METHOD_COLORS` dictionary.

---

## 10. Examples

Example scripts are in `csdtools/examples/`:

| Script | Description |
|--------|-------------|
| `basic_analysis.py` | Run all 8 methods on a synthetic CSD series, compare Kendall τ values, test significance. |
| `significance_testing.py` | Compare null distributions for CSD vs stationary series; visualise surrogate histograms. |

Run from the `CSDTools` directory:

```bash
python -m csdtools.examples.basic_analysis
python -m csdtools.examples.significance_testing
```

### All-methods comparison

The multi-method plot shows the original time series (top) alongside all 8
CSD indicators in stacked panels, each with a Theil-Sen trend line and
annotated Kendall τ / p-value.

![All-methods comparison](docs/csd_basic_multi.png)

### Surrogate significance test

The null distribution plot shows the histogram of Kendall τ values from 500
Fourier-phase surrogates.  The observed τ (red line) sits far outside the
null distribution, confirming a statistically significant trend.

![Surrogate null distribution](docs/csd_basic_significance.png)

---

## 11. References

Clarke, J., Huntingford, C., Ritchie, P., & Cox, P. (2023).
Seeking more robust early warning signals for climate tipping points:
the ratio of spectra method.
*Earth System Dynamics*.

Dakos, V., Scheffer, M., van Nes, E. H., Brovkin, V., Petoukhov, V., &
Held, H. (2008).
Slowing down as an early warning signal for abrupt climate change.
*PNAS*, 105(38), 14308–14312.

Held, H., & Kleinen, T. (2004).
Detection of climate system bifurcations by degenerate fingerprinting.
*Geophysical Research Letters*, 31(23).

Boers, N. (2021).
Observation-based early-warning signals for a collapse of the Atlantic
Meridional Overturning Circulation.
*Nature Climate Change*, 11, 680–688.

Ben-Yami, M., Skiba, V., Bathiany, S., & Boers, N. (2023).
Uncertainties in AMOC early warning signals.
*Geophysical Research Letters*.

Morr, A., & Boers, N. (2024).
Detection of Approaching Critical Transitions in Multidimensional Systems.
*Physical Review Letters*.

Morr, A., Bathiany, S., & Boers, N. (2024).
Anticipating critical transitions in multidimensional systems.
*Nature Communications* (preprint).

Blaschke, L., Morr, A., & Boers, N. (2024).
Spatial Correlation Increase in Single-Sensor Satellite Data Reveals Loss
of Amazon Rainforest Resilience.
*Nature* (preprint).

Rietkerk, M., et al. (2025).
Ambiguity of early warning signals.
*Nature Reviews Earth & Environment*.

Zhang, R., et al. (2024).
Warming and disturbances affect Arctic-boreal vegetation resilience.
*Nature Climate Change*.
