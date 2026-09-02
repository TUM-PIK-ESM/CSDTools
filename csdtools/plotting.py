"""
Plotting utilities for CSD analysis results.

Three main functions are provided:

plot_analysis(x, result, ...)
    Two-panel figure: the raw (or preprocessed) time series in the top
    panel and a single CSD indicator below it. A fitted trend line and
    stored trend statistic / p-value annotations can be overlaid on the
    indicator panel.

plot_multi_analysis(x, results, ...)
    Stacked figure: the time series in the top panel followed by one
    panel per method.  Ideal for comparing multiple indicators side by side.

plot_significance(result, ...)
    Histogram of the Fourier surrogate null distribution of the selected
    trend statistic with the observed value marked.  Requires
    significance=True to have been passed to analyze_timeseries().

All functions return the (fig, axes) pair for further customisation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict, Tuple, Union

from .trend import (
    fitted_trend_line,
    format_trend_details,
    format_trend_statistic,
    trend_analytical_test_label,
    trend_statistic_label,
)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

#: Per-method colours used in multi-method plots.
METHOD_COLORS: Dict[str, str] = {
    "ac1":          "#1976D2",   # blue
    "variance":     "#388E3C",   # green
    "skewness":     "#00796B",   # dark teal
    "lambda_ac1":   "#42A5F5",   # light blue
    "lambda_var":   "#66BB6A",   # light green
    "lambda_glsar": "#D32F2F",   # red
    "lambda_ols":   "#F57C00",   # orange
    "lambda_acs":   "#7B1FA2",   # purple
    "lambda_psd":   "#0097A7",   # teal
    "lambda_km":    "#5D4037",   # brown
    "lambda_dmd":   "#455A64",   # blue-grey
    "lambda_dmd_floquet": "#263238",  # dark blue-grey
    "rosa":         "#C2185B",   # pink
    "spatial_corr": "#8D6E63",   # taupe
}

_TS_COLOR     = "#263238"   # dark blue-grey — time series
_TREND_COLOR  = "#B71C1C"   # deep red — trend line
_SURR_COLOR   = "#B0BEC5"   # grey — surrogate histogram fill
_SURR_EDGE    = "#78909C"   # darker grey — surrogate histogram edge


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_trend_method(result, trend_method: Optional[str]) -> str:
    """Resolve an explicit method or select a useful stored default."""
    if trend_method is not None:
        if trend_method not in result.trends:
            raise ValueError(
                f"Trend method '{trend_method}' is not stored in this result. "
                f"Available methods: {list(result.trends)}"
            )
        return trend_method

    # Prefer the first stored slope fit. If every stored assessment is a
    # correlation, its registered OLS visual guide remains the default.
    for method, trend in result.trends.items():
        intercept = trend.details.get("intercept")
        if intercept is not None and np.isfinite(intercept):
            return method
    return next(iter(result.trends))


def _trend_line(
    result, trend_method: Optional[str]
) -> Tuple[np.ndarray, str]:
    """Stored fit or visual guide, evaluated at all window times."""
    method = _resolve_trend_method(result, trend_method)
    trend = result.trends[method]
    return fitted_trend_line(
        result.indicator,
        np.asarray(result.times, dtype=float) * result.dt,
        method,
        trend.statistic,
        trend.details,
    )


def _method_label(method: str) -> str:
    """Human-readable label for a method name."""
    _labels = {
        "ac1":          "AC1",
        "variance":     "Variance",
        "skewness":     "Skewness",
        "lambda_ac1":   "λ  (AC1)",
        "lambda_var":   "λ  (Var)",
        "lambda_glsar": "λ  (GLSAR)",
        "lambda_ols":   "λ  (OLS)",
        "lambda_acs":   "λ  (ACS)",
        "lambda_psd":   "λ  (PSD)",
        "lambda_km":    "λ  (KM)",
        "lambda_dmd":   "|eig|  (DMD)",
        "lambda_dmd_floquet": "|eig|  (DMD-Floquet)",
        "rosa":         "λ  (ROSA)",
        "spatial_corr": "Spatial corr.",
    }
    return _labels.get(method, method)


def _significance_stars(p: Optional[float]) -> str:
    """Return significance stars string for a given p-value."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _annotate_stats(
    ax: plt.Axes,
    result,
    loc: str = "upper left",
) -> None:
    """
    Place a stats box with all stored trend statistics and p-values.

    Parameters
    ----------
    ax : plt.Axes
    result : CSDResult
        Analysis result containing one or more stored trend assessments.
    loc : str
        'upper left', 'upper right', 'lower left', or 'lower right'.
    """
    lines = []
    for method, trend in result.trends.items():
        stars = _significance_stars(
            trend.surrogate_p_value
            if trend.surrogate_p_value is not None
            else trend.p_value
        )
        lines.append(
            f"{trend_statistic_label(method)} = "
            f"{format_trend_statistic(method, trend.statistic)}  {stars}"
        )
        p_label = trend_analytical_test_label(method)
        if trend.p_value is not None and np.isfinite(trend.p_value):
            lines.append(f"p ({p_label}) = {trend.p_value:.4f}")
        if (
            trend.surrogate_p_value is not None
            and np.isfinite(trend.surrogate_p_value)
        ):
            surrogate_label = "surrogate"
            if len(result.trends) > 1:
                method_label = p_label or trend_statistic_label(method)
                surrogate_label = f"{method_label} surrogate"
            lines.append(
                f"p ({surrogate_label}) = "
                f"{trend.surrogate_p_value:.4f}"
            )
        lines.extend(format_trend_details(method, trend.details))

    x_pos = 0.02 if "left" in loc else 0.98
    y_pos = 0.97 if "upper" in loc else 0.03
    ha    = "left" if "left" in loc else "right"
    va    = "top"  if "upper" in loc else "bottom"

    ax.text(
        x_pos, y_pos, "\n".join(lines),
        transform=ax.transAxes, ha=ha, va=va,
        fontsize=8.5, linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#CFD8DC",
            alpha=0.90,
        ),
    )


def _draw_indicator_panel(
    ax: plt.Axes,
    result,
    time_axis: Optional[np.ndarray],
    color: str,
    show_trend: bool,
    show_significance: bool,
    indicator_kwargs: dict,
    stats_loc: str,
    trend_line_method: Optional[str],
) -> None:
    """
    Render the indicator line, optional trend, and optional stats box
    onto an existing axes.
    """
    t = time_axis if time_axis is not None else result.times
    ind = result.indicator

    # Indicator line
    ax.plot(t, ind, color=color, lw=1.2, alpha=0.9, **indicator_kwargs)

    # Method-specific fit or visual trend guide
    if show_trend:
        trend, trend_label = _trend_line(result, trend_line_method)
        ax.plot(t, trend, color=_TREND_COLOR, lw=1.8, ls="--",
                label=trend_label, zorder=5)

    # Y-axis label
    ax.set_ylabel(_method_label(result.method), fontsize=9)

    # Stats annotation
    if show_significance:
        _annotate_stats(
            ax,
            result,
            loc=stats_loc,
        )

    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_analysis(
    x,
    result,
    time_axis: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    x_label: str = "Time",
    ts_label: str = "Value",
    show_trend: bool = True,
    show_significance: bool = True,
    ts_color: Optional[str] = None,
    indicator_color: Optional[str] = None,
    stats_loc: str = "upper left",
    ts_kwargs: Optional[dict] = None,
    indicator_kwargs: Optional[dict] = None,
    height_ratios: Tuple[float, float] = (1.8, 2.2),
    trend_line_method: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Two-panel figure: time series (top) + CSD indicator (bottom).

    Parameters
    ----------
    x : array-like
        The original or preprocessed time series to display in the top panel.
    result : CSDResult
        Result object from :func:`~csdtools.analyze_timeseries`.
    time_axis : array-like or None
        Physical time values (years, dates, etc.) for the x-axis.
        If None, sample indices are used.
    figsize : (float, float)
        Figure width × height in inches.
    title : str or None
        Overall figure title.
    x_label : str
        Label for the shared x-axis.
    ts_label : str
        Y-axis label for the time series panel.
    show_trend : bool
        Overlay the selected fit or visual trend guide on the indicator.
    show_significance : bool
        Add a stats box with all stored trend statistics and p-values.
    ts_color : str or None
        Line colour for the time series.  Defaults to dark blue-grey.
    indicator_color : str or None
        Line colour for the indicator.  Defaults to the method's palette
        colour if available, otherwise deep blue.
    stats_loc : str
        Location of the stats box: ``'upper left'`` (default),
        ``'upper right'``, ``'lower left'``, or ``'lower right'``.
    ts_kwargs : dict or None
        Extra keyword arguments forwarded to ``ax.plot`` for the time series.
    indicator_kwargs : dict or None
        Extra keyword arguments forwarded to ``ax.plot`` for the indicator.
    height_ratios : (float, float)
        Relative heights of the time series and indicator panels.
    trend_line_method : str or None
        Stored trend method whose fit is drawn. By default, the first stored
        slope method is preferred, otherwise the first stored correlation's
        OLS visual guide is used.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray, shape (2,)
        ``axes[0]`` = time series panel, ``axes[1]`` = indicator panel.

    Examples
    --------
    >>> result = analyze_timeseries(x, method='lambda_glsar', window_size=100)
    >>> fig, axes = plot_analysis(x, result)
    >>> plt.show()

    Pass physical time values:

    >>> years = np.arange(1850, 2020)
    >>> fig, axes = plot_analysis(x, result, time_axis=years,
    ...                           x_label='Year', title='AMOC resilience')
    """
    x = np.asarray(x, dtype=float)
    ts_kw  = ts_kwargs or {}
    ind_kw = indicator_kwargs or {}
    tc     = ts_color or _TS_COLOR
    ic     = indicator_color or METHOD_COLORS.get(result.method, "#1565C0")

    # Build the time axes
    t_full = time_axis if time_axis is not None else np.arange(len(x))
    t_ind  = (
        time_axis[result.times.astype(int)]
        if time_axis is not None
        else result.times
    )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs  = GridSpec(2, 1, figure=fig, height_ratios=list(height_ratios))
    ax_ts  = fig.add_subplot(gs[0])
    ax_ind = fig.add_subplot(gs[1], sharex=ax_ts)

    # --- Time series panel ---
    ax_ts.plot(t_full, x, color=tc, lw=0.8, **ts_kw)
    ax_ts.set_ylabel(ts_label, fontsize=9)
    ax_ts.tick_params(labelsize=8)
    ax_ts.xaxis.set_tick_params(labelbottom=False)

    # --- Indicator panel ---
    _draw_indicator_panel(
        ax_ind, result, t_ind, ic,
        show_trend, show_significance, ind_kw, stats_loc, trend_line_method,
    )
    ax_ind.set_xlabel(x_label, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="semibold")

    return fig, np.array([ax_ts, ax_ind])


def plot_multi_analysis(
    x,
    results: Union[list, dict],
    time_axis: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    x_label: str = "Time",
    ts_label: str = "Value",
    show_trend: bool = True,
    show_significance: bool = True,
    stats_loc: str = "upper left",
    ts_color: Optional[str] = None,
    ts_kwargs: Optional[dict] = None,
    indicator_kwargs: Optional[dict] = None,
    trend_line_method: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Stacked figure: time series (top) + one panel per CSD indicator.

    Parameters
    ----------
    x : array-like
        The original or preprocessed time series.
    results : list of CSDResult or dict {label: CSDResult}
        If a list, method names are used as panel labels.
        If a dict, the keys are used as panel labels (allowing custom names).
    time_axis : array-like or None
        Physical time values for the x-axis.
    figsize : (float, float) or None
        Figure size.  Defaults to (10, 2.5 × (n_methods + 1)).
    title : str or None
        Overall figure title.
    x_label : str
        X-axis label on the bottom panel.
    ts_label : str
        Y-axis label for the time series panel.
    show_trend : bool
        Overlay the selected fit or visual trend guide on each indicator panel.
    show_significance : bool
        Add stats boxes to each indicator panel.
    stats_loc : str
        Location of the stats boxes.
    ts_color : str or None
        Colour for the time series line.
    ts_kwargs : dict or None
        Extra kwargs for the time series line.
    indicator_kwargs : dict or None
        Extra kwargs for every indicator line.
    trend_line_method : str or None
        Stored trend method whose fit is drawn in every indicator panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray, shape (n_methods + 1,)
        ``axes[0]`` = time series panel, ``axes[1:]`` = indicator panels.

    Examples
    --------
    >>> from csdtools import analyze_timeseries
    >>> results = {
    ...     m: analyze_timeseries(x, method=m, window_size=100)
    ...     for m in ['ac1', 'variance', 'lambda_glsar']
    ... }
    >>> fig, axes = plot_multi_analysis(x, results, title='CSD indicators')
    >>> plt.show()
    """
    # Normalise input to ordered (label, result) pairs
    if isinstance(results, dict):
        items = list(results.items())
    else:
        items = [(r.method, r) for r in results]

    n_methods = len(items)
    if n_methods == 0:
        raise ValueError("'results' must contain at least one CSDResult.")

    x    = np.asarray(x, dtype=float)
    ts_kw  = ts_kwargs or {}
    ind_kw = indicator_kwargs or {}
    tc     = ts_color or _TS_COLOR
    fs     = figsize or (10, 2.4 * (n_methods + 1))

    # Time axes
    t_full = time_axis if time_axis is not None else np.arange(len(x))

    # Height ratios: top panel (ts) a bit taller than each indicator panel
    ratios = [1.6] + [1.0] * n_methods
    fig = plt.figure(figsize=fs, constrained_layout=True)
    gs  = GridSpec(n_methods + 1, 1, figure=fig, height_ratios=ratios)

    axes = []

    # --- Time series panel ---
    ax_ts = fig.add_subplot(gs[0])
    ax_ts.plot(t_full, x, color=tc, lw=0.8, **ts_kw)
    ax_ts.set_ylabel(ts_label, fontsize=9)
    ax_ts.tick_params(labelsize=8)
    ax_ts.xaxis.set_tick_params(labelbottom=False)
    axes.append(ax_ts)

    # --- Indicator panels ---
    for k, (label, result) in enumerate(items):
        ax = fig.add_subplot(gs[k + 1], sharex=ax_ts)

        t_ind = (
            time_axis[result.times.astype(int)]
            if time_axis is not None
            else result.times
        )
        color = METHOD_COLORS.get(result.method, "#1565C0")

        _draw_indicator_panel(
            ax, result, t_ind, color,
            show_trend, show_significance, ind_kw, stats_loc,
            trend_line_method,
        )

        # Use the dict key as the y-label when it differs from the method name
        if label != result.method:
            ax.set_ylabel(label, fontsize=9)

        if k < n_methods - 1:
            ax.xaxis.set_tick_params(labelbottom=False)
        else:
            ax.set_xlabel(x_label, fontsize=9)

        axes.append(ax)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="semibold")

    return fig, np.array(axes)


def plot_significance(
    result,
    trend_method: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4),
    title: Optional[str] = None,
    bins: int = 40,
    surrogate_color: Optional[str] = None,
    observed_color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Histogram of the Fourier surrogate null distribution of a trend statistic.

    The observed trend statistic is marked with a vertical line. The
    rejection tail(s) follow the stored trend result's surrogate alternative
    and the surrogate p-value is shown in a stats box.

    Requires ``significance=True`` to have been passed to
    :func:`~csdtools.analyze_timeseries`, which stores the null distribution
    in ``result.trends``.

    Parameters
    ----------
    result : CSDResult
        Result with a surrogate distribution stored for the selected method.
    trend_method : str or None
        Stored method to plot. This may be omitted when the result contains
        exactly one trend method; otherwise it is required.
    figsize : (float, float)
        Figure size (used only when ``ax`` is None).
    title : str or None
        Axes title.
    bins : int
        Number of histogram bins.
    surrogate_color : str or None
        Fill colour for the histogram bars.
    observed_color : str or None
        Colour for the observed-statistic vertical line.
    ax : plt.Axes or None
        If provided, draws onto this axes instead of creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If the trend method is ambiguous, unavailable, or has no surrogates.

    Examples
    --------
    >>> result = analyze_timeseries(x, method='lambda_glsar', window_size=100,
    ...                             significance=True, n_surrogates=1000)
    >>> fig, ax = plot_significance(result)
    >>> plt.show()
    """
    if trend_method is None:
        if len(result.trends) != 1:
            raise ValueError(
                "result contains multiple trend methods. Pass trend_method="
                f" explicitly; choose from {list(result.trends)}."
            )
        trend_method = next(iter(result.trends))
    elif trend_method not in result.trends:
        raise ValueError(
            f"Trend method '{trend_method}' is not stored in this result. "
            f"Available methods: {list(result.trends)}"
        )

    trend = result.trends[trend_method]
    observed = trend.statistic
    surrogate_statistics = trend.surrogate_statistics
    if surrogate_statistics is None:
        raise ValueError(
            f"result.trends['{trend_method}'].surrogate_statistics is None. "
            "Re-run analyze_timeseries() with significance=True."
        )

    surr = np.asarray(surrogate_statistics, dtype=float)
    surr = surr[np.isfinite(surr)]
    if len(surr) == 0:
        raise ValueError("The surrogate null distribution has no finite values.")
    sc     = surrogate_color or _SURR_COLOR
    oc     = observed_color  or METHOD_COLORS.get(result.method, "#D32F2F")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    # Histogram of null distribution
    ax.hist(
        surr, bins=bins,
        color=sc, edgecolor=_SURR_EDGE, linewidth=0.5,
        label=f"Surrogates (n={len(surr)})",
        zorder=2,
    )

    # Shade the rejection tail(s) selected for the surrogate test.
    counts, edges = np.histogram(surr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    alternative = trend.surrogate_alternative
    if alternative == "greater":
        tail_mask = centers >= observed
        tail_label = "Upper-tail region"
    elif alternative == "less":
        tail_mask = centers <= observed
        tail_label = "Lower-tail region"
    else:
        tail_mask = np.abs(centers) >= np.abs(observed)
        tail_label = "Two-sided region"
    if tail_mask.any():
        ax.bar(
            edges[:-1][tail_mask],
            counts[tail_mask],
            width=np.diff(edges)[tail_mask],
            align="edge",
            color=oc,
            alpha=0.45,
            zorder=3,
            label=tail_label,
        )

    # Observed statistic line
    statistic_label = trend_statistic_label(trend_method)
    ax.axvline(
        observed, color=oc, lw=2.2, zorder=5,
        label=(
            f"Observed {statistic_label} = "
            f"{format_trend_statistic(trend_method, observed)}"
        ),
    )

    # Null-distribution mean reference line
    ax.axvline(
        float(np.mean(surr)), color="#607D8B", lw=1.2, ls=":",
        zorder=4, label=f"Surrogate mean = {np.mean(surr):+.3f}",
    )

    # Stats annotation
    stars = _significance_stars(trend.surrogate_p_value)
    p_str = (
        f"p = {trend.surrogate_p_value:.4f}  {stars}"
        if trend.surrogate_p_value is not None
        else "p = n/a"
    )
    ax.text(
        0.98, 0.97, p_str,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#CFD8DC", alpha=0.9),
    )

    ax.set_xlabel(statistic_label, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8.5, loc="upper left")

    _title = title or (
        f"Surrogate null distribution — {_method_label(result.method)}"
    )
    ax.set_title(_title, fontsize=10)

    return fig, ax


def plot_indicator_on_ax(
    ax: plt.Axes,
    result,
    time_axis: Optional[np.ndarray] = None,
    show_trend: bool = True,
    show_significance: bool = True,
    color: Optional[str] = None,
    stats_loc: str = "upper left",
    trend_line_method: Optional[str] = None,
    **plot_kwargs,
) -> plt.Axes:
    """
    Draw a single CSD indicator onto an existing axes.

    Useful for embedding a CSD panel inside a larger custom figure layout.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    result : CSDResult
        Analysis result.
    time_axis : array-like or None
        Physical time values; falls back to sample indices.
    show_trend : bool
        Draw the selected fit or visual trend guide.
    show_significance : bool
        Draw stats box.
    color : str or None
        Indicator line colour.
    stats_loc : str
        Position of the stats box.
    trend_line_method : str or None
        Stored trend method whose fit is drawn.
    **plot_kwargs
        Extra kwargs for the indicator ``ax.plot`` call.

    Returns
    -------
    plt.Axes
        The same axes that was passed in.

    Examples
    --------
    >>> fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    >>> plot_indicator_on_ax(axes[0], result_ac1)
    >>> plot_indicator_on_ax(axes[1], result_var)
    >>> plot_indicator_on_ax(axes[2], result_glsar)
    >>> plt.tight_layout()
    """
    t_ind = (
        time_axis[result.times.astype(int)]
        if time_axis is not None
        else result.times
    )
    c = color or METHOD_COLORS.get(result.method, "#1565C0")
    _draw_indicator_panel(
        ax, result, t_ind, c,
        show_trend, show_significance, plot_kwargs, stats_loc,
        trend_line_method,
    )
    return ax
