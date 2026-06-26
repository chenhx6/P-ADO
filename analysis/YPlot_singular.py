from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize

from YAnalysis_singular import (
    DETJ_SIGNED_DELTA_SIGMA_COL,
    DETJ_SIGNED_THETA_SIGMA_COL,
    DETJ_SIGN_DELTA_SIGMA_COL,
    DETJ_SIGN_THETA_SIGMA_COL,
    THETA_COL,
    compute_hist2d,
    compute_sign_hist2d,
)


LABELS = {
    "ado": r"$R_{\mathrm{ADO}}$",
    "p": r"$P$",
    THETA_COL: r"$\arctan(\delta)\ (^\circ)$",
    "sigma/I": r"$\sigma/I$",
    "delta": r"$\delta$",
    "observable_density_rel_to_max": r"$f_{\mathcal{O}}/f_{\mathcal{O},\max}$",
    DETJ_SIGNED_DELTA_SIGMA_COL: r"$\det J_{\delta,\sigma/I}$",
    DETJ_SIGNED_THETA_SIGMA_COL: r"$\det J_{\arctan(\delta),\sigma/I}$",
}

SIGN_COLORS = {
    "plus": "red",
    "minus": "blue",
    "zero": "0.25",
    "nonfinite": "darkorange",
}


def _get_axes(ax=None):
    if ax is None:
        return plt.subplots()
    return ax.figure, ax


def _finish_axes(ax, xcol, ycol=None, *, xlabel=None, ylabel=None, title=None):
    ax.set_xlabel(xlabel if xlabel is not None else LABELS.get(xcol, xcol))
    if ycol is not None:
        ax.set_ylabel(ylabel if ylabel is not None else LABELS.get(ycol, ycol))
    if title:
        ax.set_title(title)


def _color_norm(values, norm):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if norm in ("log", "Log", "LOG"):
        if positive.size:
            vmin = float(positive.min())
            vmax = float(positive.max())
            if vmin == vmax:
                vmin = vmax / 10.0
            return LogNorm(vmin=vmin, vmax=vmax)
        return Normalize(vmin=0.0, vmax=1.0)
    if norm is None or norm == "linear":
        return None
    return norm


def default_sign_col_for_xcol(xcol: str) -> str:
    if xcol == THETA_COL:
        return DETJ_SIGN_THETA_SIGMA_COL
    return DETJ_SIGN_DELTA_SIGMA_COL


def _signed_col_for_sign_col(sign_col: str) -> str:
    if sign_col == DETJ_SIGN_THETA_SIGMA_COL:
        return DETJ_SIGNED_THETA_SIGMA_COL
    return DETJ_SIGNED_DELTA_SIGMA_COL


def _sign_values(series: pd.Series) -> np.ndarray:
    text = series.astype("string").str.strip().str.lower().fillna("")
    text_values = text.to_numpy(dtype=str)
    sign = pd.to_numeric(text, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    aliases = {
        "+": 1.0,
        "+1": 1.0,
        "plus": 1.0,
        "positive": 1.0,
        "pos": 1.0,
        "-": -1.0,
        "-1": -1.0,
        "minus": -1.0,
        "negative": -1.0,
        "neg": -1.0,
        "0": 0.0,
        "+0": 0.0,
        "-0": 0.0,
        "zero": 0.0,
    }
    for label, value in aliases.items():
        sign[text_values == label] = value
    finite = np.isfinite(sign)
    sign[finite] = np.sign(sign[finite])
    return sign


def _plot_binned(
    ax,
    fig,
    hist,
    xedges,
    yedges,
    *,
    norm="log",
    cmap="viridis",
    colorbar_label=None,
):
    values = np.asarray(hist, dtype=float)
    plot_values = np.ma.masked_where(~np.isfinite(values) | (values <= 0), values)
    artist = ax.pcolormesh(
        xedges,
        yedges,
        plot_values.T,
        shading="auto",
        cmap=cmap,
        norm=_color_norm(values, norm),
        rasterized=True,
    )
    colorbar = fig.colorbar(artist, ax=ax)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    return artist


def plot_hist2d_count(
    df,
    xcol,
    ycol,
    *,
    bins=300,
    range=None,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
    norm="log",
):
    fig, ax = _get_axes(ax)
    hist, xedges, yedges = compute_hist2d(
        df, xcol, ycol, bins=bins, range=range
    )
    artist = _plot_binned(
        ax,
        fig,
        hist,
        xedges,
        yedges,
        norm=norm,
        colorbar_label="Singular-point count",
    )
    _finish_axes(
        ax, xcol, ycol, xlabel=xlabel, ylabel=ylabel, title=title
    )
    return fig, ax, artist


def plot_hist2d_weighted(
    df,
    xcol,
    ycol,
    *,
    weight_col="observable_density_rel_to_max",
    bins=300,
    range=None,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
    norm="log",
):
    fig, ax = _get_axes(ax)
    hist, xedges, yedges = compute_hist2d(
        df, xcol, ycol, bins=bins, range=range, weights=weight_col
    )
    artist = _plot_binned(
        ax,
        fig,
        hist,
        xedges,
        yedges,
        norm=norm,
        colorbar_label=f"Sum of {LABELS.get(weight_col, weight_col)}",
    )
    _finish_axes(
        ax, xcol, ycol, xlabel=xlabel, ylabel=ylabel, title=title
    )
    return fig, ax, artist


def _scatter_sign_masks(df, sign_col):
    signed_col = _signed_col_for_sign_col(sign_col)
    if sign_col in df.columns:
        sign = _sign_values(df[sign_col])
    elif signed_col in df.columns:
        signed = pd.to_numeric(df[signed_col], errors="coerce").to_numpy(dtype=float)
        sign = np.sign(signed)
        sign[~np.isfinite(signed)] = np.nan
    else:
        warnings.warn(
            "No signed-detJ diagnostic column is available; points are shown as unknown.",
            RuntimeWarning,
            stacklevel=2,
        )
        sign = np.full(len(df), np.nan, dtype=float)
    nonfinite = ~np.isfinite(sign)
    if signed_col in df.columns:
        signed = pd.to_numeric(df[signed_col], errors="coerce").to_numpy(dtype=float)
        nonfinite |= ~np.isfinite(signed)
    return {
        "plus": (sign > 0) & ~nonfinite,
        "minus": (sign < 0) & ~nonfinite,
        "zero": (sign == 0) & ~nonfinite,
        "nonfinite": nonfinite,
    }


def plot_sign_scatter(
    df,
    xcol,
    ycol,
    *,
    sign_col=None,
    sample=300_000,
    random_state=0,
    s=0.5,
    alpha=0.12,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
    rasterized=True,
):
    fig, ax = _get_axes(ax)
    if sign_col is None:
        sign_col = default_sign_col_for_xcol(xcol)
    work = df
    if sample is not None and len(work) > int(sample):
        work = work.sample(n=int(sample), random_state=random_state)

    x = pd.to_numeric(work[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(work[ycol], errors="coerce").to_numpy(dtype=float)
    coordinate_valid = np.isfinite(x) & np.isfinite(y)
    masks = _scatter_sign_masks(work, sign_col)
    artists = {}
    labels = {
        "plus": "+1",
        "minus": "-1",
        "zero": "0",
        "nonfinite": "nonfinite/unknown",
    }
    for key in ("nonfinite", "zero", "minus", "plus"):
        selected = coordinate_valid & masks[key]
        if not selected.any():
            continue
        artists[key] = ax.scatter(
            x[selected],
            y[selected],
            s=s,
            alpha=alpha,
            c=SIGN_COLORS[key],
            linewidths=0,
            label=f"{labels[key]} ({selected.sum():,})",
            rasterized=rasterized,
        )
    if artists:
        ax.legend(markerscale=max(2.0, 4.0 / max(np.sqrt(s), 0.1)), frameon=False)
    _finish_axes(
        ax, xcol, ycol, xlabel=xlabel, ylabel=ylabel, title=title
    )
    return fig, ax, artists


def plot_dominant_sign_map(
    df,
    xcol,
    ycol,
    *,
    bins=220,
    range=None,
    sign_col=None,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
):
    fig, ax = _get_axes(ax)
    if sign_col is None:
        sign_col = default_sign_col_for_xcol(xcol)
    result = compute_sign_hist2d(
        df, xcol, ycol, bins=bins, range=range, sign_col=sign_col
    )
    stack = np.stack(
        [result["minus"], result["zero"], result["plus"], result["nonfinite"]]
    )
    dominant = np.argmax(stack, axis=0).astype(float)
    dominant[result["total"] <= 0] = np.nan
    cmap = ListedColormap(
        [SIGN_COLORS["minus"], SIGN_COLORS["zero"], SIGN_COLORS["plus"], SIGN_COLORS["nonfinite"]]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    artist = ax.pcolormesh(
        result["xedges"],
        result["yedges"],
        np.ma.masked_invalid(dominant).T,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    colorbar = fig.colorbar(artist, ax=ax, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["-1", "0", "+1", "nonfinite"])
    colorbar.set_label("Dominant signed-detJ class")
    _finish_axes(
        ax, xcol, ycol, xlabel=xlabel, ylabel=ylabel, title=title
    )
    return fig, ax, artist


def plot_mixed_sign_fraction(
    df,
    xcol,
    ycol,
    *,
    bins=220,
    range=None,
    sign_col=None,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
):
    fig, ax = _get_axes(ax)
    if sign_col is None:
        sign_col = default_sign_col_for_xcol(xcol)
    result = compute_sign_hist2d(
        df, xcol, ycol, bins=bins, range=range, sign_col=sign_col
    )
    values = np.ma.masked_where(result["total"] <= 0, result["mixed_fraction"])
    artist = ax.pcolormesh(
        result["xedges"],
        result["yedges"],
        values.T,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    colorbar = fig.colorbar(artist, ax=ax)
    colorbar.set_label(r"$2\min(n_+,n_-)/(n_+ + n_-)$")
    _finish_axes(
        ax, xcol, ycol, xlabel=xlabel, ylabel=ylabel, title=title
    )
    return fig, ax, artist


def plot_histogram(
    df,
    col,
    *,
    bins=100,
    range=None,
    logy=True,
    ax=None,
    xlabel=None,
    title=None,
):
    fig, ax = _get_axes(ax)
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        artist = ax.hist(values, bins=bins, range=range, color="0.25", alpha=0.85)
    else:
        empty_range = range if range is not None else (0.0, 1.0)
        artist = ax.hist([], bins=bins, range=empty_range, color="0.25", alpha=0.85)
        ax.text(
            0.5,
            0.5,
            "No finite values",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.set_xlabel(xlabel if xlabel is not None else LABELS.get(col, col))
    ax.set_ylabel("Count")
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    return fig, ax, artist


def add_observable_center(
    ax,
    *,
    ado=0.390,
    p=-0.120,
    ado_err=0.050,
    p_err_left=0.120,
    p_err_right=0.110,
):
    return ax.errorbar(
        ado,
        p,
        xerr=ado_err,
        yerr=np.array([[p_err_left], [p_err_right]]),
        fmt="o",
        ms=5,
        mfc="white",
        mec="black",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        label="Experimental center",
        zorder=20,
    )


def save_figure_pair(fig, outdir, stem):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    return png_path, pdf_path
