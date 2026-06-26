from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def get_root_kbird_cmap() -> LinearSegmentedColormap:
    """Return a ROOT kBird-like blue-green-yellow colormap."""
    stops = [
        (0.2082, 0.1664, 0.5293),
        (0.0592, 0.3599, 0.8683),
        (0.0780, 0.5041, 0.8385),
        (0.0232, 0.6419, 0.7914),
        (0.1802, 0.7178, 0.6425),
        (0.5301, 0.7492, 0.4662),
        (0.8186, 0.7328, 0.3498),
        (0.9959, 0.7862, 0.1968),
    ]
    return LinearSegmentedColormap.from_list("root_kbird", stops, N=256)


def plot_density2d(
    density,
    *,
    ax=None,
    cmap="viridis",
    shading=None,
    add_colorbar=True,
    colorbar_label=None,
    rasterized=True,
    vmin=None,
    vmax=None,
    grid=False,
    **pcolormesh_kwargs,
):
    """Plot a rectilinear Density2D object with Axes.pcolormesh."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    mesh_kind = getattr(density, "mesh_kind", "rectilinear")
    if mesh_kind == "curvilinear":
        raise ValueError(
            "Curvilinear P-ADO density should be plotted with plot_pado_scatter(), "
            "because pcolormesh can incorrectly connect folded observable-space points."
        )

    z = np.asarray(density.z)
    if getattr(density, "mask", None) is not None:
        z = np.ma.array(z, mask=density.mask)

    if shading is None:
        shading = "auto"

    artist = ax.pcolormesh(
        density.x,
        density.y,
        z,
        cmap=cmap,
        shading=shading,
        rasterized=rasterized,
        vmin=vmin,
        vmax=vmax,
        **pcolormesh_kwargs,
    )
    ax.grid(visible=grid)
    xlabel = getattr(density, "xlabel", "") or getattr(density, "xcol", "")
    ylabel = getattr(density, "ylabel", "") or getattr(density, "ycol", "")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if add_colorbar:
        cbar = fig.colorbar(artist, ax=ax)
        label = colorbar_label
        if label is None:
            label = getattr(density, "zlabel", "") or getattr(density, "zcol", "")
        if label:
            cbar.set_label(label)

    return fig, ax, artist


def plot_pado_scatter(
    ado,
    p=None,
    gaus2d=None,
    *,
    ax=None,
    cmap="viridis",
    marker_size=0.05,
    alpha=1.0,
    add_colorbar=True,
    colorbar_label="gaus2d",
    rasterized=True,
    vmin=None,
    vmax=None,
    grid=False,
    max_points=None,
    random_state=0,
    **scatter_kwargs,
):
    """Plot P-ADO observable-space points without connecting them into cells."""
    if hasattr(ado, "x") and hasattr(ado, "y") and hasattr(ado, "z"):
        if p is not None or gaus2d is not None:
            raise ValueError(
                "When ado is a Density2D-like object, p and gaus2d must be omitted."
            )
        density = ado
        ado = density.x
        p = density.y
        gaus2d = density.z
    elif p is None or gaus2d is None:
        raise ValueError(
            "Pass either a P-ADO Density2D object or ado, p, and gaus2d arrays."
        )

    ado = np.asarray(ado, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    gaus2d = np.asarray(gaus2d, dtype=float).ravel()
    if ado.shape != p.shape or ado.shape != gaus2d.shape:
        raise ValueError("ado, p, and gaus2d must have matching shapes.")

    finite = np.isfinite(ado) & np.isfinite(p) & np.isfinite(gaus2d)
    if not np.any(finite):
        raise ValueError("No finite P-ADO points are available for plotting.")
    ado = ado[finite]
    p = p[finite]
    gaus2d = gaus2d[finite]

    if max_points is not None:
        max_points = int(max_points)
        if max_points < 1:
            raise ValueError("max_points must be a positive integer or None.")
        if ado.size > max_points:
            rng = np.random.default_rng(random_state)
            selected = rng.choice(ado.size, size=max_points, replace=False)
            ado = ado[selected]
            p = p[selected]
            gaus2d = gaus2d[selected]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    scatter_options = dict(scatter_kwargs)
    scatter_options.update(
        c=gaus2d,
        s=marker_size,
        cmap=cmap,
        alpha=alpha,
        linewidths=0,
        rasterized=rasterized,
        vmin=vmin,
        vmax=vmax,
    )
    artist = ax.scatter(ado, p, **scatter_options)
    ax.set_xlabel(r"$R_\mathrm{ADO}$")
    ax.set_ylabel(r"$P$")
    ax.grid(visible=grid)
    if add_colorbar:
        cbar = fig.colorbar(artist, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    return fig, ax, artist


def plot_projection1d(
    projection,
    *,
    ax=None,
    color="black",
    lw=1.8,
    grid=False,
    label=None,
    add_eq_lines=False,
    add_hdi_spans=False,
):
    """Plot a Projection1D object as a line."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    (line,) = ax.plot(projection.coord, projection.pdf, color=color, lw=lw, label=label)
    ax.set_xlabel(getattr(projection, "coord_col", "coord"))
    ax.set_ylabel("Probability density")

    summary = getattr(projection, "summary", None)
    if summary and add_hdi_spans:
        hdi = summary.get("hdi", {})
        for interval in hdi.get("intervals", []):
            ax.axvspan(
                interval["left"],
                interval["right"],
                color=line.get_color(),
                alpha=0.14,
                lw=0,
            )
        threshold = hdi.get("density_threshold")
        if threshold is not None:
            ax.axhline(
                threshold,
                color=line.get_color(),
                lw=1.0,
                ls=":",
                alpha=0.75,
            )
    ax.grid(visible=grid)

    if summary and add_eq_lines:
        eq = summary.get("eq", {})
        if eq:
            ax.axvline(eq["q16"], color=line.get_color(), lw=1.0, ls="--", alpha=0.55)
            ax.axvline(eq["q50"], color=line.get_color(), lw=1.2, ls="-", alpha=0.75)
            ax.axvline(eq["q84"], color=line.get_color(), lw=1.0, ls="--", alpha=0.55)

    if label:
        ax.legend(frameon=False)
    return fig, ax, line


def plot(obj, **kwargs):
    """Dispatch to the 2D-density or 1D-projection plotting helper."""
    if hasattr(obj, "mesh_kind") and hasattr(obj, "z"):
        if (
            getattr(obj, "xcol", "") == "ado"
            and getattr(obj, "ycol", "") == "p"
            and getattr(obj, "zcol", "") == "gaus2d"
        ):
            return plot_pado_scatter(obj, **kwargs)
        return plot_density2d(obj, **kwargs)
    if hasattr(obj, "coord") and hasattr(obj, "pdf"):
        return plot_projection1d(obj, **kwargs)
    raise TypeError("plot() expects an object like Density2D or Projection1D.")


def apply_style(
    ax,
    *,
    preset="journal",
    top=True,
    right=True,
    tick_direction="in",
    spine_width=1.2,
    major_tick_width=1.2,
    minor_tick_width=1.0,
    major_tick_length=6,
    minor_tick_length=3,
    label_size=14,
    tick_label_size=12,
    title_size=14,
    legend_size=10,
    font_family="Times New Roman",
    grid=False,
):
    """Apply axis-level publication styling without mutating global rcParams."""
    if preset not in ("journal", "presentation", None):
        raise ValueError("preset must be 'journal', 'presentation', or None.")
    if preset == "presentation":
        if top is True:
            top = False
        if right is True:
            right = False
        if tick_direction == "in":
            tick_direction = "out"
        label_size = max(label_size, 16)
        tick_label_size = max(tick_label_size, 13)
        title_size = max(title_size, 16)
        legend_size = max(legend_size, 11)

    for name, spine in ax.spines.items():
        spine.set_linewidth(spine_width)
        if name == "top":
            spine.set_visible(top)
        elif name == "right":
            spine.set_visible(right)
        else:
            spine.set_visible(True)

    ax.tick_params(
        axis="both",
        which="major",
        direction=tick_direction,
        top=top,
        right=right,
        width=major_tick_width,
        length=major_tick_length,
        labelsize=tick_label_size,
    )
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="minor",
        direction=tick_direction,
        top=top,
        right=right,
        width=minor_tick_width,
        length=minor_tick_length,
    )

    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(title_size)
    for text in [
        ax.xaxis.label,
        ax.yaxis.label,
        ax.title,
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
    ]:
        text.set_fontfamily(font_family)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)
            text.set_fontfamily(font_family)

    if grid:
        ax.grid(True, which="major", alpha=0.25)
    else:
        ax.grid(False)
    return ax


@contextmanager
def style_context(**rc_params):
    """Temporarily apply matplotlib rcParams for code that explicitly asks for it."""
    with mpl.rc_context(rc_params):
        yield


def save_figure(
    fig,
    path,
    *,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.05,
    rasterize_heavy=False,
) -> Path:
    """Create the parent directory, save a figure, and return the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rasterize_heavy:
        for ax in fig.axes:
            for collection in ax.collections:
                collection.set_rasterized(True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    return path
