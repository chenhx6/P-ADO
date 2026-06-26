from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

DELTA_COL = "delta"
SIGMA_I_COL = "sigma/I"
THETA_COL = "ArcTan[delta](AT)"

PDF_DELTA_SIGMA_COL = "pdf density in delta and sigma/I dimensions"
PDF_THETA_SIGMA_COL = "pdf density in AT and sigma/I dimensions"

P_COL = "p"
ADO_COL = "ado"
GAUS2D_COL = "gaus2d"

JAC_DELTA_SIGMA_COL = "jacobian(delta,sigma/I)"
JAC_THETA_SIGMA_COL = "jacobian(ArcTan[delta](AT),sigma/I)"

REQUIRED_COLUMNS = [
    DELTA_COL,
    SIGMA_I_COL,
    PDF_DELTA_SIGMA_COL,
    THETA_COL,
    PDF_THETA_SIGMA_COL,
    P_COL,
    ADO_COL,
    JAC_DELTA_SIGMA_COL,
    GAUS2D_COL,
]


RangeLike = tuple[float, float] | None


def _require_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def _as_1d_float_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _validate_range(value: RangeLike, *, name: str) -> RangeLike:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    lo, hi = float(value[0]), float(value[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{name} must contain finite values.")
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    return lo, hi


def _range_mask(coord: np.ndarray, value: RangeLike) -> np.ndarray:
    value = _validate_range(value, name="range")
    if value is None:
        return np.ones(coord.shape, dtype=bool)
    lo, hi = value
    return (coord >= lo) & (coord <= hi)


def _validate_1d_pdf_inputs(coord: Any, pdf: Any) -> tuple[np.ndarray, np.ndarray]:
    coord_arr = _as_1d_float_array(coord, name="coord")
    pdf_arr = _as_1d_float_array(pdf, name="pdf")
    if coord_arr.shape != pdf_arr.shape:
        raise ValueError(
            f"coord and pdf must have the same shape; got {coord_arr.shape} and {pdf_arr.shape}."
        )
    if coord_arr.size < 2:
        raise ValueError("At least two coordinate points are required.")
    order = np.argsort(coord_arr)
    coord_arr = coord_arr[order]
    pdf_arr = pdf_arr[order]
    if np.any(np.diff(coord_arr) <= 0):
        raise ValueError("coord values must be unique after sorting.")
    if np.any(pdf_arr < 0):
        raise ValueError("pdf must be non-negative.")
    return coord_arr, pdf_arr


def _trapz(values: np.ndarray, coord: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.trapz(values, coord, axis=axis)


def normalize_pdf(coord: Any, pdf: Any) -> np.ndarray:
    """Return a 1D PDF normalized to unit trapezoidal area."""
    coord_arr, pdf_arr = _validate_1d_pdf_inputs(coord, pdf)
    area = float(_trapz(pdf_arr, coord_arr))
    if not np.isfinite(area) or area <= 0:
        raise ValueError(f"Cannot normalize PDF with non-positive area: {area!r}.")
    return pdf_arr / area


def compute_cdf(coord: Any, pdf: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute a normalized CDF using trapezoidal bin areas."""
    coord_arr, pdf_arr = _validate_1d_pdf_inputs(coord, pdf)
    pdf_arr = normalize_pdf(coord_arr, pdf_arr)
    dx = np.diff(coord_arr)
    segment_area = 0.5 * (pdf_arr[:-1] + pdf_arr[1:]) * dx
    cdf = np.concatenate(([0.0], np.cumsum(segment_area)))
    final = float(cdf[-1])
    if final <= 0 or not np.isfinite(final):
        raise ValueError("CDF normalization failed because total mass is non-positive.")
    cdf = np.clip(cdf / final, 0.0, 1.0)
    cdf[-1] = 1.0
    return coord_arr, cdf


def compute_quantiles(
    coord: Any,
    pdf: Any,
    probs: tuple[float, ...] | list[float] | np.ndarray = (0.16, 0.50, 0.84),
) -> np.ndarray:
    """Compute equal-tail quantiles from a 1D PDF."""
    probs_arr = np.asarray(probs, dtype=float)
    if probs_arr.ndim != 1:
        raise ValueError("probs must be a 1D sequence.")
    if np.any((probs_arr < 0) | (probs_arr > 1)):
        raise ValueError("All probabilities must be in [0, 1].")
    cdf_x, cdf = compute_cdf(coord, pdf)
    unique_cdf, unique_idx = np.unique(cdf, return_index=True)
    unique_x = cdf_x[unique_idx]
    return np.interp(probs_arr, unique_cdf, unique_x)


def compute_mean_std(coord: Any, pdf: Any) -> tuple[float, float]:
    """Return the mean and standard deviation of a 1D PDF."""
    coord_arr, pdf_arr = _validate_1d_pdf_inputs(coord, pdf)
    pdf_arr = normalize_pdf(coord_arr, pdf_arr)
    mean = float(_trapz(coord_arr * pdf_arr, coord_arr))
    variance = float(_trapz((coord_arr - mean) ** 2 * pdf_arr, coord_arr))
    std = float(np.sqrt(max(variance, 0.0)))
    return mean, std


def compute_eq_summary(
    coord: Any,
    pdf: Any,
    *,
    quantile_probs: tuple[float, float, float] = (0.16, 0.50, 0.84),
) -> dict[str, float]:
    """Return the q16/q50/q84 equal-tail summary."""
    q16, q50, q84 = compute_quantiles(coord, pdf, quantile_probs)
    return {
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "center": float(q50),
        "err_plus": float(q84 - q50),
        "err_minus": float(q16 - q50),
    }


def _mass_above_threshold(
    coord: np.ndarray, pdf: np.ndarray, threshold: float
) -> float:
    mass = 0.0
    for x0, x1, y0, y1 in zip(coord[:-1], coord[1:], pdf[:-1], pdf[1:]):
        if y0 >= threshold and y1 >= threshold:
            mass += 0.5 * (y0 + y1) * (x1 - x0)
        elif y0 < threshold and y1 < threshold:
            continue
        elif y0 == y1:
            if y0 >= threshold:
                mass += y0 * (x1 - x0)
        else:
            frac = (threshold - y0) / (y1 - y0)
            xc = x0 + frac * (x1 - x0)
            if y0 >= threshold > y1:
                mass += 0.5 * (y0 + threshold) * (xc - x0)
            elif y0 < threshold <= y1:
                mass += 0.5 * (threshold + y1) * (x1 - xc)
    return float(mass)


def _intervals_above_threshold(
    coord: np.ndarray,
    pdf: np.ndarray,
    threshold: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for x0, x1, y0, y1 in zip(coord[:-1], coord[1:], pdf[:-1], pdf[1:]):
        if y0 >= threshold and y1 >= threshold:
            intervals.append((float(x0), float(x1)))
        elif y0 < threshold and y1 < threshold:
            continue
        elif y0 == y1:
            if y0 >= threshold:
                intervals.append((float(x0), float(x1)))
        else:
            frac = (threshold - y0) / (y1 - y0)
            xc = float(x0 + frac * (x1 - x0))
            if y0 >= threshold > y1:
                intervals.append((float(x0), xc))
            elif y0 < threshold <= y1:
                intervals.append((xc, float(x1)))

    merged: list[tuple[float, float]] = []
    tol = 1e-12 * max(1.0, float(np.ptp(coord)))
    for left, right in intervals:
        if right < left:
            left, right = right, left
        if right - left <= tol:
            continue
        if not merged or left > merged[-1][1] + tol:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def _integrate_pdf_between(
    coord: np.ndarray,
    pdf: np.ndarray,
    left: float,
    right: float,
) -> float:
    """Integrate a normalized 1D PDF between arbitrary left/right boundaries."""
    coord = np.asarray(coord, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    if coord.ndim != 1 or pdf.ndim != 1:
        raise ValueError("coord and pdf must be 1D arrays.")
    if coord.shape != pdf.shape:
        raise ValueError(
            f"coord and pdf must have the same shape; got {coord.shape} and {pdf.shape}."
        )
    if coord.size < 2:
        raise ValueError("At least two coordinate points are required.")
    if np.any(np.diff(coord) <= 0):
        raise ValueError("coord must be strictly increasing.")

    left = float(left)
    right = float(right)
    if right < left:
        left, right = right, left
    left = max(left, float(coord[0]))
    right = min(right, float(coord[-1]))
    if right <= left:
        return 0.0

    inside = (coord > left) & (coord < right)
    x_points = np.concatenate(([left], coord[inside], [right]))
    y_points = np.interp(x_points, coord, pdf)
    return float(_trapz(y_points, x_points))


def compute_hdi(
    coord: Any,
    pdf: Any,
    *,
    hdi_mass: float = 0.68,
) -> dict[str, Any]:
    """Return a density-threshold HDI summary that may contain many intervals."""
    if not (0 < hdi_mass <= 1):
        raise ValueError("hdi_mass must be in (0, 1].")
    coord_arr, pdf_arr = _validate_1d_pdf_inputs(coord, pdf)
    pdf_arr = normalize_pdf(coord_arr, pdf_arr)
    peak = float(np.max(pdf_arr))
    if peak <= 0:
        raise ValueError("Cannot compute HDI for a zero PDF.")

    if hdi_mass >= 1.0:
        threshold = 0.0
    else:
        lo, hi = 0.0, peak
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            mass = _mass_above_threshold(coord_arr, pdf_arr, mid)
            if mass >= hdi_mass:
                lo = mid
            else:
                hi = mid
        threshold = lo

    raw_intervals = _intervals_above_threshold(coord_arr, pdf_arr, threshold)
    if not raw_intervals:
        max_idx = int(np.argmax(pdf_arr))
        x0 = float(coord_arr[max_idx])
        raw_intervals = [(x0, x0)]

    interval_summaries: list[dict[str, float | int | str]] = []
    for idx, (left, right) in enumerate(raw_intervals, start=1):
        inside = (coord_arr >= left) & (coord_arr <= right)
        if np.any(inside):
            local_pdf = pdf_arr[inside]
            local_coord = coord_arr[inside]
            local_idx = int(np.argmax(local_pdf))
            mode = float(local_coord[local_idx])
            peak_density = float(local_pdf[local_idx])
        else:
            mode = float(0.5 * (left + right))
            peak_density = float(threshold)
        interval_mass = _integrate_pdf_between(coord_arr, pdf_arr, left, right)
        interval_summaries.append(
            {
                "name": f"interval{idx}",
                "left": float(left),
                "right": float(right),
                "width": float(right - left),
                "mass": interval_mass,
                "mass_fraction": 0.0,
                "mass_rank": 0,
                "mode": mode,
                "err_minus": float(left - mode),
                "err_plus": float(right - mode),
                "peak_density": peak_density,
            }
        )

    total_interval_mass = float(
        sum(float(interval["mass"]) for interval in interval_summaries)
    )
    ranked_indices = sorted(
        range(len(interval_summaries)),
        key=lambda i: (-float(interval_summaries[i]["mass"]), i),
    )
    ranks = {interval_idx: rank for rank, interval_idx in enumerate(ranked_indices, start=1)}
    for idx, interval in enumerate(interval_summaries):
        interval["mass_fraction"] = (
            float(interval["mass"]) / total_interval_mass
            if total_interval_mass > 0
            else 0.0
        )
        interval["mass_rank"] = ranks[idx]

    global_mode_idx = int(np.argmax(pdf_arr))
    return {
        "mass": float(hdi_mass),
        "density_threshold": float(threshold),
        "intervals": interval_summaries,
        "global_mode": float(coord_arr[global_mode_idx]),
    }


def compute_hdi_summary(
    coord: Any,
    pdf: Any,
    *,
    hdi_mass: float = 0.68,
) -> dict[str, Any]:
    """Alias for compute_hdi kept as a named public helper."""
    return compute_hdi(coord, pdf, hdi_mass=hdi_mass)


def summarize_1d(
    x: Any,
    pdf: Any,
    *,
    hdi_mass: float = 0.68,
    quantile_probs: tuple[float, float, float] = (0.16, 0.50, 0.84),
) -> dict[str, Any]:
    """Compute mean/std, mode, equal-tail quantiles, HDI, and CDF arrays."""
    coord_arr, pdf_arr = _validate_1d_pdf_inputs(x, pdf)
    pdf_arr = normalize_pdf(coord_arr, pdf_arr)
    mean, std = compute_mean_std(coord_arr, pdf_arr)
    eq = compute_eq_summary(coord_arr, pdf_arr, quantile_probs=quantile_probs)
    hdi = compute_hdi_summary(coord_arr, pdf_arr, hdi_mass=hdi_mass)
    cdf_x, cdf = compute_cdf(coord_arr, pdf_arr)
    mode = float(coord_arr[int(np.argmax(pdf_arr))])
    return {
        "mean": mean,
        "std": std,
        "mode": mode,
        "eq": eq,
        "hdi": hdi,
        "cdf_x": cdf_x,
        "cdf": cdf,
    }


def _validate_rectilinear_density(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Rectilinear density x and y must be 1D arrays.")
    if z.ndim != 2:
        raise ValueError("Rectilinear density z must be a 2D array.")
    if z.shape != (y.size, x.size):
        raise ValueError(
            "Rectilinear density must satisfy z.shape == (len(y), len(x)); "
            f"got z.shape={z.shape}, len(y)={y.size}, len(x)={x.size}."
        )
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("Rectilinear density coordinates must be strictly increasing.")
    if not np.all(np.isfinite(z)):
        raise ValueError("Density z contains NaN or infinite values.")


def _normalize_density2d_rectilinear(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    _validate_rectilinear_density(x, y, z)
    integral = float(_trapz(_trapz(z, x, axis=1), y))
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError(
            f"Cannot normalize 2D density with non-positive area: {integral!r}."
        )
    return z / integral


def _pivot_grid(
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    zcol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_columns(df, [xcol, ycol, zcol])
    duplicated = df.duplicated(subset=[xcol, ycol], keep=False)
    if bool(duplicated.any()):
        examples = df.loc[duplicated, [xcol, ycol]].head(5).to_dict("records")
        raise ValueError(
            f"Duplicate grid point(s) for ({xcol}, {ycol}); examples: {examples}"
        )
    x = np.sort(df[xcol].to_numpy(dtype=float))
    y = np.sort(df[ycol].to_numpy(dtype=float))
    x = np.unique(x)
    y = np.unique(y)
    if x.size * y.size != len(df):
        raise ValueError(
            f"Incomplete rectilinear grid for ({xcol}, {ycol}): "
            f"{len(df)} rows but {x.size} x {y.size} grid points are expected."
        )
    pivot = df.pivot(index=ycol, columns=xcol, values=zcol).reindex(index=y, columns=x)
    z = pivot.to_numpy(dtype=float)
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"Unexpected pivot shape {z.shape}; expected ({y.size}, {x.size})."
        )
    if np.isnan(z).any():
        raise ValueError(
            f"Incomplete grid for ({xcol}, {ycol}); pivot contains NaN values."
        )
    if not np.all(np.isfinite(z)):
        raise ValueError(f"Grid for {zcol!r} contains NaN or infinite values.")
    return x, y, z


def _pivot_on_base_grid(
    df: pd.DataFrame,
    *,
    base_xcol: str,
    base_ycol: str,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _pivot_grid(df, xcol=base_xcol, ycol=base_ycol, zcol=value_col)


def _build_weight(
    coord: np.ndarray,
    *,
    weight_kind: Literal["constant", "gaussian"],
    weight_mu: float | None,
    weight_sigma: float | None,
) -> np.ndarray:
    if weight_kind == "constant":
        return np.ones(coord.shape, dtype=float)
    if weight_kind == "gaussian":
        if weight_mu is None:
            raise ValueError("weight_mu is required for gaussian weighting.")
        if weight_sigma is None:
            raise ValueError("weight_sigma is required for gaussian weighting.")
        weight_sigma = float(weight_sigma)
        if not np.isfinite(weight_sigma) or weight_sigma <= 0:
            raise ValueError("weight_sigma must be a positive finite value.")
        weight_mu = float(weight_mu)
        if not np.isfinite(weight_mu):
            raise ValueError("weight_mu must be finite.")
        return np.exp(-0.5 * ((coord - weight_mu) / weight_sigma) ** 2)
    raise ValueError(f"Unsupported weight_kind: {weight_kind!r}.")


def _format_optional_range(value: RangeLike, *, full_text: str = "full") -> str:
    if value is None:
        return full_text
    return f"[{value[0]:.12g}, {value[1]:.12g}]"


def _project_with_optional_single_point(
    z: np.ndarray,
    coord: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    if coord.size == 1:
        return np.take(z, indices=0, axis=axis)
    return _trapz(z, coord, axis=axis)


@dataclass
class Projection1D:
    coord: np.ndarray
    pdf: np.ndarray
    coord_col: str
    source_name: str = ""
    target_axis: str = "x"
    integrated_axis: str = "y"
    weight_kind: str = "constant"
    weight_coord: np.ndarray | None = None
    weight_values: np.ndarray | None = None
    weight_range: tuple[float, float] | None = None
    summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({self.coord_col: self.coord, "pdf": self.pdf})

    def weight_to_dataframe(self) -> pd.DataFrame:
        if self.weight_coord is None or self.weight_values is None:
            return pd.DataFrame(columns=["weight_coord", "weight"])
        return pd.DataFrame(
            {"weight_coord": self.weight_coord, "weight": self.weight_values}
        )

    def format_summary(self, digits: int = 6, *, show_global_mode: bool = False) -> str:
        if self.summary is None:
            return "No summary is available."
        summary = self.summary
        fmt = f".{digits}g"
        weight_mu = self.metadata.get("gaus_mu", self.metadata.get("weight_mu"))
        weight_sigma = self.metadata.get(
            "gaus_sigma", self.metadata.get("weight_sigma")
        )
        weight_text = f"weight: {self.weight_kind}"
        if self.weight_kind == "gaussian":
            weight_text += (
                f", gaus_mu={weight_mu:{fmt}}, gaus_sigma={weight_sigma:{fmt}}"
            )
        weight_text += f", range = {_format_optional_range(self.weight_range)}"
        lines = [
            f"source: {self.source_name or self.coord_col}",
            f"quantity: {self.coord_col}",
            weight_text,
            f"mean (expectation value) = {summary['mean']:{fmt}}",
            f"std (standard deviation) = {summary['std']:{fmt}}",
        ]
        if show_global_mode:
            lines.append(f"global mode = {summary['mode']:{fmt}}")
        eq = summary.get("eq")
        if eq:
            lines.append(
                "EQ 68%: "
                f"q16 = {eq['q16']:{fmt}}, "
                f"q50 = {eq['q50']:{fmt}}, "
                f"q84 = {eq['q84']:{fmt}}, "
                f"err = ({eq['err_minus']:{fmt}}, {eq['err_plus']:+{fmt}})"
            )
        hdi = summary.get("hdi")
        if hdi:
            percent = 100 * float(hdi.get("mass", 0.68))
            lines.append(f"HDI {percent:.3g}%:")
            for interval in hdi.get("intervals", []):
                lines.append(
                    f"  {interval['name']}: "
                    f"[{interval['left']:{fmt}}, {interval['right']:{fmt}}], "
                    f"local mode = {interval['mode']:{fmt}}, "
                    f"err = ({interval['err_minus']:{fmt}}, {interval['err_plus']:+{fmt}}), "
                    f"width = {interval['width']:{fmt}}, "
                    f"mass = {interval['mass']:{fmt}}, "
                    f"mass_fraction = {interval['mass_fraction']:{fmt}}, "
                    f"mass_rank = {interval['mass_rank']}"
                )
        return "\n".join(lines)


@dataclass
class Density2D:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    xcol: str
    ycol: str
    zcol: str
    name: str = ""
    mesh_kind: str = "rectilinear"
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def project(
        self,
        target_axis: Literal["x", "y"] = "x",
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        weight_kind: Literal["constant", "gaussian"] = "constant",
        weight_mu: float | None = None,
        weight_sigma: float | None = None,
        gaus_mu: float | None = None,
        gaus_sigma: float | None = None,
        weight_range: tuple[float, float] | None = None,
        normalize: bool = True,
        evaluate: bool = True,
        hdi_mass: float = 0.68,
    ) -> Projection1D:
        if self.mesh_kind != "rectilinear":
            raise NotImplementedError(
                "Projection is defined only for rectilinear densities; "
                f"got mesh_kind={self.mesh_kind!r}."
            )
        x = _as_1d_float_array(self.x, name="x")
        y = _as_1d_float_array(self.y, name="y")
        z = np.asarray(self.z, dtype=float)
        _validate_rectilinear_density(x, y, z)
        if gaus_mu is not None:
            if weight_mu is not None and float(weight_mu) != float(gaus_mu):
                raise ValueError(
                    "Use either weight_mu or gaus_mu, not conflicting values."
                )
            weight_mu = gaus_mu
        if gaus_sigma is not None:
            if weight_sigma is not None and float(weight_sigma) != float(gaus_sigma):
                raise ValueError(
                    "Use either weight_sigma or gaus_sigma, not conflicting values."
                )
            weight_sigma = gaus_sigma
        x_range = _validate_range(x_range, name="x_range")
        y_range = _validate_range(y_range, name="y_range")
        weight_range = _validate_range(weight_range, name="weight_range")

        x_mask = _range_mask(x, x_range)
        y_mask = _range_mask(y, y_range)
        if target_axis == "x":
            y_mask &= _range_mask(y, weight_range)
        elif target_axis == "y":
            x_mask &= _range_mask(x, weight_range)
        else:
            raise ValueError("target_axis must be 'x' or 'y'.")
        if not np.any(x_mask):
            raise ValueError("x_range/weight_range selected no x coordinates.")
        if not np.any(y_mask):
            raise ValueError("y_range/weight_range selected no y coordinates.")

        x_sel = x[x_mask]
        y_sel = y[y_mask]
        z_sel = z[np.ix_(y_mask, x_mask)]

        if target_axis == "x":
            weight_coord = y_sel
            weights = _build_weight(
                weight_coord,
                weight_kind=weight_kind,
                weight_mu=weight_mu,
                weight_sigma=weight_sigma,
            )
            weighted = z_sel * weights[:, None]
            projected = _project_with_optional_single_point(
                weighted, weight_coord, axis=0
            )
            coord = x_sel
            coord_col = self.xcol
            integrated_axis = "y"
        else:
            weight_coord = x_sel
            weights = _build_weight(
                weight_coord,
                weight_kind=weight_kind,
                weight_mu=weight_mu,
                weight_sigma=weight_sigma,
            )
            weighted = z_sel * weights[None, :]
            projected = _project_with_optional_single_point(
                weighted, weight_coord, axis=1
            )
            coord = y_sel
            coord_col = self.ycol
            integrated_axis = "x"

        if np.any(projected < 0):
            min_value = float(np.min(projected))
            raise ValueError(
                f"Projected PDF contains negative values; minimum={min_value}."
            )
        pdf = (
            normalize_pdf(coord, projected)
            if normalize
            else np.asarray(projected, dtype=float)
        )
        summary = summarize_1d(coord, pdf, hdi_mass=hdi_mass) if evaluate else None
        return Projection1D(
            coord=coord,
            pdf=pdf,
            coord_col=coord_col,
            source_name=self.name,
            target_axis=target_axis,
            integrated_axis=integrated_axis,
            weight_kind=weight_kind,
            weight_coord=weight_coord,
            weight_values=weights,
            weight_range=weight_range,
            summary=summary,
            metadata={
                "x_range": x_range,
                "y_range": y_range,
                "weight_mu": weight_mu,
                "weight_sigma": weight_sigma,
                "gaus_mu": weight_mu,
                "gaus_sigma": weight_sigma,
                "normalize": normalize,
                **self.metadata,
            },
        )


def load_transition_csv(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Load a P-ADO transition CSV and optionally validate required columns."""
    path = Path(path)
    df = pd.read_csv(path)
    if validate:
        _require_columns(df, REQUIRED_COLUMNS)
    return df


class TransitionData:
    def __init__(
        self,
        path_or_df: str | Path | pd.DataFrame,
        *,
        name: str | None = None,
        validate: bool = True,
    ) -> None:
        self.path: Path | None = None
        if isinstance(path_or_df, pd.DataFrame):
            self.df = path_or_df.copy()
            self.name = name or "transition"
        else:
            self.path = Path(path_or_df)
            self.df = load_transition_csv(self.path, validate=validate)
            self.name = name or self.path.stem
        if validate:
            _require_columns(self.df, REQUIRED_COLUMNS)

    def col(self, name: str) -> np.ndarray:
        _require_columns(self.df, [name])
        return self.df[name].to_numpy(dtype=float)

    def has_column(self, name: str) -> bool:
        return name in self.df.columns

    def optional_col(self, name: str) -> np.ndarray | None:
        if not self.has_column(name):
            return None
        return self.col(name)

    def text_col(self, name: str) -> pd.Series:
        _require_columns(self.df, [name])
        return self.df[name].astype("string")

    def make_rectilinear_density(
        self,
        xcol: str,
        ycol: str,
        zcol: str,
        *,
        name: str = "",
        normalize: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
    ) -> Density2D:
        x, y, z = _pivot_grid(self.df, xcol=xcol, ycol=ycol, zcol=zcol)
        if normalize:
            z = _normalize_density2d_rectilinear(x, y, z)
        return Density2D(
            x=x,
            y=y,
            z=z,
            xcol=xcol,
            ycol=ycol,
            zcol=zcol,
            name=name or zcol,
            mesh_kind="rectilinear",
            xlabel=xlabel if xlabel is not None else xcol,
            ylabel=ylabel if ylabel is not None else ycol,
            zlabel=zlabel if zlabel is not None else zcol,
            metadata={
                "transition": self.name,
                "path": str(self.path) if self.path is not None else None,
                "normalized": normalize,
            },
        )

    def make_delta_sigma_field(
        self,
        value_col: str,
        *,
        name: str = "",
        normalize: bool = False,
        zlabel: str | None = None,
    ) -> Density2D:
        return self.make_rectilinear_density(
            xcol=DELTA_COL,
            ycol=SIGMA_I_COL,
            zcol=value_col,
            name=name,
            normalize=normalize,
            xlabel=r"$\delta$",
            ylabel=r"$\sigma/I$",
            zlabel=zlabel,
        )

    def make_curvilinear_density(
        self,
        xcol: str,
        ycol: str,
        zcol: str,
        *,
        base_xcol: str = "delta",
        base_ycol: str = "sigma/I",
        name: str = "",
        normalize: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
    ) -> Density2D:
        base_x, base_y, x2d = _pivot_on_base_grid(
            self.df, base_xcol=base_xcol, base_ycol=base_ycol, value_col=xcol
        )
        _, _, y2d = _pivot_on_base_grid(
            self.df, base_xcol=base_xcol, base_ycol=base_ycol, value_col=ycol
        )
        _, _, z2d = _pivot_on_base_grid(
            self.df, base_xcol=base_xcol, base_ycol=base_ycol, value_col=zcol
        )
        if x2d.shape != y2d.shape or x2d.shape != z2d.shape:
            raise ValueError("Curvilinear x/y/z arrays must have identical shapes.")
        if normalize:
            z2d = _normalize_density2d_rectilinear(base_x, base_y, z2d)
        return Density2D(
            x=x2d,
            y=y2d,
            z=z2d,
            xcol=xcol,
            ycol=ycol,
            zcol=zcol,
            name=name or zcol,
            mesh_kind="curvilinear",
            xlabel=xlabel if xlabel is not None else xcol,
            ylabel=ylabel if ylabel is not None else ycol,
            zlabel=zlabel if zlabel is not None else zcol,
            metadata={
                "transition": self.name,
                "path": str(self.path) if self.path is not None else None,
                "base_xcol": base_xcol,
                "base_ycol": base_ycol,
                "normalized": normalize,
            },
        )

    def make_delta_sigma_density(self, *, normalize: bool = False) -> Density2D:
        return self.make_rectilinear_density(
            xcol=DELTA_COL,
            ycol=SIGMA_I_COL,
            zcol=PDF_DELTA_SIGMA_COL,
            name="delta_sigma",
            normalize=normalize,
            xlabel=r"$\delta$",
            ylabel=r"$\sigma/I$",
            zlabel="Probability density",
        )

    def make_theta_sigma_density(self, *, normalize: bool = False) -> Density2D:
        return self.make_rectilinear_density(
            xcol=THETA_COL,
            ycol=SIGMA_I_COL,
            zcol=PDF_THETA_SIGMA_COL,
            name="theta_sigma",
            normalize=normalize,
            xlabel=r"$\arctan(\delta)\ (^\circ)$",
            ylabel=r"$\sigma/I$",
            zlabel="Probability density",
        )

    def make_jacobian_delta_sigma_field(self) -> Density2D:
        return self.make_delta_sigma_field(
            JAC_DELTA_SIGMA_COL,
            name="jacobian_delta_sigma",
            normalize=False,
            zlabel=JAC_DELTA_SIGMA_COL,
        )

    def make_jacobian_theta_sigma_field(self) -> Density2D:
        return self.make_delta_sigma_field(
            JAC_THETA_SIGMA_COL,
            name="jacobian_theta_sigma",
            normalize=False,
            zlabel=JAC_THETA_SIGMA_COL,
        )

    def make_pado_density(
        self, *, name: str = "pado", normalize: bool = False
    ) -> Density2D:
        return self.make_curvilinear_density(
            xcol=ADO_COL,
            ycol=P_COL,
            zcol=GAUS2D_COL,
            base_xcol=DELTA_COL,
            base_ycol=SIGMA_I_COL,
            name=name,
            normalize=normalize,
            xlabel=r"$R_\mathrm{ADO}$",
            ylabel=r"$P$",
            zlabel="gaus2d",
        )


def _range_to_text(value: RangeLike) -> str:
    if value is None:
        return "full"
    return _format_optional_range(value)


def _range_left(value: RangeLike) -> float | None:
    return None if value is None else float(value[0])


def _range_right(value: RangeLike) -> float | None:
    return None if value is None else float(value[1])


def _gate_type(weight_kind: str, weight_range: RangeLike) -> str:
    suffix = "gate" if weight_range is not None else "full"
    if weight_kind == "constant":
        return f"constant_{suffix}"
    if weight_kind == "gaussian":
        return f"gaussian_{suffix}"
    return f"{weight_kind}_{suffix}"


def _format_hdi_interval_list(
    intervals: list[dict[str, Any]], *, digits: int = 6
) -> str:
    """Format HDI intervals as '[left, right],[left, right],...'."""
    pieces = []
    for interval in intervals:
        left = interval.get("left")
        right = interval.get("right")
        if left is None or right is None:
            continue
        pieces.append(f"[{left:.{digits}g}, {right:.{digits}g}]")
    return ",".join(pieces)


def _format_interval_field(
    intervals: list[dict[str, Any]], key: str, *, digits: int = 6
) -> str:
    pieces = []
    for interval in intervals:
        value = interval.get(key)
        if value is None:
            continue
        if key == "mass_rank":
            pieces.append(str(int(value)))
        else:
            pieces.append(f"{float(value):.{digits}g}")
    return ",".join(pieces)


def projection_summary_to_row(
    projection: Projection1D,
    *,
    transition_name: str = "",
    quantity: str = "",
    gate_label: str = "",
    label: str = "",
) -> dict[str, Any]:
    """Flatten a Projection1D summary into a row suitable for a CSV table."""
    summary = projection.summary or summarize_1d(projection.coord, projection.pdf)
    eq = summary.get("eq", {})
    hdi = summary.get("hdi", {})
    intervals = hdi.get("intervals", [])
    std = summary.get("std")
    x_range = projection.metadata.get("x_range")
    y_range = projection.metadata.get("y_range")
    weight_mu = projection.metadata.get("weight_mu")
    weight_sigma = projection.metadata.get("weight_sigma")
    summary_text = projection.format_summary()
    row: dict[str, Any] = {
        "label": label or gate_label,
        "transition": transition_name,
        "quantity": quantity or projection.coord_col,
        "source_name": projection.source_name,
        "coord_col": projection.coord_col,
        "target_axis": projection.target_axis,
        "integrated_axis": projection.integrated_axis,
        "weight_kind": projection.weight_kind,
        "gate_type": _gate_type(projection.weight_kind, projection.weight_range),
        "gate_label": gate_label,
        "gate_range": _range_to_text(projection.weight_range),
        "gate_range_left": _range_left(projection.weight_range),
        "gate_range_right": _range_right(projection.weight_range),
        "weight_range": _range_to_text(projection.weight_range),
        "weight_range_left": _range_left(projection.weight_range),
        "weight_range_right": _range_right(projection.weight_range),
        "weight_mu": weight_mu,
        "weight_sigma": weight_sigma,
        "gaus_mu": projection.metadata.get("gaus_mu", weight_mu),
        "gaus_sigma": projection.metadata.get("gaus_sigma", weight_sigma),
        "x_range": _range_to_text(x_range),
        "y_range": _range_to_text(y_range),
        "mean": summary.get("mean"),
        "std": std,
        "mode": summary.get("mode"),
        "eq_q16": eq.get("q16"),
        "eq_q50": eq.get("q50"),
        "eq_q84": eq.get("q84"),
        "eq_err_minus": eq.get("err_minus"),
        "eq_err_plus": eq.get("err_plus"),
        "hdi_mass": hdi.get("mass"),
        "hdi_density_threshold": hdi.get("density_threshold"),
        "hdi_global_mode": hdi.get("global_mode"),
        "hdi_interval_count": len(intervals),
        "hdi_interval": _format_hdi_interval_list(intervals),
        "hdi_interval_mass": _format_interval_field(intervals, "mass"),
        "hdi_interval_width": _format_interval_field(intervals, "width"),
        "hdi_interval_mass_fraction": _format_interval_field(
            intervals, "mass_fraction"
        ),
        "hdi_interval_mass_rank": _format_interval_field(intervals, "mass_rank"),
        "summary_text": summary_text,
    }
    for idx in range(1, 6):
        prefix = f"hdi_interval{idx}"
        row[f"{prefix}_left"] = None
        row[f"{prefix}_mode"] = None
        row[f"{prefix}_right"] = None
        row[f"{prefix}_err_minus"] = None
        row[f"{prefix}_err_plus"] = None
        row[f"{prefix}_width"] = None
        row[f"{prefix}_mass"] = None
        row[f"{prefix}_mass_fraction"] = None
        row[f"{prefix}_mass_rank"] = None
        row[f"{prefix}_peak_density"] = None
    for idx, interval in enumerate(intervals[:5], start=1):
        prefix = f"hdi_interval{idx}"
        row[f"{prefix}_left"] = interval.get("left")
        row[f"{prefix}_mode"] = interval.get("mode")
        row[f"{prefix}_right"] = interval.get("right")
        row[f"{prefix}_err_minus"] = interval.get("err_minus")
        row[f"{prefix}_err_plus"] = interval.get("err_plus")
        row[f"{prefix}_width"] = interval.get("width")
        row[f"{prefix}_mass"] = interval.get("mass")
        row[f"{prefix}_mass_fraction"] = interval.get("mass_fraction")
        row[f"{prefix}_mass_rank"] = interval.get("mass_rank")
        row[f"{prefix}_peak_density"] = interval.get("peak_density")
    return row


DEFAULT_SIGMA_GATES: dict[str, tuple[float, float] | None] = {
    "all": None,
    "s0.1": (0.09, 0.11),
    "s0.2": (0.19, 0.21),
    "s0.3": (0.29, 0.31),
    "s0.4": (0.39, 0.41),
    "s0.5": (0.49, 0.51),
    "s0.6": (0.59, 0.61),
    "s0.7": (0.69, 0.71),
    "s0.8": (0.79, 0.81),
}


def summarize_transition(
    csv_path: str | Path,
    *,
    sigma_gates: dict[str, tuple[float, float] | None] | None = None,
    gaussian_weights: list[dict[str, Any]] | None = None,
    hdi_mass: float = 0.68,
    quantities: tuple[str, ...] = ("delta",),
) -> pd.DataFrame:
    """Compute weighted-projection summaries for one transition CSV.

    By default, only the delta projection is evaluated. To also evaluate
    arctan(delta), pass quantities=("delta", "theta").
    """
    data = TransitionData(csv_path)

    allowed_quantities = {"delta", "theta"}
    invalid = [q for q in quantities if q not in allowed_quantities]
    if invalid:
        raise ValueError(
            f"Unsupported quantities: {invalid}. "
            "Allowed values are 'delta' and 'theta'."
        )

    density_map: dict[str, Density2D] = {}

    if "delta" in quantities:
        density_map["delta"] = data.make_delta_sigma_density(normalize=True)

    if "theta" in quantities:
        density_map["theta"] = data.make_theta_sigma_density(normalize=True)

    sigma_gates = DEFAULT_SIGMA_GATES if sigma_gates is None else sigma_gates
    gaussian_weights = [] if gaussian_weights is None else gaussian_weights

    rows: list[dict[str, Any]] = []

    for label, gate_range in sigma_gates.items():
        for quantity, density in density_map.items():
            projection = density.project(
                target_axis="x",
                weight_kind="constant",
                weight_range=gate_range,
                normalize=True,
                evaluate=True,
                hdi_mass=hdi_mass,
            )
            rows.append(
                projection_summary_to_row(
                    projection,
                    transition_name=data.name,
                    quantity=quantity,
                    gate_label=label,
                )
            )

    for config in gaussian_weights:
        label = str(config.get("label", "gaussian"))
        gaus_mu = config.get("gaus_mu", config.get("mu"))
        gaus_sigma = config.get("gaus_sigma", config.get("sigma"))
        weight_range = config.get("range")

        for quantity, density in density_map.items():
            projection = density.project(
                target_axis="x",
                weight_kind="gaussian",
                gaus_mu=gaus_mu,
                gaus_sigma=gaus_sigma,
                weight_range=weight_range,
                normalize=True,
                evaluate=True,
                hdi_mass=hdi_mass,
            )
            rows.append(
                projection_summary_to_row(
                    projection,
                    transition_name=data.name,
                    quantity=quantity,
                    gate_label=label,
                )
            )

    return pd.DataFrame(rows)
