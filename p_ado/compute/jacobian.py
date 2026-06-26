import numpy as np

from ..config import (
    DENSITY_REL_THRESHOLDS,
    DETJ_NEAR_ZERO_THRESHOLD,
    TERMINAL_DENSITY_THRESHOLD,
)

DEFAULT_HIGH_DENSITY_REL_THRESHOLDS = DENSITY_REL_THRESHOLDS
CORE_DENSITY_REL_THRESHOLD = TERMINAL_DENSITY_THRESHOLD


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0.0:
        return np.nan
    return float(numerator / denominator)


def _build_sign_change_report(raw_det) -> dict:
    raw_det = np.asarray(raw_det, dtype=float)
    finite_nonzero = np.isfinite(raw_det) & (raw_det != 0.0)

    delta_pairs = finite_nonzero[1:, :] & finite_nonzero[:-1, :]
    sigma_pairs = finite_nonzero[:, 1:] & finite_nonzero[:, :-1]
    delta_changes = int(
        np.count_nonzero(
            delta_pairs & (np.signbit(raw_det[1:, :]) != np.signbit(raw_det[:-1, :]))
        )
    )
    sigma_changes = int(
        np.count_nonzero(
            sigma_pairs & (np.signbit(raw_det[:, 1:]) != np.signbit(raw_det[:, :-1]))
        )
    )
    return {
        "along_delta": delta_changes,
        "along_sigma_i": sigma_changes,
        "total": delta_changes + sigma_changes,
    }


def _build_density_threshold_report(
    raw_det,
    abs_det,
    center,
    obs_density,
    finite_det_mask,
    near_zero_threshold,
    relative_threshold,
    density_max,
    total_density_weight,
    transformed_density,
    total_transformed_density_weight,
):
    absolute_threshold = float(relative_threshold * density_max)
    mask = obs_density >= absolute_threshold
    active_count = int(np.count_nonzero(mask))
    total_count = int(raw_det.size)
    active_fraction = active_count / total_count if total_count else np.nan

    finite_obs_density = np.isfinite(obs_density)
    active_density_mask = mask & finite_obs_density
    density_weight_fraction = _safe_fraction(
        float(np.sum(obs_density[active_density_mask])),
        total_density_weight,
    )

    finite_mask = mask & finite_det_mask
    finite_abs = abs_det[finite_mask]
    finite_count = int(np.count_nonzero(finite_mask))
    nonfinite_count = active_count - finite_count
    zero_mask = finite_det_mask & (raw_det == 0.0)
    near_zero_mask = finite_det_mask & (abs_det <= near_zero_threshold)
    zero_count = int(np.count_nonzero(mask & zero_mask))
    near_zero_count = int(np.count_nonzero(mask & near_zero_mask))

    zero_density_weight_fraction = _safe_fraction(
        float(np.sum(obs_density[mask & zero_mask & finite_obs_density])),
        total_density_weight,
    )
    near_zero_density_weight_fraction = _safe_fraction(
        float(np.sum(obs_density[mask & near_zero_mask & finite_obs_density])),
        total_density_weight,
    )

    finite_transformed_density = np.isfinite(transformed_density)
    zero_transformed_density_fraction = _safe_fraction(
        float(
            np.sum(
                transformed_density[
                    mask & zero_mask & finite_transformed_density
                ]
            )
        ),
        total_transformed_density_weight,
    )
    near_zero_transformed_density_fraction = _safe_fraction(
        float(
            np.sum(
                transformed_density[
                    mask & near_zero_mask & finite_transformed_density
                ]
            )
        ),
        total_transformed_density_weight,
    )

    if finite_count:
        min_flat_index = int(np.flatnonzero(finite_mask)[np.argmin(finite_abs)])
        min_point = center.reshape(-1, center.shape[-1])[min_flat_index]
        min_location = {
            "delta": float(min_point[0]),
            "sigma/I": float(min_point[2]),
            "P": float(min_point[3]),
            "ADO": float(min_point[4]),
            "observable_density": float(obs_density.reshape(-1)[min_flat_index]),
        }
        abs_min = float(np.min(finite_abs))
        abs_p01 = float(np.percentile(finite_abs, 1.0))
        abs_median = float(np.median(finite_abs))
        abs_max = float(np.max(finite_abs))
    else:
        min_location = {
            "delta": np.nan,
            "sigma/I": np.nan,
            "P": np.nan,
            "ADO": np.nan,
            "observable_density": np.nan,
        }
        abs_min = np.nan
        abs_p01 = np.nan
        abs_median = np.nan
        abs_max = np.nan

    return {
        "relative_threshold": float(relative_threshold),
        "absolute_threshold": absolute_threshold,
        "active_count": active_count,
        "active_fraction": float(active_fraction),
        "density_weight_fraction": density_weight_fraction,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "zero_count": zero_count,
        "near_zero_count": near_zero_count,
        "zero_fraction": _safe_fraction(zero_count, active_count),
        "near_zero_fraction": _safe_fraction(near_zero_count, active_count),
        "zero_density_weight_fraction": zero_density_weight_fraction,
        "near_zero_density_weight_fraction": near_zero_density_weight_fraction,
        "zero_transformed_density_fraction": zero_transformed_density_fraction,
        "near_zero_transformed_density_fraction": (
            near_zero_transformed_density_fraction
        ),
        "abs_min": abs_min,
        "abs_p01": abs_p01,
        "abs_median": abs_median,
        "abs_max": abs_max,
        "min_location": min_location,
        "status": (
            "accepted"
            if nonfinite_count == 0 and zero_count == 0 and near_zero_count == 0
            else "warning"
        ),
    }


def build_jacobian_report(
    raw_det,
    abs_det,
    center,
    *,
    name,
    near_zero_threshold=DETJ_NEAR_ZERO_THRESHOLD,
    obs_density=None,
    high_density_rel_thresholds=DEFAULT_HIGH_DENSITY_REL_THRESHOLDS,
    high_density_rel_threshold=None,
):
    raw_det = np.asarray(raw_det, dtype=float)
    abs_det = np.asarray(abs_det, dtype=float)
    center = np.asarray(center, dtype=float)

    if raw_det.shape != abs_det.shape or center.shape[:-1] != raw_det.shape:
        raise ValueError("Jacobian determinant and center-grid shapes must match")

    finite_mask = np.isfinite(raw_det) & np.isfinite(abs_det)
    finite_raw = raw_det[finite_mask]
    finite_abs = abs_det[finite_mask]
    total_count = int(raw_det.size)
    finite_count = int(np.count_nonzero(finite_mask))
    nonfinite_count = total_count - finite_count
    zero_count = int(np.count_nonzero(finite_mask & (raw_det == 0.0)))
    near_zero_count = int(
        np.count_nonzero(finite_mask & (abs_det <= near_zero_threshold))
    )

    if finite_count:
        min_flat_index = int(np.flatnonzero(finite_mask)[np.argmin(finite_abs)])
        min_point = center.reshape(-1, center.shape[-1])[min_flat_index]
        abs_min = float(np.min(finite_abs))
        abs_p01 = float(np.percentile(finite_abs, 1.0))
        abs_median = float(np.median(finite_abs))
        abs_max = float(np.max(finite_abs))
        signed_min = float(np.min(finite_raw))
        signed_max = float(np.max(finite_raw))
        min_location = {
            "delta": float(min_point[0]),
            "sigma/I": float(min_point[2]),
            "P": float(min_point[3]),
            "ADO": float(min_point[4]),
        }
    else:
        abs_min = np.nan
        abs_p01 = np.nan
        abs_median = np.nan
        abs_max = np.nan
        signed_min = np.nan
        signed_max = np.nan
        min_location = {
            "delta": np.nan,
            "sigma/I": np.nan,
            "P": np.nan,
            "ADO": np.nan,
        }

    report = {
        "name": name,
        "total_count": total_count,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "zero_count": zero_count,
        "near_zero_threshold": float(near_zero_threshold),
        "near_zero_count": near_zero_count,
        "abs_min": abs_min,
        "abs_p01": abs_p01,
        "abs_median": abs_median,
        "abs_max": abs_max,
        "signed_min": signed_min,
        "signed_max": signed_max,
        "positive_count": int(np.count_nonzero(finite_raw > 0.0)),
        "negative_count": int(np.count_nonzero(finite_raw < 0.0)),
        "min_location": min_location,
        "sign_change": _build_sign_change_report(raw_det),
        "status": (
            "accepted" if nonfinite_count == 0 and zero_count == 0 else "warning"
        ),
    }

    if obs_density is not None:
        obs_density = np.asarray(obs_density, dtype=float)
        if obs_density.shape != raw_det.shape:
            raise ValueError("Observable density and determinant shapes must match")

        if high_density_rel_threshold is not None:
            high_density_rel_thresholds = (high_density_rel_threshold,)
        thresholds = tuple(float(value) for value in high_density_rel_thresholds)
        if np.any(~np.isnan(obs_density)):
            density_max = float(np.nanmax(obs_density))
        else:
            density_max = np.nan
        finite_obs_density = np.isfinite(obs_density)
        total_density_weight = float(np.sum(obs_density[finite_obs_density]))
        with np.errstate(invalid="ignore", over="ignore"):
            transformed_density = obs_density * abs_det
        finite_transformed_density = np.isfinite(transformed_density)
        total_transformed_density_weight = float(
            np.sum(transformed_density[finite_transformed_density])
        )
        report["high_density"] = {
            "density_source": "observable_density",
            "thresholds": [
                _build_density_threshold_report(
                    raw_det=raw_det,
                    abs_det=abs_det,
                    center=center,
                    obs_density=obs_density,
                    finite_det_mask=finite_mask,
                    near_zero_threshold=near_zero_threshold,
                    relative_threshold=threshold,
                    density_max=density_max,
                    total_density_weight=total_density_weight,
                    transformed_density=transformed_density,
                    total_transformed_density_weight=(
                        total_transformed_density_weight
                    ),
                )
                for threshold in thresholds
            ],
        }

    return report


def _format_report_value(value) -> str:
    return f"{float(value):.6g}"


def _escape_markdown_table_cell(value) -> str:
    return str(value).replace("|", "\\|")


def _find_density_threshold(report: dict, threshold: float):
    high_density = report.get("high_density", {})
    for entry in high_density.get("thresholds", []):
        if np.isclose(entry["relative_threshold"], threshold, rtol=0.0, atol=1e-15):
            return entry
    return None


def format_jacobian_report_text(report: dict) -> str:
    location = report["min_location"]
    lines = [
        "-" * 72,
        "Jacobian check",
        f"  determinant: {report['name']}",
        f"    total interior points = {report['total_count']}",
        f"    finite count = {report['finite_count']}",
        f"    non-finite count = {report['nonfinite_count']}",
        f"    zero count = {report['zero_count']}",
        f"    near-zero threshold = {report['near_zero_threshold']:.1e}",
        f"    near-zero count = {report['near_zero_count']}",
        f"    min |detJ| = {_format_report_value(report['abs_min'])}",
        f"    p01 |detJ| = {_format_report_value(report['abs_p01'])}",
        f"    median |detJ| = {_format_report_value(report['abs_median'])}",
        f"    max |detJ| = {_format_report_value(report['abs_max'])}",
        f"    signed min detJ = {_format_report_value(report['signed_min'])}",
        f"    signed max detJ = {_format_report_value(report['signed_max'])}",
        f"    positive count = {report['positive_count']}",
        f"    negative count = {report['negative_count']}",
        "    min location:",
        f"      delta = {_format_report_value(location['delta'])}",
        f"      sigma/I = {_format_report_value(location['sigma/I'])}",
        f"      P = {_format_report_value(location['P'])}",
        f"      ADO = {_format_report_value(location['ADO'])}",
        f"    status = {report['status']}",
    ]
    core_density = _find_density_threshold(report, CORE_DENSITY_REL_THRESHOLD)
    if core_density is not None:
        core_location = core_density["min_location"]
        lines.extend(
            [
                "    core-density region:",
                (
                    "      mask = f_O >= "
                    f"{core_density['relative_threshold']:.1e} * max(f_O)"
                ),
                f"      active points = {core_density['active_count']}",
                (
                    "      active fraction = "
                    f"{_format_report_value(core_density['active_fraction'])}"
                ),
                (
                    "      density weight fraction = "
                    f"{_format_report_value(core_density['density_weight_fraction'])}"
                ),
                f"      zero count = {core_density['zero_count']}",
                f"      near-zero count = {core_density['near_zero_count']}",
                (
                    "      near-zero density weight fraction = "
                    f"{_format_report_value(core_density['near_zero_density_weight_fraction'])}"
                ),
                (
                    "      near-zero transformed-density fraction = "
                    f"{_format_report_value(core_density['near_zero_transformed_density_fraction'])}"
                ),
                (
                    "      min |detJ| = "
                    f"{_format_report_value(core_density['abs_min'])}"
                ),
                (
                    "      median |detJ| = "
                    f"{_format_report_value(core_density['abs_median'])}"
                ),
                "      min location:",
                f"        delta = {_format_report_value(core_location['delta'])}",
                (
                    "        sigma/I = "
                    f"{_format_report_value(core_location['sigma/I'])}"
                ),
                f"        P = {_format_report_value(core_location['P'])}",
                f"        ADO = {_format_report_value(core_location['ADO'])}",
                (
                    "        f_O = "
                    f"{_format_report_value(core_location['observable_density'])}"
                ),
                f"      status = {core_density['status']}",
            ]
        )
    lines.append("-" * 72)
    return "\n".join(lines)


def format_jacobian_report_markdown(report: dict, heading: str) -> str:
    location = report["min_location"]
    sign_change = report.get("sign_change", {})
    rows = [
        ("determinant", report["name"]),
        ("total interior points", report["total_count"]),
        ("finite count", report["finite_count"]),
        ("non-finite count", report["nonfinite_count"]),
        ("zero count", report["zero_count"]),
        ("near-zero threshold", f"{report['near_zero_threshold']:.1e}"),
        ("near-zero count", report["near_zero_count"]),
        ("min |detJ|", _format_report_value(report["abs_min"])),
        ("p01 |detJ|", _format_report_value(report["abs_p01"])),
        ("median |detJ|", _format_report_value(report["abs_median"])),
        ("max |detJ|", _format_report_value(report["abs_max"])),
        ("signed min detJ", _format_report_value(report["signed_min"])),
        ("signed max detJ", _format_report_value(report["signed_max"])),
        ("positive count", report["positive_count"]),
        ("negative count", report["negative_count"]),
        ("min location: delta", _format_report_value(location["delta"])),
        ("min location: sigma/I", _format_report_value(location["sigma/I"])),
        ("min location: P", _format_report_value(location["P"])),
        ("min location: ADO", _format_report_value(location["ADO"])),
        ("signed detJ sign-change edges along delta", sign_change.get("along_delta", 0)),
        (
            "signed detJ sign-change edges along sigma/I",
            sign_change.get("along_sigma_i", 0),
        ),
        ("signed detJ total sign-change edges", sign_change.get("total", 0)),
        ("status", report["status"]),
    ]
    lines = [
        heading,
        "",
        "##### Full-grid diagnostic",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_escape_markdown_table_cell(metric)} | "
        f"{_escape_markdown_table_cell(value)} |"
        for metric, value in rows
    )

    high_density = report.get("high_density", {})
    threshold_entries = high_density.get("thresholds", [])
    if threshold_entries:
        lines.extend(
            [
                "",
                "##### Density-threshold diagnostics",
                "",
                (
                    "| rel threshold | active points | density weight fraction | "
                    "zero count | near-zero count | near-zero density weight fraction | "
                    "near-zero transformed-density fraction | status |"
                ),
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        lines.extend(
            (
                f"| {entry['relative_threshold']:.1e} | {entry['active_count']} | "
                f"{_format_report_value(entry['density_weight_fraction'])} | "
                f"{entry['zero_count']} | {entry['near_zero_count']} | "
                f"{_format_report_value(entry['near_zero_density_weight_fraction'])} | "
                f"{_format_report_value(entry['near_zero_transformed_density_fraction'])} | "
                f"{entry['status']} |"
            )
            for entry in threshold_entries
        )

        core_density = _find_density_threshold(report, CORE_DENSITY_REL_THRESHOLD)
        if core_density is not None:
            core_location = core_density["min_location"]
            lines.extend(
                [
                    "",
                    "**Core-density detail (1.0e-03)**",
                    "",
                    "| Metric | Value |",
                    "| --- | ---: |",
                    (
                        "| absolute threshold | "
                        f"{_format_report_value(core_density['absolute_threshold'])} |"
                    ),
                    f"| finite count | {core_density['finite_count']} |",
                    f"| non-finite count | {core_density['nonfinite_count']} |",
                    f"| min \\|detJ\\| | {_format_report_value(core_density['abs_min'])} |",
                    f"| p01 \\|detJ\\| | {_format_report_value(core_density['abs_p01'])} |",
                    (
                        "| median \\|detJ\\| | "
                        f"{_format_report_value(core_density['abs_median'])} |"
                    ),
                    f"| max \\|detJ\\| | {_format_report_value(core_density['abs_max'])} |",
                    f"| min location: delta | {_format_report_value(core_location['delta'])} |",
                    (
                        "| min location: sigma/I | "
                        f"{_format_report_value(core_location['sigma/I'])} |"
                    ),
                    f"| min location: P | {_format_report_value(core_location['P'])} |",
                    f"| min location: ADO | {_format_report_value(core_location['ADO'])} |",
                    (
                        "| min location: observable density | "
                        f"{_format_report_value(core_location['observable_density'])} |"
                    ),
                ]
            )
    return "\n".join(lines)


def transform_pdf_with_central_difference(
    pado_points,
    p_pdf,
    ado_pdf,
    *,
    return_report: bool = False,
    return_array: bool = False,
    return_export_data: bool = False,
    near_zero_threshold: float = DETJ_NEAR_ZERO_THRESHOLD,
    high_density_rel_thresholds=DEFAULT_HIGH_DENSITY_REL_THRESHOLDS,
    high_density_rel_threshold=None,
):
    # Interior region only, matching the original central-difference loops:
    # i = 1..n_delta-2 and j = 1..n_sigma-2 in Python indexing.
    center = pado_points[1:-1, 1:-1, :]

    # Central-difference denominators along the delta/theta direction.
    # These match:
    #   pado_points[i + 1, 0, col] - pado_points[i - 1, 0, col]
    delta_den = pado_points[2:, 0, 0] - pado_points[:-2, 0, 0]
    theta_den = pado_points[2:, 0, 1] - pado_points[:-2, 0, 1]

    # Central-difference denominator along the sigma/I direction.
    # This matches:
    #   pado_points[i, j + 1, 2] - pado_points[i, j - 1, 2]
    sigma_den = pado_points[1:-1, 2:, 2] - pado_points[1:-1, :-2, 2]

    p_center = center[:, :, 3]
    ado_center = center[:, :, 4]

    # Observable-space product density used by the original Mathematica code:
    # gaus2d = pPdf[p] * adoPdf[ado]
    obs_density = p_pdf(p_center) * ado_pdf(ado_center)
    gaus2d = obs_density

    # Central differences for p(delta, sigma/I) and ado(delta, sigma/I).
    dPdDelta = (pado_points[2:, 1:-1, 3] - pado_points[:-2, 1:-1, 3]) / delta_den[
        :, None
    ]
    dPdSigma = (pado_points[1:-1, 2:, 3] - pado_points[1:-1, :-2, 3]) / sigma_den
    dRdDelta = (pado_points[2:, 1:-1, 4] - pado_points[:-2, 1:-1, 4]) / delta_den[
        :, None
    ]
    dRdSigma = (pado_points[1:-1, 2:, 4] - pado_points[1:-1, :-2, 4]) / sigma_den

    # Same central differences, but with theta = arctan(delta) as the first variable.
    dPdTheta = (pado_points[2:, 1:-1, 3] - pado_points[:-2, 1:-1, 3]) / theta_den[
        :, None
    ]
    dRdTheta = (pado_points[2:, 1:-1, 4] - pado_points[:-2, 1:-1, 4]) / theta_den[
        :, None
    ]

    det_delta_sigma = dPdDelta * dRdSigma - dPdSigma * dRdDelta
    jac_delta_sigma = np.abs(det_delta_sigma)

    det_theta_sigma = dPdTheta * dRdSigma - dPdSigma * dRdTheta
    jac_theta_sigma = np.abs(det_theta_sigma)

    # Output columns keep the original CSV-ready structure:
    # 0: delta                          -> Mathematica delta
    # 1: sigma/I                        -> Mathematica sigma/I
    # 2: pdf density in (delta, sigma/I) -> gaus2d * |d(P,ADO)/d(delta,sigma/I)|
    # 3: ArcTan[delta] (theta_deg)      -> Mathematica AT
    # 4: pdf density in (AT, sigma/I)   -> gaus2d * |d(P,ADO)/d(AT,sigma/I)|
    # 5: p                              -> theoretical P at the grid point
    # 6: ado                            -> theoretical ADO at the grid point
    # 7: jacobian(delta,sigma/I)        -> |d(P,ADO)/d(delta,sigma/I)|
    # 8: gaus2d                         -> p_pdf(p) * ado_pdf(ado)
    # 9: jacobian(AT,sigma/I)           -> |d(P,ADO)/d(AT,sigma/I)|
    rows = np.stack(
        [
            center[:, :, 0],
            center[:, :, 2],
            gaus2d * jac_delta_sigma,
            center[:, :, 1],
            gaus2d * jac_theta_sigma,
            p_center,
            ado_center,
            jac_delta_sigma,
            gaus2d,
            jac_theta_sigma,
        ],
        axis=-1,
    )

    result_rows = rows if return_array else rows.tolist()
    if not return_report and not return_export_data:
        return result_rows

    jacobian_report = {
        "delta_sigma": build_jacobian_report(
            det_delta_sigma,
            jac_delta_sigma,
            center,
            name="d(P,ADO)/d(delta,sigma/I)",
            near_zero_threshold=near_zero_threshold,
            obs_density=obs_density,
            high_density_rel_thresholds=high_density_rel_thresholds,
            high_density_rel_threshold=high_density_rel_threshold,
        ),
        "theta_sigma": build_jacobian_report(
            det_theta_sigma,
            jac_theta_sigma,
            center,
            name="d(P,ADO)/d(AT,sigma/I)",
            near_zero_threshold=near_zero_threshold,
            obs_density=obs_density,
            high_density_rel_thresholds=high_density_rel_thresholds,
            high_density_rel_threshold=high_density_rel_threshold,
        ),
    }
    export_data = {
        "detj_signed_delta_sigma": det_delta_sigma,
        "detj_signed_theta_sigma": det_theta_sigma,
    }
    if return_report and return_export_data:
        return result_rows, jacobian_report, export_data
    if return_report:
        return result_rows, jacobian_report
    return result_rows, export_data
