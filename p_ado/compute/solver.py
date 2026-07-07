from time import perf_counter
from typing import Any, Dict, Literal, Tuple, overload

from ..config import (
    DENSITY_REL_THRESHOLDS,
    DETJ_NEAR_ZERO_THRESHOLD,
    PdfConfig,
)
from ..pdf.evaluators import build_pdf_evaluator
from ..pdf.fitters import fit_pdf_model
from ..physics.pado import build_pado_points
from ..runtime_report import (
    _format_transition_runtime_report,
    format_csv_export_text,
    format_transition_markdown_report,
)
from .grid import build_delta_grid, build_sigma_grid
from .jacobian import transform_pdf_with_central_difference


def _build_transition_report(
    tr,
    grid_cfg,
    theta_deg_list,
    sigma_i_list,
    pado_points,
    fit_elapsed: float,
    points_elapsed: float,
    transform_elapsed: float,
    total_elapsed: float,
    p_model: dict,
    ado_model: dict,
    jacobian_report: dict,
) -> dict:
    theta_min, theta_max, theta_step = grid_cfg.theta_range()
    sigma_i_min, sigma_i_max, sigma_i_step = grid_cfg.sigma_i_range()
    interior_shape = (pado_points.shape[0] - 2, pado_points.shape[1] - 2)
    return {
        "label": tr.label or f"{tr.ji:.1f}_{tr.jf:.1f}",
        "input": {
            "Ji": float(tr.ji),
            "Jf": float(tr.jf),
            "P": float(tr.p_value),
            "P_errL": float(tr.p_err_l),
            "P_errR": float(tr.p_err_r),
            "ADO": float(tr.ado_value),
            "ADO_errL": float(tr.ado_err_l),
            "ADO_errR": float(tr.ado_err_r),
        },
        "grid": {
            "mode": grid_cfg.mode,
            "theta_range": (theta_min, theta_max, theta_step),
            "sigma_i_range": (sigma_i_min, sigma_i_max, sigma_i_step),
            "padded_theta_range": (
                float(theta_deg_list[0]),
                float(theta_deg_list[-1]),
            ),
            "padded_sigma_i_range": (
                float(sigma_i_list[0]),
                float(sigma_i_list[-1]),
            ),
            "pado_points_shape": tuple(int(v) for v in pado_points.shape),
            "interior_grid_shape": tuple(int(v) for v in interior_shape),
        },
        "jacobian": jacobian_report,
        "p_model": p_model,
        "ado_model": ado_model,
        "runtime": {
            "fit_models": float(fit_elapsed),
            "build_pado_points": float(points_elapsed),
            "transform_pdf": float(transform_elapsed),
            "total": float(total_elapsed),
        },
    }


@overload
def solve_transition(
    tr: Any,
    grid_cfg: Any,
    pdf_cfg: PdfConfig,
    return_report: Literal[True],
    return_array: bool = False,
    return_export_data: Literal[False] = False,
    high_density_rel_thresholds: Any = DENSITY_REL_THRESHOLDS,
    high_density_rel_threshold: Any = None,
    detj_near_zero_threshold: float = DETJ_NEAR_ZERO_THRESHOLD,
) -> Tuple[Any, Dict[str, Any]]:
    ...


@overload
def solve_transition(
    tr: Any,
    grid_cfg: Any,
    pdf_cfg: PdfConfig,
    return_report: Literal[True],
    return_array: bool = False,
    return_export_data: Literal[True] = True,
    high_density_rel_thresholds: Any = DENSITY_REL_THRESHOLDS,
    high_density_rel_threshold: Any = None,
    detj_near_zero_threshold: float = DETJ_NEAR_ZERO_THRESHOLD,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    ...


@overload
def solve_transition(
    tr: Any,
    grid_cfg: Any,
    pdf_cfg: PdfConfig,
    return_report: Literal[False] = False,
    return_array: bool = False,
    return_export_data: Literal[False] = False,
    high_density_rel_thresholds: Any = DENSITY_REL_THRESHOLDS,
    high_density_rel_threshold: Any = None,
    detj_near_zero_threshold: float = DETJ_NEAR_ZERO_THRESHOLD,
) -> Any:
    ...


@overload
def solve_transition(
    tr: Any,
    grid_cfg: Any,
    pdf_cfg: PdfConfig,
    return_report: Literal[False] = False,
    return_array: bool = False,
    return_export_data: Literal[True] = True,
    high_density_rel_thresholds: Any = DENSITY_REL_THRESHOLDS,
    high_density_rel_threshold: Any = None,
    detj_near_zero_threshold: float = DETJ_NEAR_ZERO_THRESHOLD,
) -> Tuple[Any, Dict[str, Any]]:
    ...


@overload
def solve_transition(
    tr: Any,
    grid_cfg: Any,
    pdf_cfg: PdfConfig,
    return_report: bool = False,
    return_array: bool = False,
    return_export_data: bool = False,
    high_density_rel_thresholds: Any = DENSITY_REL_THRESHOLDS,
    high_density_rel_threshold: Any = None,
    detj_near_zero_threshold: float = DETJ_NEAR_ZERO_THRESHOLD,
) -> Any:
    ...


def solve_transition(
    tr,
    grid_cfg,
    pdf_cfg: PdfConfig,
    return_report: bool = False,
    return_array: bool = False,
    return_export_data: bool = False,
    high_density_rel_thresholds=DENSITY_REL_THRESHOLDS,
    high_density_rel_threshold=None,
    detj_near_zero_threshold: float = DETJ_NEAR_ZERO_THRESHOLD,
) -> Any:
    t_total0 = perf_counter()
    theta_deg_list, delta_list = build_delta_grid(grid_cfg)
    sigma_list, sigma_i_list = build_sigma_grid(tr.ji, grid_cfg)

    t_fit0 = perf_counter()
    p_model = fit_pdf_model(tr.p_value, tr.p_err_l, tr.p_err_r, pdf_cfg)
    ado_model = fit_pdf_model(tr.ado_value, tr.ado_err_l, tr.ado_err_r, pdf_cfg)
    fit_elapsed = perf_counter() - t_fit0

    p_pdf = build_pdf_evaluator(p_model)
    ado_pdf = build_pdf_evaluator(ado_model)

    t_points0 = perf_counter()
    pado_points = build_pado_points(
        ji=tr.ji,
        jf=tr.jf,
        delta_list=delta_list,
        theta_deg_list=theta_deg_list,
        sigma_list=sigma_list,
        sigma_i_list=sigma_i_list,
    )
    points_elapsed = perf_counter() - t_points0

    t_transform0 = perf_counter()
    transform_output = transform_pdf_with_central_difference(
        pado_points=pado_points,
        p_pdf=p_pdf,
        ado_pdf=ado_pdf,
        return_report=True,
        return_array=return_array,
        return_export_data=return_export_data,
        near_zero_threshold=detj_near_zero_threshold,
        high_density_rel_thresholds=high_density_rel_thresholds,
        high_density_rel_threshold=high_density_rel_threshold,
    )
    if return_export_data:
        results, jacobian_report, export_data = transform_output
    else:
        results, jacobian_report = transform_output
    transform_elapsed = perf_counter() - t_transform0
    total_elapsed = perf_counter() - t_total0

    transition_report = _build_transition_report(
        tr=tr,
        grid_cfg=grid_cfg,
        theta_deg_list=theta_deg_list,
        sigma_i_list=sigma_i_list,
        pado_points=pado_points,
        fit_elapsed=fit_elapsed,
        points_elapsed=points_elapsed,
        transform_elapsed=transform_elapsed,
        total_elapsed=total_elapsed,
        p_model=p_model,
        ado_model=ado_model,
        jacobian_report=jacobian_report,
    )

    if pdf_cfg.verbose_pdf_fit:
        print(_format_transition_runtime_report(transition_report))

    if return_report and return_export_data:
        return results, transition_report, export_data
    if return_report:
        return results, transition_report
    if return_export_data:
        return results, export_data
    return results
