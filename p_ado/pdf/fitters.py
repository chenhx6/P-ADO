"""
Model-selection and parameter-fitting logic for 1D observable PDFs.

Overview
--------
The project starts from experimental inputs of the form:

- central value
- left error
- right error

For each observable (P and ADO), this module converts that uncertainty
description into a concrete 1D PDF model. The result is a small dictionary
describing one of the following:

1. Gaussian
2. Skew-normal
3. Continuous split-normal

This module answers three main questions:

1. What algorithm do we use?
2. When do we use skew-normal?
3. When do we use continuous split-normal?

Algorithm used here
-------------------
The fitting strategy is quantile-based.

Given:
    center = x50
    left error = eL
    right error = eR

we interpret the measurement as approximately specifying:

    x16 = center - eL
    x50 = center
    x84 = center + eR

These correspond to the 16%, 50%, and 84% quantiles, which are the natural
analogs of -1 sigma, median, and +1 sigma for a Gaussian-like uncertainty.

The workflow is:

1. If the errors are symmetric enough:
   use a Gaussian directly.

2. If the errors are asymmetric:
   fit a skew-normal so that its quantiles match x16, x50, x84 as closely as
   possible.

3. If the skew-normal fit is poor but fallback is enabled:
   use a continuous split-normal with left/right widths taken directly from the
   reported errors.

What this module returns
------------------------
It does not return a callable PDF directly.
Instead, it returns a model dictionary such as:

    {"kind": "gaussian", ...}
    {"kind": "skew_normal", ...}
    {"kind": "split_normal", ...}

That dictionary is then turned into an evaluator by `pdf/evaluators.py`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import skewnorm

from ..config import GAUSSIAN_SIGMA_FLOOR, PdfConfig


def _make_skew_bounds(mu: float, sigma: float, pdf_cfg: PdfConfig):
    return [
        (
            mu - pdf_cfg.skew_center_shift_factor * sigma,
            mu + pdf_cfg.skew_center_shift_factor * sigma,
        ),
        (GAUSSIAN_SIGMA_FLOOR, pdf_cfg.skew_scale_factor_upper * sigma),
        (-pdf_cfg.skew_shape_bound, pdf_cfg.skew_shape_bound),
    ]


def _normalized_bound_distance(value: float, lower: float, upper: float) -> float:
    span = upper - lower
    if not np.isfinite(span) or span <= 0.0:
        return np.inf
    return min(value - lower, upper - value) / span


def _build_split_fallback(
    mu: float,
    eL: float,
    eR: float,
    fit_err: float,
    diagnostics: dict,
    rejection_reason: str,
):
    return {
        "kind": "split_normal",
        "accepted": False,
        "candidate_kind": "skew_normal",
        "rejection_reason": rejection_reason,
        "mu": mu,
        "sigma_l": eL,
        "sigma_r": eR,
        "fit_err": fit_err,
        "diagnostics": diagnostics,
    }


def _format_float_list(values) -> str:
    return "[" + ", ".join(f"{float(v):.6g}" for v in values) + "]"


def _collect_pdf_warnings(model: dict) -> list[str]:
    diagnostics = model.get("diagnostics", {})
    warnings: list[str] = list(diagnostics.get("soft_warnings", []))

    if not warnings:
        near_bound_flags = diagnostics.get("near_bound_flags", {})
        near_bound_names = [name for name, flag in near_bound_flags.items() if flag]
        if near_bound_names:
            warnings.append(f"near optimizer bound: {', '.join(near_bound_names)}")

        candidate_params = diagnostics.get("candidate_params", {})
        alpha = candidate_params.get("alpha")
        shape_limit = diagnostics.get("shape_soft_limit")
        if (
            alpha is not None
            and shape_limit is not None
            and np.isfinite(alpha)
            and np.isfinite(shape_limit)
            and abs(alpha) > 0.8 * shape_limit
        ):
            warnings.append(
                "large alpha relative to soft limit: "
                f"|alpha|={abs(alpha):.6g}, limit={shape_limit:.6g}"
            )

        quantile_warning_flags = diagnostics.get("quantile_warning_flags", {})
        norm_errors = diagnostics.get("norm_errors", {})
        warning_tolerances = diagnostics.get("warning_tolerances", {})
        for q_name, tol_key in (
            ("q16", "q16_tol_frac"),
            ("q50", "q50_tol_frac"),
            ("q84", "q84_tol_frac"),
        ):
            if not quantile_warning_flags.get(q_name, False):
                continue
            err = norm_errors.get(q_name)
            tol = warning_tolerances.get(tol_key)
            if err is None or tol is None:
                warnings.append(f"{q_name} quantile mismatch warning")
            else:
                warnings.append(
                    f"{q_name} quantile mismatch warning: err={err:.6g}, tol={tol:.6g}"
                )

    return warnings


def _format_report_list(title: str, items: list[str]) -> list[str]:
    lines = [f"  {title}:"]
    if not items:
        lines.append("    - none")
        return lines

    for item in items:
        lines.append(f"    - {item}")
    return lines


def _format_parameters_block(pairs: list[tuple[str, object]]) -> list[str]:
    lines = ["  parameters:"]
    for key, value in pairs:
        if value is None:
            lines.append(f"    {key}=None")
        else:
            lines.append(f"    {key}={float(value):.6g}")
    return lines


def _format_quantiles_block(targets: dict, fitted: dict) -> list[str]:
    return [
        "  quantiles:",
        "    target = "
        + _format_float_list(
            [
                targets.get("q16", np.nan),
                targets.get("q50", np.nan),
                targets.get("q84", np.nan),
            ]
        ),
        "    fitted = "
        + _format_float_list(
            [
                fitted.get("q16", np.nan),
                fitted.get("q50", np.nan),
                fitted.get("q84", np.nan),
            ]
        ),
    ]


def _format_fit_quality_block(model: dict) -> list[str]:
    return [
        "  fit quality:",
        f"    fit_err = {float(model.get('fit_err', np.nan)):.6g}",
        f"    accepted = {model.get('accepted', False)}",
    ]


def _model_status(model: dict, diagnostics: dict) -> str:
    if diagnostics.get("accepted_with_warnings", False):
        return "accepted with warnings"
    if model.get("accepted", False):
        return "accepted"
    hard_reason = diagnostics.get("hard_rejection_reason") or model.get(
        "rejection_reason"
    )
    if hard_reason:
        return "hard rejection -> split_normal fallback"
    return "rejected"


def format_pdf_model_report(name: str, model: dict) -> str:
    """
    Format a concise terminal report for one fitted observable PDF.

    The returned string is intended for one summary block per transition rather
    than per-grid-point logging.
    """

    kind = model.get("kind", "unknown")
    diagnostics = model.get("diagnostics", {})
    lines = [f"{name} PDF"]

    if kind == "gaussian":
        lines.extend(
            [
                "  kind: gaussian",
                (
                    "  formula: f(x) = 1/(sqrt(2*pi)*sigma) "
                    "* exp(-(x-mu)^2/(2*sigma^2))"
                ),
            ]
        )
        lines.extend(
            _format_parameters_block(
                [
                    ("mu", model.get("mu", np.nan)),
                    ("sigma", model.get("sigma", np.nan)),
                ]
            )
        )
        lines.extend(
            [
                f"  status: {_model_status(model, diagnostics)}",
                (
                    "  note: "
                    f"{diagnostics.get('symmetry_note', 'symmetric errors triggered the gaussian branch')}"
                ),
            ]
        )
        return "\n".join(lines)

    if kind == "skew_normal":
        targets = diagnostics.get("targets", {})
        fitted = diagnostics.get("fitted_quantiles", {})
        near_bound_flags = diagnostics.get("near_bound_flags", {})
        soft_warnings = _collect_pdf_warnings(model)
        hard_rejection_reason = diagnostics.get("hard_rejection_reason")
        lines.extend(
            [
                "  kind: skew_normal",
                (
                    "  formula: f(x) = 2/omega * phi((x-xi)/omega) "
                    "* Phi(alpha * (x-xi)/omega)"
                ),
            ]
        )
        lines.extend(
            _format_parameters_block(
                [
                    ("xi", model.get("xi", np.nan)),
                    ("omega", model.get("omega", np.nan)),
                    ("alpha", model.get("alpha", np.nan)),
                ]
            )
        )
        lines.extend(_format_quantiles_block(targets, fitted))
        lines.extend(_format_fit_quality_block(model))
        lines.extend(
            [
                "  boundary check:",
                f"    near_bound_flags = {near_bound_flags if near_bound_flags else {}}",
                f"  status: {_model_status(model, diagnostics)}",
            ]
        )
        lines.extend(_format_report_list("soft_warnings", soft_warnings))
        lines.extend(
            _format_report_list(
                "hard_rejection_reason",
                [hard_rejection_reason] if hard_rejection_reason else [],
            )
        )
        return "\n".join(lines)

    if kind == "split_normal":
        soft_warnings = _collect_pdf_warnings(model)
        hard_rejection_reason = diagnostics.get("hard_rejection_reason") or model.get(
            "rejection_reason"
        )
        targets = diagnostics.get("targets", {})
        fitted = diagnostics.get("fitted_quantiles", {})
        lines.extend(
            [
                "  kind: split_normal",
                (
                    "  formula: f(x) = sqrt(2/pi)/(sigma_L + sigma_R) "
                    "* exp(-(x-mu)^2/(2*sigma_L^2)) for x < mu"
                ),
                (
                    "           sqrt(2/pi)/(sigma_L + sigma_R) "
                    "* exp(-(x-mu)^2/(2*sigma_R^2)) for x >= mu"
                ),
            ]
        )
        lines.extend(
            _format_parameters_block(
                [
                    ("mu", model.get("mu", np.nan)),
                    ("sigma_L", model.get("sigma_l", np.nan)),
                    ("sigma_R", model.get("sigma_r", np.nan)),
                ]
            )
        )
        lines.extend(_format_fit_quality_block(model))
        lines.extend(
            [
                "  candidate model:",
                f"    candidate_kind = {model.get('candidate_kind', 'none')}",
            ]
        )
        lines.extend(_format_quantiles_block(targets, fitted))
        lines.append(f"  status: {_model_status(model, diagnostics)}")
        lines.extend(_format_report_list("soft_warnings", soft_warnings))
        lines.extend(
            _format_report_list(
                "hard_rejection_reason",
                [hard_rejection_reason] if hard_rejection_reason else [],
            )
        )
        return "\n".join(lines)

    lines.append(f"  kind: {kind}")
    lines.append(f"  accepted: {model.get('accepted', 'unknown')}")
    return "\n".join(lines)


def fit_pdf_model(center: float, err_l: float, err_r: float, pdf_cfg: PdfConfig):
    """
    Fit or choose a 1D PDF model from a central value and asymmetric errors.

    Parameters
    ----------
    center:
        Measured central value.
    err_l:
        Left uncertainty.
    err_r:
        Right uncertainty.
    pdf_cfg:
        PDF-related configuration thresholds and bounds.

    Returns
    -------
    dict
        A model description consumed later by `build_pdf_evaluator`.

    Detailed decision logic
    -----------------------
    Step 1: sanitize the inputs
    - Errors are converted to positive values.
    - Very small errors are clamped by `GAUSSIAN_SIGMA_FLOOR` to avoid
      degenerate zero-width distributions.

    Step 2: test whether the uncertainty is effectively symmetric
    - If left/right errors differ only within a small tolerance, we use a
      Gaussian with:

          mu = center
          sigma = (eL + eR) / 2

    Step 3: otherwise try skew-normal
    - We interpret the measurement as target quantiles:

          x16 = center - eL
          x50 = center
          x84 = center + eR

    - We then search for skew-normal parameters (xi, omega, alpha) whose
      quantiles at probabilities [0.16, 0.50, 0.84] best match those targets.
    - The optimization objective is the sum of squared quantile residuals.

    Step 4: decide whether the skew-normal fit is acceptable
    - Hard-reject only numerically pathological or clearly implausible fits.
    - Moderate q16/q50/q84 mismatch is treated as a warning by default rather
      than an automatic rejection.

    Step 5: otherwise use continuous split-normal fallback
    - This fallback directly preserves different left/right widths and is more
      robust for strongly asymmetric inputs.

    When skew-normal is appropriate
    -------------------------------
    Use skew-normal when:
    - the uncertainty is asymmetric
    - a single smooth skewed distribution is desired
    - the fitted quantiles reproduce the input well enough

    When continuous split-normal is appropriate
    -------------------------------------------
    Use split-normal fallback when:
    - the uncertainty is asymmetric
    - skew-normal optimization fails, or
    - skew-normal exists mathematically but cannot match the requested quantiles
      within the configured quality threshold

    In short:
    - symmetric input -> Gaussian
    - asymmetric and well-fit -> skew-normal
    - asymmetric but poorly fit -> continuous split-normal
    """

    mu = float(center)
    eL = max(abs(float(err_l)), GAUSSIAN_SIGMA_FLOOR)
    eR = max(abs(float(err_r)), GAUSSIAN_SIGMA_FLOOR)
    sigma = 0.5 * (eL + eR)

    # If the two sides are nearly identical, a Gaussian is the simplest and
    # most stable description of the uncertainty.
    if abs(eL - eR) <= pdf_cfg.skew_symmetry_tolerance * max(sigma, 1.0):
        return {
            "kind": "gaussian",
            "accepted": True,
            "mu": mu,
            "sigma": sigma,
            "diagnostics": {
                "sigma": sigma,
                "symmetry_note": "left/right errors are within the symmetry tolerance",
                "err_l": eL,
                "err_r": eR,
            },
        }

    # Convert the measurement plus asymmetric errors into target quantiles.
    # This is the key modeling assumption behind the quantile-matching fit.
    x16 = mu - eL
    x50 = mu
    x84 = mu + eR
    probs = np.array([0.16, 0.50, 0.84], dtype=float)
    targets = np.array([x16, x50, x84], dtype=float)

    # Initial guess for the skew-normal parameters:
    # - xi starts from the center
    # - omega starts from the average width
    # - alpha starts from 0, i.e. no skew
    x0 = np.array([mu, sigma, 0.0], dtype=float)  # xi, omega, alpha

    # Bounds keep the optimizer in a numerically reasonable region.
    # They are not derived from a strict physical model; they mainly prevent
    # unrealistic parameter excursions during quantile matching.
    bounds = _make_skew_bounds(mu, sigma, pdf_cfg)

    def objective(params):
        """
        Objective function for quantile matching.

        Given a trial skew-normal (xi, omega, alpha), compute its 16%, 50%, and
        84% quantiles and compare them with the target values implied by the
        measurement. The smaller the returned value, the better the fit.
        """

        xi, omega, alpha = params
        q = skewnorm.ppf(
            probs,
            a=alpha,
            loc=xi,
            scale=max(omega, GAUSSIAN_SIGMA_FLOOR),
        )
        if not np.all(np.isfinite(q)):
            return 1e9
        res = q - targets
        return float(np.sum(res * res))

    # L-BFGS-B is used because:
    # - the parameter space is low-dimensional
    # - box constraints are needed
    # - we only need a practical bounded optimizer, not symbolic fitting
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    fit_err = float(res.fun) if res.success else np.inf

    if res.success:
        xi, omega, alpha = map(float, res.x)
    else:
        xi, omega, alpha = np.nan, np.nan, np.nan

    q = skewnorm.ppf(
        probs,
        a=alpha,
        loc=xi,
        scale=max(abs(omega), GAUSSIAN_SIGMA_FLOOR) if np.isfinite(omega) else np.nan,
    )
    q16, q50, q84 = map(float, q)

    abs_errors = (
        np.abs(q - targets)
        if np.all(np.isfinite(q))
        else np.array([np.inf, np.inf, np.inf])
    )
    norm_scales = np.array(
        [
            max(eL, GAUSSIAN_SIGMA_FLOOR),
            max(sigma, GAUSSIAN_SIGMA_FLOOR),
            max(eR, GAUSSIAN_SIGMA_FLOOR),
        ],
        dtype=float,
    )
    norm_errors = abs_errors / norm_scales

    xi_bounds = bounds[0]
    omega_bounds = bounds[1]
    alpha_bounds = bounds[2]

    diagnostics = {
        "optimizer_success": bool(res.success),
        "optimizer_status": int(res.status),
        "optimizer_message": str(res.message),
        "targets": {"q16": x16, "q50": x50, "q84": x84},
        "candidate_params": {"xi": xi, "omega": omega, "alpha": alpha},
        "bounds": {
            "xi": tuple(map(float, xi_bounds)),
            "omega": tuple(map(float, omega_bounds)),
            "alpha": tuple(map(float, alpha_bounds)),
        },
        "fitted_quantiles": {"q16": q16, "q50": q50, "q84": q84},
        "abs_errors": {
            "q16": float(abs_errors[0]),
            "q50": float(abs_errors[1]),
            "q84": float(abs_errors[2]),
        },
        "norm_errors": {
            "q16": float(norm_errors[0]),
            "q50": float(norm_errors[1]),
            "q84": float(norm_errors[2]),
        },
        "fit_err": fit_err,
        "sigma_ref": sigma,
        "normalization_scales": {
            "q16": float(norm_scales[0]),
            "q50": float(norm_scales[1]),
            "q84": float(norm_scales[2]),
        },
        "shape_soft_limit": pdf_cfg.skew_accept_shape_soft_limit,
        "input_errors": {"err_l": eL, "err_r": eR},
        "near_bound_fraction": pdf_cfg.skew_accept_near_bound_frac,
        "warning_tolerances": {
            "q16_tol_frac": pdf_cfg.skew_accept_q16_tol_frac,
            "q50_tol_frac": pdf_cfg.skew_accept_q50_tol_frac,
            "q84_tol_frac": pdf_cfg.skew_accept_q84_tol_frac,
        },
        "reject_tolerances": {
            "q16_tol_frac": pdf_cfg.skew_reject_q16_tol_frac,
            "q50_tol_frac": pdf_cfg.skew_reject_q50_tol_frac,
            "q84_tol_frac": pdf_cfg.skew_reject_q84_tol_frac,
        },
    }

    quantile_warning_flags = {
        "q16": bool(norm_errors[0] > pdf_cfg.skew_accept_q16_tol_frac),
        "q50": bool(norm_errors[1] > pdf_cfg.skew_accept_q50_tol_frac),
        "q84": bool(norm_errors[2] > pdf_cfg.skew_accept_q84_tol_frac),
    }
    quantile_rejection_flags = {
        "q16": bool(norm_errors[0] > pdf_cfg.skew_reject_q16_tol_frac),
        "q50": bool(norm_errors[1] > pdf_cfg.skew_reject_q50_tol_frac),
        "q84": bool(norm_errors[2] > pdf_cfg.skew_reject_q84_tol_frac),
    }
    diagnostics["quantile_warning_flags"] = quantile_warning_flags
    diagnostics["quantile_rejection_flags"] = quantile_rejection_flags

    # A low global fit_err alone is not enough: a candidate can still be
    # numerically fragile by sitting on a bound, collapsing to an unrealistically
    # small scale, or showing very large one-sided quantile errors even if the
    # summed objective value looks acceptable.
    rejection_reason = None
    if not res.success:
        rejection_reason = "optimizer_failed"
    elif fit_err > pdf_cfg.skew_fit_quality_cutoff:
        rejection_reason = "fit_err_above_cutoff"
    elif not np.all(np.isfinite([xi, omega, alpha, q16, q50, q84])):
        rejection_reason = "non_finite_candidate"
    elif abs(alpha) > pdf_cfg.skew_accept_shape_soft_limit:
        rejection_reason = "shape_soft_limit_exceeded"
    elif omega < pdf_cfg.skew_accept_scale_min_frac * sigma:
        rejection_reason = "scale_too_small"
    else:
        xi_near = (
            _normalized_bound_distance(xi, *xi_bounds)
            <= pdf_cfg.skew_accept_near_bound_frac
        )
        omega_near = (
            _normalized_bound_distance(omega, *omega_bounds)
            <= pdf_cfg.skew_accept_near_bound_frac
        )
        alpha_near = (
            _normalized_bound_distance(alpha, *alpha_bounds)
            <= pdf_cfg.skew_accept_near_bound_frac
        )
        diagnostics["near_bound_flags"] = {
            "xi": bool(xi_near),
            "omega": bool(omega_near),
            "alpha": bool(alpha_near),
        }
        if xi_near or omega_near or alpha_near:
            rejection_reason = "parameter_near_optimizer_bound"
        elif quantile_rejection_flags["q16"]:
            rejection_reason = "q16_reject_tolerance_exceeded"
        elif quantile_rejection_flags["q50"]:
            rejection_reason = "q50_reject_tolerance_exceeded"
        elif quantile_rejection_flags["q84"]:
            rejection_reason = "q84_reject_tolerance_exceeded"

    soft_warnings: list[str] = []
    if (
        np.isfinite(alpha)
        and np.isfinite(pdf_cfg.skew_accept_shape_soft_limit)
        and abs(alpha) > 0.8 * pdf_cfg.skew_accept_shape_soft_limit
    ):
        soft_warnings.append(
            "large alpha relative to soft limit: "
            f"|alpha|={abs(alpha):.6g}, limit={pdf_cfg.skew_accept_shape_soft_limit:.6g}"
        )
    for q_name, tol_key in (
        ("q16", "q16_tol_frac"),
        ("q50", "q50_tol_frac"),
        ("q84", "q84_tol_frac"),
    ):
        if not quantile_warning_flags[q_name]:
            continue
        soft_warnings.append(
            f"{q_name} quantile mismatch warning: "
            f"err={diagnostics['norm_errors'][q_name]:.6g}, "
            f"tol={diagnostics['warning_tolerances'][tol_key]:.6g}"
        )
    diagnostics["soft_warnings"] = soft_warnings
    diagnostics["accepted_with_warnings"] = (
        bool(soft_warnings) and rejection_reason is None
    )
    diagnostics["hard_rejection_reason"] = rejection_reason

    if rejection_reason is None:
        return {
            "kind": "skew_normal",
            "accepted": True,
            "xi": xi,
            "omega": float(max(abs(omega), GAUSSIAN_SIGMA_FLOOR)),
            "alpha": alpha,
            "fit_err": fit_err,
            "diagnostics": diagnostics,
        }

    # If skew-normal is not reliable enough, fall back to a continuous
    # split-normal that directly respects asymmetric left/right widths.
    if pdf_cfg.use_split_normal_fallback:
        return _build_split_fallback(
            mu=mu,
            eL=eL,
            eR=eR,
            fit_err=fit_err,
            diagnostics=diagnostics,
            rejection_reason=rejection_reason,
        )

    raise RuntimeError("Skew-normal fit failed and split-normal fallback is disabled.")
