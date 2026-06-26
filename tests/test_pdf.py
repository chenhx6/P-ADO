import numpy as np
from scipy.stats import skewnorm

from p_ado.config import PdfConfig
from p_ado.pdf.evaluators import build_pdf_evaluator
from p_ado.pdf.fitters import fit_pdf_model
from p_ado.pdf.models import gaussian_pdf, skew_normal_pdf, split_normal_continuous_pdf


def test_symmetric_returns_gaussian():
    model = fit_pdf_model(0.48, 0.06, 0.06, PdfConfig())
    assert model["kind"] == "gaussian"
    assert model["accepted"] is True
    assert model["diagnostics"]["sigma"] > 0.0


def test_pdf_model_helpers_accept_scalar_and_array_inputs():
    gaussian_scalar = gaussian_pdf(0.1, 0.0, 0.5)
    gaussian_array = gaussian_pdf(np.array([-0.1, 0.0, 0.1]), 0.0, 0.5)

    skew_scalar = skew_normal_pdf(0.2, 0.1, 0.3, 4.0)
    skew_array = skew_normal_pdf(np.array([-0.2, 0.0, 0.2]), 0.1, 0.3, 4.0)

    split_scalar = split_normal_continuous_pdf(0.2, 0.0, 0.1, 0.3)
    split_array = split_normal_continuous_pdf(
        np.array([-0.2, 0.0, 0.2]),
        0.0,
        0.1,
        0.3,
    )

    assert isinstance(gaussian_scalar, float)
    assert isinstance(skew_scalar, float)
    assert isinstance(split_scalar, float)
    assert gaussian_array.shape == (3,)
    assert skew_array.shape == (3,)
    assert split_array.shape == (3,)


def test_skew_normal_direct_formula_matches_scipy_reference():
    x = np.array([-0.4, -0.1, 0.2, 0.6], dtype=float)
    xi = 0.15
    omega = 0.35
    alpha = 3.5

    expected = skewnorm.pdf(x, a=alpha, loc=xi, scale=omega)
    actual = skew_normal_pdf(x, xi, omega, alpha)

    assert np.allclose(actual, expected)


def test_build_pdf_evaluator_supports_bulk_array_evaluation():
    gaussian_eval = build_pdf_evaluator(
        {"kind": "gaussian", "mu": 0.0, "sigma": 0.5}
    )
    skew_eval = build_pdf_evaluator(
        {"kind": "skew_normal", "xi": 0.1, "omega": 0.3, "alpha": 2.0}
    )
    split_eval = build_pdf_evaluator(
        {"kind": "split_normal", "mu": 0.0, "sigma_l": 0.1, "sigma_r": 0.2}
    )

    x = np.array([-0.2, 0.0, 0.2], dtype=float)

    assert gaussian_eval(x).shape == (3,)
    assert skew_eval(x).shape == (3,)
    assert split_eval(x).shape == (3,)


def test_skew_fit_rejection_returns_split_normal_with_diagnostics():
    cfg = PdfConfig(
        skew_accept_shape_soft_limit=0.1,
        use_split_normal_fallback=True,
    )
    model = fit_pdf_model(0.48, 0.02, 0.12, cfg)

    assert model["kind"] == "split_normal"
    assert model["accepted"] is False
    assert model["candidate_kind"] == "skew_normal"
    assert "rejection_reason" in model
    assert "diagnostics" in model
    assert "candidate_params" in model["diagnostics"]
    assert "fitted_quantiles" in model["diagnostics"]
    assert model["diagnostics"]["hard_rejection_reason"] == model["rejection_reason"]


def test_skew_fit_can_be_accepted_with_quantile_warnings():
    cfg = PdfConfig(
        skew_accept_q16_tol_frac=1e-8,
        skew_accept_q50_tol_frac=1e-8,
        skew_accept_q84_tol_frac=1e-8,
        skew_reject_q16_tol_frac=1.0,
        skew_reject_q50_tol_frac=1.0,
        skew_reject_q84_tol_frac=1.0,
        use_split_normal_fallback=True,
    )
    model = fit_pdf_model(0.48, 0.02, 0.12, cfg)

    assert model["kind"] == "skew_normal"
    assert model["accepted"] is True
    assert model["diagnostics"]["accepted_with_warnings"] is True
    assert any(model["diagnostics"]["quantile_warning_flags"].values())
