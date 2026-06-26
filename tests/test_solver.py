import numpy as np

from p_ado.compute.jacobian import (
    format_jacobian_report_text,
    transform_pdf_with_central_difference,
)
from p_ado.compute.solver import solve_transition
from p_ado.config import GridConfig, PdfConfig
from p_ado.io.io import CSV_HEADER, TransitionInput


def _sample_pado_points():
    return np.array(
        [
            [
                [-2.0, -63.0, 0.10, 0.10, 1.00],
                [-2.0, -63.0, 0.20, 0.20, 1.10],
                [-2.0, -63.0, 0.30, 0.30, 1.20],
                [-2.0, -63.0, 0.40, 0.40, 1.30],
            ],
            [
                [-1.0, -45.0, 0.10, 0.40, 1.40],
                [-1.0, -45.0, 0.20, 0.50, 1.50],
                [-1.0, -45.0, 0.30, 0.60, 1.60],
                [-1.0, -45.0, 0.40, 0.70, 1.70],
            ],
            [
                [1.0, 45.0, 0.10, 0.80, 1.80],
                [1.0, 45.0, 0.20, 0.90, 1.90],
                [1.0, 45.0, 0.30, 1.00, 2.00],
                [1.0, 45.0, 0.40, 1.10, 2.10],
            ],
            [
                [2.0, 63.0, 0.10, 1.20, 2.20],
                [2.0, 63.0, 0.20, 1.30, 2.30],
                [2.0, 63.0, 0.30, 1.40, 2.40],
                [2.0, 63.0, 0.40, 1.50, 2.50],
            ],
        ],
        dtype=float,
    )


def test_solver_runs_in_test_mode():
    tr = TransitionInput(
        ji=10.5,
        jf=9.5,
        p_value=-0.03,
        p_err_l=0.13,
        p_err_r=0.05,
        ado_value=0.48,
        ado_err_l=0.06,
        ado_err_r=0.06,
    )
    results = solve_transition(tr, GridConfig(mode="test"), PdfConfig())
    assert len(results) > 0
    assert len(results[0]) > 0
    assert len(results[0][0]) == len(CSV_HEADER)


def test_jacobian_vectorized_matches_loop_reference():
    pado_points = _sample_pado_points()

    def p_pdf(x):
        return 2.0 * np.asarray(x) + 1.0

    def ado_pdf(x):
        return np.asarray(x) ** 2 + 0.5

    expected = []
    n_delta, n_sigma, _ = pado_points.shape
    for i in range(1, n_delta - 1):
        delta_den = pado_points[i + 1, 0, 0] - pado_points[i - 1, 0, 0]
        theta_den = pado_points[i + 1, 0, 1] - pado_points[i - 1, 0, 1]
        block = []
        for j in range(1, n_sigma - 1):
            p0 = pado_points[i, j, 3]
            ado0 = pado_points[i, j, 4]
            gaus2d = p_pdf(p0) * ado_pdf(ado0)

            sigma_den = pado_points[i, j + 1, 2] - pado_points[i, j - 1, 2]

            dPdDelta = (pado_points[i + 1, j, 3] - pado_points[i - 1, j, 3]) / delta_den
            dPdSigma = (pado_points[i, j + 1, 3] - pado_points[i, j - 1, 3]) / sigma_den
            dRdDelta = (pado_points[i + 1, j, 4] - pado_points[i - 1, j, 4]) / delta_den
            dRdSigma = (pado_points[i, j + 1, 4] - pado_points[i, j - 1, 4]) / sigma_den

            dPdTheta = (pado_points[i + 1, j, 3] - pado_points[i - 1, j, 3]) / theta_den
            dRdTheta = (pado_points[i + 1, j, 4] - pado_points[i - 1, j, 4]) / theta_den

            jac_delta_sigma = abs(dPdDelta * dRdSigma - dPdSigma * dRdDelta)
            jac_theta_sigma = abs(dPdTheta * dRdSigma - dPdSigma * dRdTheta)

            block.append(
                [
                    float(pado_points[i, j, 0]),
                    float(pado_points[i, j, 2]),
                    float(gaus2d * jac_delta_sigma),
                    float(pado_points[i, j, 1]),
                    float(gaus2d * jac_theta_sigma),
                    float(p0),
                    float(ado0),
                    float(jac_delta_sigma),
                    float(gaus2d),
                    float(jac_theta_sigma),
                ]
            )
        expected.append(block)

    actual = transform_pdf_with_central_difference(pado_points, p_pdf, ado_pdf)
    assert np.allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
    )


def test_jacobian_report_is_opt_in_and_preserves_legacy_rows():
    pado_points = _sample_pado_points()

    def p_pdf(x):
        return 2.0 * np.asarray(x) + 1.0

    def ado_pdf(x):
        return np.asarray(x) ** 2 + 0.5

    default_rows = transform_pdf_with_central_difference(pado_points, p_pdf, ado_pdf)
    explicit_legacy_rows = transform_pdf_with_central_difference(
        pado_points,
        p_pdf,
        ado_pdf,
        return_report=False,
    )
    report_rows, report = transform_pdf_with_central_difference(
        pado_points,
        p_pdf,
        ado_pdf,
        return_report=True,
    )

    assert isinstance(default_rows, list)
    assert explicit_legacy_rows == default_rows
    assert report_rows == default_rows
    assert set(report) == {"delta_sigma", "theta_sigma"}
    assert report["delta_sigma"]["name"] == "d(P,ADO)/d(delta,sigma/I)"
    assert report["theta_sigma"]["name"] == "d(P,ADO)/d(AT,sigma/I)"
    assert "high_density" in report["delta_sigma"]
    assert "high_density" in report["theta_sigma"]
    expected_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for key in ("delta_sigma", "theta_sigma"):
        thresholds = report[key]["high_density"]["thresholds"]
        assert [entry["relative_threshold"] for entry in thresholds] == (
            expected_thresholds
        )
        assert all(
            "near_zero_transformed_density_fraction" in entry
            for entry in thresholds
        )

    terminal_report = format_jacobian_report_text(report["delta_sigma"])
    assert "core-density region:" in terminal_report
    assert "mask = f_O >= 1.0e-03 * max(f_O)" in terminal_report
    for excluded_threshold in ("1.0e-06", "1.0e-05", "1.0e-04", "1.0e-02"):
        assert f"mask = f_O >= {excluded_threshold}" not in terminal_report

    array_rows, array_report, export_data = transform_pdf_with_central_difference(
        pado_points,
        p_pdf,
        ado_pdf,
        return_report=True,
        return_array=True,
        return_export_data=True,
    )
    assert isinstance(array_rows, np.ndarray)
    assert np.allclose(array_rows, np.asarray(default_rows))
    assert array_report["delta_sigma"]["name"] == report["delta_sigma"]["name"]
    assert export_data["detj_signed_delta_sigma"].shape == array_rows.shape[:2]


def test_zero_jacobian_is_reported_without_removing_rows():
    delta = np.array([-2.0, -1.0, 1.0, 2.0])
    theta = np.array([-63.0, -45.0, 45.0, 63.0])
    sigma_i = np.array([0.1, 0.2, 0.3, 0.4])
    delta_grid, sigma_grid = np.meshgrid(delta, sigma_i, indexing="ij")
    theta_grid = np.broadcast_to(theta[:, None], delta_grid.shape)
    p_grid = delta_grid + sigma_grid
    ado_grid = 2.0 * p_grid
    pado_points = np.stack(
        [delta_grid, theta_grid, sigma_grid, p_grid, ado_grid],
        axis=-1,
    )

    rows, report = transform_pdf_with_central_difference(
        pado_points,
        lambda x: np.ones_like(x),
        lambda x: np.ones_like(x),
        return_report=True,
    )

    assert len(rows) == 2
    assert len(rows[0]) == 2
    assert report["delta_sigma"]["zero_count"] > 0
    assert report["delta_sigma"]["status"] == "warning"


def test_zero_jacobian_outside_high_density_region_is_not_counted():
    delta = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    theta = np.array([-63.0, -45.0, 0.0, 45.0, 63.0])
    sigma_i = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    delta_grid, sigma_grid = np.meshgrid(delta, sigma_i, indexing="ij")
    theta_grid = np.broadcast_to(theta[:, None], delta_grid.shape)
    p_grid = delta_grid
    ado_grid = delta_grid * sigma_grid
    pado_points = np.stack(
        [delta_grid, theta_grid, sigma_grid, p_grid, ado_grid],
        axis=-1,
    )

    def p_pdf(values):
        values = np.asarray(values)
        return np.where(np.abs(values) < 0.5, 1e-12, 1.0)

    rows, report = transform_pdf_with_central_difference(
        pado_points,
        p_pdf,
        lambda x: np.ones_like(x),
        return_report=True,
    )

    obs_density = np.asarray(rows, dtype=float)[:, :, 8]
    expected_mask = obs_density >= 1e-6 * np.nanmax(obs_density)
    expected_weight_fraction = np.nansum(obs_density[expected_mask]) / np.nansum(
        obs_density
    )

    for key in ("delta_sigma", "theta_sigma"):
        full_report = report[key]
        threshold_entries = full_report["high_density"]["thresholds"]
        assert [entry["relative_threshold"] for entry in threshold_entries] == [
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
        ]
        high_density = threshold_entries[0]
        assert full_report["zero_count"] > 0
        assert high_density["relative_threshold"] == 1e-6
        assert high_density["absolute_threshold"] == 1e-6 * np.nanmax(obs_density)
        assert high_density["active_count"] == np.count_nonzero(expected_mask)
        assert np.isclose(
            high_density["density_weight_fraction"],
            expected_weight_fraction,
        )
        assert high_density["zero_count"] == 0
        assert high_density["near_zero_count"] == 0
        assert high_density["zero_density_weight_fraction"] == 0.0
        assert high_density["near_zero_density_weight_fraction"] == 0.0
        assert high_density["zero_transformed_density_fraction"] == 0.0
        assert high_density["near_zero_transformed_density_fraction"] == 0.0
        assert high_density["status"] == "accepted"
