import datetime

import numpy as np

from p_ado.compute.jacobian import build_jacobian_report
from p_ado.io.io import output_markdown_name, write_markdown_report
from p_ado.runtime_report import format_markdown_runtime_report


def test_density_threshold_weight_fractions_follow_definitions():
    center = np.array(
        [
            [
                [0.1, 5.0, 0.2, 0.3, 0.4],
                [0.2, 6.0, 0.3, 0.4, 0.5],
                [0.3, 7.0, 0.4, 0.5, 0.6],
            ]
        ],
        dtype=float,
    )
    raw_det = np.array([[0.0, 1e-13, 2.0]], dtype=float)
    obs_density = np.array([[1.0, 2.0, 3.0]], dtype=float)
    report = build_jacobian_report(
        raw_det,
        np.abs(raw_det),
        center,
        name="test determinant",
        obs_density=obs_density,
    )
    threshold = report["high_density"]["thresholds"][0]
    transformed_density = obs_density * np.abs(raw_det)

    assert threshold["zero_density_weight_fraction"] == 1.0 / 6.0
    assert threshold["near_zero_density_weight_fraction"] == 3.0 / 6.0
    assert threshold["zero_transformed_density_fraction"] == 0.0
    assert np.isclose(
        threshold["near_zero_transformed_density_fraction"],
        transformed_density[0, :2].sum() / transformed_density.sum(),
    )


def test_signed_detj_sign_change_edges_are_reported():
    center = np.zeros((2, 2, 5), dtype=float)
    raw_det = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=float)
    report = build_jacobian_report(
        raw_det,
        np.abs(raw_det),
        center,
        name="test determinant",
    )

    assert report["sign_change"] == {
        "along_delta": 2,
        "along_sigma_i": 2,
        "total": 4,
    }


def test_markdown_report_writes_required_sections(tmp_path):
    center = np.array([[[0.25, 14.0, 0.3, -0.1, 0.4]]], dtype=float)
    raw_det = np.array([[2.0]], dtype=float)
    jacobian_report = {
        "delta_sigma": build_jacobian_report(
            raw_det,
            np.abs(raw_det),
            center,
            name="d(P,ADO)/d(delta,sigma/I)",
            obs_density=np.ones_like(raw_det),
        ),
        "theta_sigma": build_jacobian_report(
            raw_det,
            np.abs(raw_det),
            center,
            name="d(P,ADO)/d(AT,sigma/I)",
            obs_density=np.ones_like(raw_det),
        ),
    }
    gaussian_model = {
        "kind": "gaussian",
        "accepted": True,
        "mu": 0.0,
        "sigma": 0.1,
        "diagnostics": {"symmetry_note": "test model"},
    }
    transition_report = {
        "label": "5.5_4.5",
        "input": {
            "Ji": 5.5,
            "Jf": 4.5,
            "P": -0.12,
            "P_errL": 0.12,
            "P_errR": 0.11,
            "ADO": 0.39,
            "ADO_errL": 0.05,
            "ADO_errR": 0.05,
        },
        "grid": {
            "mode": "test",
            "theta_range": (-45.0, 45.0, 0.2),
            "sigma_i_range": (0.1, 0.6, 0.02),
            "padded_theta_range": (-45.2, 45.2),
            "padded_sigma_i_range": (0.08, 0.62),
            "pado_points_shape": (453, 28, 5),
            "interior_grid_shape": (451, 26),
        },
        "jacobian": jacobian_report,
        "p_model": gaussian_model,
        "ado_model": gaussian_model,
        "runtime": {
            "fit_models": 0.1,
            "build_pado_points": 0.2,
            "transform_pdf": 0.3,
            "total": 0.6,
        },
        "csv_export": {
            "detj_split_criterion": "jacobian(delta,sigma/I)",
            "detj_near_zero_threshold": 1e-12,
            "nonfinite_policy": (
                "non-finite detJ rows are included in detJ_singular"
            ),
            "density_threshold_policy": (
                "density-relative thresholds are diagnostic-only and do not affect split exports"
            ),
            "official_benchmark_dataset": "all",
            "one_pass_write_time_seconds": 0.25,
            "datasets": [
                {
                    "dataset": "all",
                    "path": str(tmp_path / "result_5.5_4.5_all.csv.gz"),
                    "file": "result_5.5_4.5_all.csv.gz",
                    "rule": "all transformed interior grid points",
                    "row_count": 100,
                    "fraction": 1.0,
                    "compressed": True,
                    "compression_method": "gzip",
                    "compression_level": 1,
                    "file_size_bytes": 1234,
                    "write_time_seconds": 0.25,
                },
                {
                    "dataset": "detJ_singular",
                    "path": str(tmp_path / "result_5.5_4.5_detJ_singular.csv.gz"),
                    "file": "result_5.5_4.5_detJ_singular.csv.gz",
                    "rule": "nonfinite or abs(detJ) <= threshold",
                    "row_count": 2,
                    "fraction": 0.02,
                    "compressed": True,
                    "compression_method": "gzip",
                    "compression_level": 1,
                    "file_size_bytes": 234,
                    "write_time_seconds": 0.25,
                },
                {
                    "dataset": "detJ_regular",
                    "path": str(tmp_path / "result_5.5_4.5_detJ_regular.csv.gz"),
                    "file": "result_5.5_4.5_detJ_regular.csv.gz",
                    "rule": "finite and abs(detJ) > threshold",
                    "row_count": 98,
                    "fraction": 0.98,
                    "compressed": True,
                    "compression_method": "gzip",
                    "compression_level": 1,
                    "file_size_bytes": 1000,
                    "write_time_seconds": 0.25,
                },
            ],
        },
    }
    timestamp = datetime.datetime(2026, 6, 14, 12, 0, 0)
    input_path = tmp_path / "example_input.dat"
    report_path = tmp_path / output_markdown_name(input_path)
    content = format_markdown_runtime_report(
        input_path=input_path,
        output_dir=tmp_path,
        mode="test",
        transition_reports=[transition_report],
        started_at=timestamp,
        completed_at=timestamp,
    )

    write_markdown_report(report_path, content)

    assert report_path.name == "example_input_reports.md"
    assert report_path.is_file()
    written = report_path.read_text(encoding="utf-8")
    assert written.startswith("# example_input_export\n")
    assert "## Transition 5.5_4.5" in written
    assert "### Jacobian check" in written
    assert "#### Jacobian(delta,sigma/I)" in written
    assert "#### Jacobian(AT,sigma/I)" in written
    assert written.count("##### Full-grid diagnostic") == 2
    assert written.count("##### Density-threshold diagnostics") == 2
    assert "density weight fraction" in written
    assert "near-zero transformed-density fraction" in written
    for threshold in ("1.0e-06", "1.0e-05", "1.0e-04", "1.0e-03", "1.0e-02"):
        assert written.count(threshold) >= 2
    assert "### CSV export" in written
    assert "result_5.5_4.5_all.csv.gz" in written
    assert "result_5.5_4.5_detJ_singular.csv.gz" in written
    assert "result_5.5_4.5_detJ_regular.csv.gz" in written
    assert str(tmp_path) not in written
    assert "| 100 |" in written
    assert "gzip level 1" in written
    assert "detJ near-zero threshold: `1.0e-12`" in written
    assert "Non-finite detJ rows are included in the singular export." in written
    assert "diagnostic-only and do not affect split exports" in written
    assert "signed detJ total sign-change edges" in written
