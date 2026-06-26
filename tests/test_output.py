import csv
import gzip

import numpy as np
import pytest

from p_ado.config import OutputConfig
from p_ado.io.io import (
    CSV_HEADER,
    SINGULAR_DIAGNOSTIC_COLUMNS,
    write_csv_exports,
)
from p_ado.main import main, parse_args


def _sample_export_grid():
    rows = np.zeros((2, 2, 10), dtype=float)
    rows[:, :, 0] = [[-1.0, -1.0], [1.0, 1.0]]
    rows[:, :, 1] = [[0.2, 0.3], [0.2, 0.3]]
    rows[:, :, 3] = [[-45.0, -45.0], [45.0, 45.0]]
    rows[:, :, 5] = [[0.1, 0.2], [0.3, 0.4]]
    rows[:, :, 6] = [[1.1, 1.2], [1.3, 1.4]]
    rows[:, :, 7] = [[np.nan, 0.0], [1e-12, 2e-3]]
    rows[:, :, 8] = [[1.0, 2.0], [3.0, 4.0]]
    rows[:, :, 9] = [[np.nan, 0.0], [5e-12, -4e-3]]
    signed_det = np.array([[np.nan, 0.0], [-1e-12, 2e-3]], dtype=float)
    signed_det_theta = np.array([[np.nan, 0.0], [5e-12, -4e-3]], dtype=float)
    return rows, signed_det, signed_det_theta


def _read_csv(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_default_compressed_exports_partition_rows_and_preserve_schemas(tmp_path):
    rows, signed_det, signed_det_theta = _sample_export_grid()
    manifest = write_csv_exports(
        output_dir=tmp_path,
        stem="result",
        ji=5.5,
        jf=4.5,
        results=rows,
        detj_signed_delta_sigma=signed_det,
        detj_signed_theta_sigma=signed_det_theta,
        output_cfg=OutputConfig(),
    )

    datasets = {entry["dataset"]: entry for entry in manifest["datasets"]}
    assert set(datasets) == {"all", "detJ_singular", "detJ_regular"}
    assert {entry["file"] for entry in datasets.values()} == {
        "result_5.5_4.5_all.csv.gz",
        "result_5.5_4.5_detJ_singular.csv.gz",
        "result_5.5_4.5_detJ_regular.csv.gz",
    }
    assert datasets["all"]["row_count"] == 4
    assert datasets["detJ_singular"]["row_count"] == 3
    assert datasets["detJ_regular"]["row_count"] == 1
    assert datasets["all"]["row_count"] == (
        datasets["detJ_singular"]["row_count"]
        + datasets["detJ_regular"]["row_count"]
    )

    for entry in datasets.values():
        path = tmp_path / entry["file"]
        assert path.suffix == ".gz"
        assert path.read_bytes()[:2] == b"\x1f\x8b"
        assert entry["compressed"] is True
        assert entry["compression_method"] == "gzip"
        assert entry["compression_level"] == 1

    all_rows = _read_csv(tmp_path / datasets["all"]["file"])
    regular_rows = _read_csv(tmp_path / datasets["detJ_regular"]["file"])
    singular_rows = _read_csv(tmp_path / datasets["detJ_singular"]["file"])
    assert all_rows[0] == CSV_HEADER
    assert regular_rows[0] == CSV_HEADER
    assert singular_rows[0] == CSV_HEADER + SINGULAR_DIAGNOSTIC_COLUMNS

    diagnostic_index = {
        name: singular_rows[0].index(name) for name in SINGULAR_DIAGNOSTIC_COLUMNS
    }
    classes = [row[diagnostic_index["detJ_class"]] for row in singular_rows[1:]]
    signs = [
        row[diagnostic_index["detJ_sign(delta,sigma/I)"]]
        for row in singular_rows[1:]
    ]
    theta_signs = [
        row[diagnostic_index["detJ_sign(ArcTan[delta](AT),sigma/I)"]]
        for row in singular_rows[1:]
    ]
    relative_density = [
        float(row[diagnostic_index["observable_density_rel_to_max"]])
        for row in singular_rows[1:]
    ]
    assert classes == ["nonfinite", "zero", "near_zero"]
    assert signs == ["nonfinite", "0", "-1"]
    assert theta_signs == ["nonfinite", "0", "+1"]
    assert np.allclose(relative_density, [0.25, 0.5, 0.75])
    near_zero_row = singular_rows[-1]
    assert float(near_zero_row[diagnostic_index["detJ_abs(delta,sigma/I)"]]) == 1e-12
    assert (
        float(
            near_zero_row[
                diagnostic_index["detJ_abs(ArcTan[delta](AT),sigma/I)"]
            ]
        )
        == 5e-12
    )
    assert (
        float(near_zero_row[diagnostic_index["detJ_near_zero_threshold"]])
        == 1e-12
    )
    assert "abs(detJ_signed" in near_zero_row[diagnostic_index["detJ_split_rule"]]


def test_plain_and_optional_split_exports(tmp_path):
    rows, signed_det, signed_det_theta = _sample_export_grid()
    no_regular = write_csv_exports(
        output_dir=tmp_path / "no_regular",
        stem="result",
        ji=5.5,
        jf=4.5,
        results=rows,
        detj_signed_delta_sigma=signed_det,
        detj_signed_theta_sigma=signed_det_theta,
        output_cfg=OutputConfig(
            compress_csv=False,
            export_detj_regular=False,
        ),
    )
    no_regular_names = {entry["dataset"] for entry in no_regular["datasets"]}
    assert no_regular_names == {"all", "detJ_singular"}
    assert all(entry["file"].endswith(".csv") for entry in no_regular["datasets"])
    assert all(not entry["compressed"] for entry in no_regular["datasets"])

    no_split = write_csv_exports(
        output_dir=tmp_path / "no_split",
        stem="result",
        ji=5.5,
        jf=4.5,
        results=rows,
        detj_signed_delta_sigma=signed_det,
        detj_signed_theta_sigma=signed_det_theta,
        output_cfg=OutputConfig(export_detj_split=False),
    )
    assert [entry["dataset"] for entry in no_split["datasets"]] == ["all"]
    assert not list((tmp_path / "no_split").glob("*detJ*.csv.gz"))


def test_cli_output_options_and_validation():
    args = parse_args(
        [
            "--input",
            "input.dat",
            "--no-compress-csv",
            "--csv-compress-level",
            "4",
            "--no-detj-split-export",
            "--no-detj-regular-export",
            "--detj-near-zero-threshold",
            "2e-10",
        ]
    )
    assert args.no_compress_csv is True
    assert args.csv_compress_level == 4
    assert args.no_detj_split_export is True
    assert args.no_detj_regular_export is True
    assert args.detj_near_zero_threshold == 2e-10

    with pytest.raises(SystemExit):
        parse_args(["--input", "input.dat", "--csv-compress-level", "10"])
    with pytest.raises(SystemExit):
        parse_args(
            ["--input", "input.dat", "--detj-near-zero-threshold", "nan"]
        )


def test_main_wires_default_and_plain_output_options(tmp_path):
    input_path = tmp_path / "example_input.dat"
    input_path.write_text(
        "10.5 9.5 -0.03 0.13 0.05 0.48 0.06 0.06\n",
        encoding="utf-8",
    )

    compressed_dir = tmp_path / "compressed"
    main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(compressed_dir),
            "--output-stem",
            "bench",
            "--mode",
            "test",
        ]
    )
    assert (compressed_dir / "bench_10.5_9.5_all.csv.gz").is_file()
    assert (compressed_dir / "bench_10.5_9.5_detJ_singular.csv.gz").is_file()
    assert (compressed_dir / "bench_10.5_9.5_detJ_regular.csv.gz").is_file()
    markdown = (compressed_dir / "example_input_reports.md").read_text(encoding="utf-8")
    assert "### CSV export" in markdown
    assert "The official benchmark uses the all-data file." in markdown

    plain_dir = tmp_path / "plain"
    main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(plain_dir),
            "--output-stem",
            "bench",
            "--mode",
            "test",
            "--no-compress-csv",
            "--no-detj-regular-export",
        ]
    )
    assert (plain_dir / "bench_10.5_9.5_all.csv").is_file()
    assert (plain_dir / "bench_10.5_9.5_detJ_singular.csv").is_file()
    assert not (plain_dir / "bench_10.5_9.5_detJ_regular.csv").exists()
