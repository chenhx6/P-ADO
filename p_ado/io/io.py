from __future__ import annotations

import argparse
import csv
import gzip
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..config import CSV_COMPRESS_LEVEL, OutputConfig
from ..path_display import format_display_path


CSV_HEADER = [
    "delta",
    "sigma/I",
    "pdf density in delta and sigma/I dimensions",
    "ArcTan[delta](AT)",
    "pdf density in AT and sigma/I dimensions",
    "p",
    "ado",
    "jacobian(delta,sigma/I)",
    "gaus2d",
    "jacobian(ArcTan[delta](AT),sigma/I)",
]

SINGULAR_DIAGNOSTIC_COLUMNS = [
    "detJ_signed(delta,sigma/I)",
    "detJ_abs(delta,sigma/I)",
    "detJ_sign(delta,sigma/I)",
    "detJ_signed(ArcTan[delta](AT),sigma/I)",
    "detJ_abs(ArcTan[delta](AT),sigma/I)",
    "detJ_sign(ArcTan[delta](AT),sigma/I)",
    "detJ_class",
    "detJ_near_zero_threshold",
    "detJ_split_rule",
    "observable_density_rel_to_max",
]

DETJ_SPLIT_CRITERION = "jacobian(delta,sigma/I)"
DETJ_SPLIT_RULE = (
    "nonfinite or abs(detJ_signed(delta,sigma/I)) <= detJ_near_zero_threshold"
)


def _compression_level(value: str) -> int:
    try:
        level = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("compression level must be an integer") from exc
    if not 0 <= level <= 9:
        raise argparse.ArgumentTypeError("compression level must be between 0 and 9")
    return level


def _nonnegative_finite_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a floating-point number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return number


@dataclass(frozen=True)
class TransitionInput:
    ji: float
    jf: float
    p_value: float
    p_err_l: float
    p_err_r: float
    ado_value: float
    ado_err_l: float
    ado_err_r: float
    label: str = ""


def read_transitions_dat(path: str | Path) -> list[TransitionInput]:
    transitions: list[TransitionInput] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            parts = raw.split()
            if len(parts) < 8:
                raise ValueError(
                    f"Line {lineno} in {path} has fewer than 8 columns: {raw}"
                )

            transitions.append(
                TransitionInput(
                    ji=float(parts[0]),
                    jf=float(parts[1]),
                    p_value=float(parts[2]),
                    p_err_l=abs(float(parts[3])),
                    p_err_r=abs(float(parts[4])),
                    ado_value=float(parts[5]),
                    ado_err_l=abs(float(parts[6])),
                    ado_err_r=abs(float(parts[7])),
                    label=f"{float(parts[0]):.1f}_{float(parts[1]):.1f}",
                )
            )

    return transitions


def output_csv_name(stem: str, ji: float, jf: float) -> str:
    return f"{stem}_{ji:.1f}_{jf:.1f}.csv"


def output_dataset_name(
    stem: str,
    ji: float,
    jf: float,
    dataset: str,
    *,
    compressed: bool,
) -> str:
    extension = ".csv.gz" if compressed else ".csv"
    return f"{stem}_{ji:.1f}_{jf:.1f}_{dataset}{extension}"


def output_markdown_name(input_path: str | Path) -> str:
    return f"{Path(input_path).stem}_reports.md"


def write_markdown_report(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _open_csv_text(
    path: Path,
    *,
    compressed: bool,
    compresslevel: int = CSV_COMPRESS_LEVEL,
):
    if compressed:
        return gzip.open(
            path,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=compresslevel,
        )
    return path.open("w", encoding="utf-8", newline="")


def write_csv(
    path: str | Path,
    results,
    include_header: bool = True,
    compresslevel: int = CSV_COMPRESS_LEVEL,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = path.name.endswith(".gz")

    with _open_csv_text(
        path,
        compressed=compressed,
        compresslevel=compresslevel,
    ) as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow(CSV_HEADER)

        for block in results:
            for row in block:
                writer.writerow(row)


def _observable_density_max(rows: np.ndarray) -> float:
    density = rows[..., 8]
    finite_mask = np.isfinite(density)
    if not np.any(finite_mask):
        return np.nan
    return float(np.max(density, where=finite_mask, initial=-np.inf))


def _detj_sign_value(detj_signed: float) -> str:
    if not math.isfinite(detj_signed):
        return "nonfinite"
    if detj_signed == 0.0:
        return "0"
    return "-1" if detj_signed < 0.0 else "+1"


def _singular_diagnostic_values(
    row,
    detj_signed_delta_sigma: float,
    detj_signed_theta_sigma: float,
    detj_threshold: float,
    observable_density_max: float,
):
    if not math.isfinite(detj_signed_delta_sigma):
        detj_class = "nonfinite"
    elif detj_signed_delta_sigma == 0.0:
        detj_class = "zero"
    else:
        detj_class = "near_zero"

    detj_delta_sign = _detj_sign_value(detj_signed_delta_sigma)
    detj_theta_sign = _detj_sign_value(detj_signed_theta_sigma)

    observable_density = float(row[8])
    if (
        math.isfinite(observable_density)
        and math.isfinite(observable_density_max)
        and observable_density_max != 0.0
    ):
        density_relative_to_max = observable_density / observable_density_max
    else:
        density_relative_to_max = np.nan

    return [
        detj_signed_delta_sigma,
        abs(detj_signed_delta_sigma),
        detj_delta_sign,
        detj_signed_theta_sigma,
        abs(detj_signed_theta_sigma),
        detj_theta_sign,
        detj_class,
        detj_threshold,
        DETJ_SPLIT_RULE,
        density_relative_to_max,
    ]


def write_csv_exports(
    output_dir: str | Path,
    stem: str,
    ji: float,
    jf: float,
    results,
    detj_signed_delta_sigma,
    detj_signed_theta_sigma,
    output_cfg: OutputConfig,
) -> dict:
    if not 0 <= output_cfg.csv_compress_level <= 9:
        raise ValueError("CSV compression level must be between 0 and 9")
    if (
        not math.isfinite(output_cfg.detj_near_zero_threshold)
        or output_cfg.detj_near_zero_threshold < 0.0
    ):
        raise ValueError("detJ near-zero threshold must be finite and non-negative")

    rows = np.asarray(results)
    signed_det = np.asarray(detj_signed_delta_sigma, dtype=float)
    signed_det_theta = np.asarray(detj_signed_theta_sigma, dtype=float)
    if rows.ndim != 3 or rows.shape[-1] != len(CSV_HEADER):
        raise ValueError(
            f"Results must have shape (n_delta, n_sigma, {len(CSV_HEADER)})"
        )
    if signed_det.shape != rows.shape[:2]:
        raise ValueError("Signed detJ shape must match the result grid")
    if signed_det_theta.shape != rows.shape[:2]:
        raise ValueError("AT-space signed detJ shape must match the result grid")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compressed = output_cfg.compress_csv
    compression_method = "gzip" if compressed else "none"
    compression_level = output_cfg.csv_compress_level if compressed else None
    dataset_specs = [
        (
            "all",
            "all transformed interior grid points",
            CSV_HEADER,
        )
    ]
    if output_cfg.export_detj_split:
        dataset_specs.append(
            (
                "detJ_singular",
                DETJ_SPLIT_RULE,
                CSV_HEADER + SINGULAR_DIAGNOSTIC_COLUMNS,
            )
        )
        if output_cfg.export_detj_regular:
            dataset_specs.append(
                (
                    "detJ_regular",
                    "finite detJ_signed(delta,sigma/I) with abs(detJ) > threshold",
                    CSV_HEADER,
                )
            )

    paths = {
        dataset: output_dir
        / output_dataset_name(
            stem,
            ji,
            jf,
            dataset,
            compressed=compressed,
        )
        for dataset, _, _ in dataset_specs
    }
    counts = {dataset: 0 for dataset, _, _ in dataset_specs}
    observable_density_max = _observable_density_max(rows)
    started = perf_counter()

    with ExitStack() as stack:
        writers = {}
        for dataset, _, header in dataset_specs:
            handle = stack.enter_context(
                _open_csv_text(
                    paths[dataset],
                    compressed=compressed,
                    compresslevel=output_cfg.csv_compress_level,
                )
            )
            writer = csv.writer(handle)
            writer.writerow(header)
            writers[dataset] = writer

        for i, block in enumerate(rows):
            for j, row in enumerate(block):
                writers["all"].writerow(row)
                counts["all"] += 1

                if not output_cfg.export_detj_split:
                    continue

                detj_signed = float(signed_det[i, j])
                is_singular = (not math.isfinite(detj_signed)) or (
                    abs(detj_signed) <= output_cfg.detj_near_zero_threshold
                )
                if is_singular:
                    detj_signed_theta = float(signed_det_theta[i, j])
                    diagnostic_values = _singular_diagnostic_values(
                        row,
                        detj_signed,
                        detj_signed_theta,
                        output_cfg.detj_near_zero_threshold,
                        observable_density_max,
                    )
                    writers["detJ_singular"].writerow(
                        [*row, *diagnostic_values]
                    )
                    counts["detJ_singular"] += 1
                elif output_cfg.export_detj_regular:
                    writers["detJ_regular"].writerow(row)
                    counts["detJ_regular"] += 1

    write_elapsed = perf_counter() - started
    all_count = counts["all"]
    datasets = []
    for dataset, rule, _ in dataset_specs:
        path = paths[dataset]
        row_count = counts[dataset]
        datasets.append(
            {
                "dataset": dataset,
                "path": format_display_path(path),
                "file": path.name,
                "rule": rule,
                "row_count": row_count,
                "fraction": row_count / all_count if all_count else np.nan,
                "compressed": compressed,
                "compression_method": compression_method,
                "compression_level": compression_level,
                "file_size_bytes": path.stat().st_size if path.exists() else None,
                "write_time_seconds": write_elapsed,
            }
        )

    return {
        "detj_split_criterion": DETJ_SPLIT_CRITERION,
        "detj_near_zero_threshold": output_cfg.detj_near_zero_threshold,
        "nonfinite_policy": "non-finite detJ rows are included in detJ_singular",
        "density_threshold_policy": (
            "density-relative thresholds are diagnostic-only and do not affect split exports"
        ),
        "official_benchmark_dataset": "all",
        "one_pass_write_time_seconds": write_elapsed,
        "datasets": datasets,
    }
