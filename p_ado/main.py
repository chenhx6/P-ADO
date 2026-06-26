"""
Date: 2026-04-08 13:30:14
LastEditTime: 2026-06-03 11:01:47
"""

import argparse
import datetime
from pathlib import Path

from .compute.solver import solve_transition
from .runtime_report import (
    format_csv_export_text,
    format_markdown_runtime_report,
)
from .config import (
    ADO_THETA_L_DEG,
    ADO_THETA_S_DEG,
    CSV_COMPRESS_LEVEL,
    DETJ_NEAR_ZERO_THRESHOLD,
    P_THETA_DEG,
    GridConfig,
    OutputConfig,
    PdfConfig,
)
from .io.io import (
    _compression_level,
    _nonnegative_finite_float,
    output_markdown_name,
    read_transitions_dat,
    write_csv_exports,
    write_markdown_report,
)
from .path_display import format_display_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="P-ADO CSV generation for mixing-ratio analysis."
    )
    parser.add_argument("--input", required=True, help="Path to input .dat file")
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for output CSV files"
    )
    parser.add_argument(
        "--output-stem", default="result", help="Base stem for output CSV name"
    )
    parser.add_argument(
        "--mode", choices=["full", "test"], default="full", help="Grid mode"
    )
    parser.add_argument(
        "--verbose-pdf-fit",
        action="store_true",
        help="Print one PDF fitting summary block per transition.",
    )
    parser.add_argument(
        "--no-compress-csv",
        action="store_true",
        help="Write plain .csv files instead of gzip-compressed .csv.gz files.",
    )
    parser.add_argument(
        "--csv-compress-level",
        type=_compression_level,
        default=CSV_COMPRESS_LEVEL,
        metavar="INT",
        help=(
            "Gzip compression level from 0 to 9 "
            f"(default: {CSV_COMPRESS_LEVEL})."
        ),
    )
    parser.add_argument(
        "--no-detj-split-export",
        action="store_true",
        help="Write only the all-data CSV and skip detJ diagnostic split files.",
    )
    parser.add_argument(
        "--no-detj-regular-export",
        action="store_true",
        help="Skip only the detJ regular diagnostic CSV.",
    )
    parser.add_argument(
        "--detj-near-zero-threshold",
        type=_nonnegative_finite_float,
        default=DETJ_NEAR_ZERO_THRESHOLD,
        metavar="FLOAT",
        help="Absolute signed-detJ threshold used for diagnostic split exports.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started_at = datetime.datetime.now()
    grid_cfg = GridConfig(mode=args.mode)
    pdf_cfg = PdfConfig(verbose_pdf_fit=args.verbose_pdf_fit)
    output_cfg = OutputConfig(
        compress_csv=not args.no_compress_csv,
        csv_compress_level=args.csv_compress_level,
        export_detj_split=not args.no_detj_split_export,
        export_detj_regular=not args.no_detj_regular_export,
        detj_near_zero_threshold=args.detj_near_zero_threshold,
    )

    if args.verbose_pdf_fit:
        print("[info] verbose PDF fit reporting is enabled")

    transitions = read_transitions_dat(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file name: {Path(args.input).name}")
    print(f"Input path: {format_display_path(args.input)}")
    print(f"Output directory: {format_display_path(output_dir, trailing_slash=True)}")
    print(f"ADO_THETA_S_DEG = {ADO_THETA_S_DEG} degrees")
    print(f"ADO_THETA_L_DEG = {ADO_THETA_L_DEG} degrees")
    print(f"P_THETA_DEG = {P_THETA_DEG} degrees")
    print(
    f"sigma/I: from {grid_cfg.sigma_i_range()[0]} to {grid_cfg.sigma_i_range()[1]}, step {grid_cfg.sigma_i_range()[2]}"
    )
    print(
    f"theta: from {grid_cfg.theta_range()[0]} to {grid_cfg.theta_range()[1]}, step {grid_cfg.theta_range()[2]}"
    )
    print(f"Total transitions to process: {len(transitions)}")

    transition_reports = []
    for idx, tr in enumerate(transitions, start=1):
        print(
            f"[{idx}/{len(transitions)}] "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(
            f"- Ji={tr.ji:.1f}, Jf={tr.jf:.1f}, "
            f"P={tr.p_value:.3f}, P_errL={tr.p_err_l:.3f}, P_errR={tr.p_err_r:.3f}, "
            f"ADO={tr.ado_value:.3f}, "
            f"ADO_errL={tr.ado_err_l:.3f}, ADO_errR={tr.ado_err_r:.3f}"
        )
        print(f"- Solving transition {tr.label}...")
        results, transition_report, export_data = solve_transition(
            tr,
            grid_cfg,
            pdf_cfg,
            return_report=True,
            return_array=True,
            return_export_data=True,
            detj_near_zero_threshold=output_cfg.detj_near_zero_threshold,
        )
        print("- Writing CSV exports...")
        export_manifest = write_csv_exports(
            output_dir=output_dir,
            stem=args.output_stem,
            ji=tr.ji,
            jf=tr.jf,
            results=results,
            detj_signed_delta_sigma=export_data["detj_signed_delta_sigma"],
            detj_signed_theta_sigma=export_data["detj_signed_theta_sigma"],
            output_cfg=output_cfg,
        )
        del results, export_data
        transition_report["csv_export"] = export_manifest
        transition_reports.append(transition_report)
        for dataset in export_manifest["datasets"]:
            print(f"- Saved {dataset['dataset']}: {dataset['path']}")
        if args.verbose_pdf_fit:
            print(format_csv_export_text(export_manifest))

    completed_at = datetime.datetime.now()
    markdown_name = output_markdown_name(args.input)
    markdown_path = output_dir / markdown_name
    markdown_content = format_markdown_runtime_report(
        input_path=args.input,
        output_dir=output_dir,
        mode=args.mode,
        transition_reports=transition_reports,
        started_at=started_at,
        completed_at=completed_at,
    )
    write_markdown_report(markdown_path, markdown_content)
    print(f"Markdown report saved: {format_display_path(markdown_path)}")

    print(
        f"All transitions processed.\n{completed_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
