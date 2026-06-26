from pathlib import Path

from .compute.jacobian import (
    format_jacobian_report_markdown,
    format_jacobian_report_text,
)
from .config import (
    ADO_THETA_L_DEG,
    ADO_THETA_S_DEG,
    L1,
    L2,
    P_THETA_DEG,
)
from .pdf.fitters import format_pdf_model_report
from .path_display import format_display_path


def _format_transition_runtime_report(report: dict) -> str:
    values = report["input"]
    grid = report["grid"]
    theta_min, theta_max, theta_step = grid["theta_range"]
    sigma_i_min, sigma_i_max, sigma_i_step = grid["sigma_i_range"]
    padded_theta_min, padded_theta_max = grid["padded_theta_range"]
    padded_sigma_min, padded_sigma_max = grid["padded_sigma_i_range"]
    runtime = report["runtime"]
    p_model = report["p_model"]
    ado_model = report["ado_model"]
    lines = [
        "=" * 72,
        "Transition summary",
        f"  Ji={values['Ji']:.1f}",
        f"  Jf={values['Jf']:.1f}",
        f"  L1={L1}",
        f"  L2={L2}",
        "-" * 72,
        "Grid setup",
        f"  grid mode={grid['mode']}",
        f"  theta range=[{theta_min:.6g}, {theta_max:.6g}] step={theta_step:.6g}",
        (
            f"  sigma/I range=[{sigma_i_min:.6g}, {sigma_i_max:.6g}] "
            f"step={sigma_i_step:.6g}"
        ),
        (
            f"  padded theta range=[{padded_theta_min:.6g}, "
            f"{padded_theta_max:.6g}]"
        ),
        (
            f"  padded sigma/I range=[{padded_sigma_min:.6g}, "
            f"{padded_sigma_max:.6g}]"
        ),
        f"  P-ADO points shape={grid['pado_points_shape']}",
        f"  interior grid shape={grid['interior_grid_shape']}",
        f"  P model kind={p_model.get('kind', 'unknown')}",
        f"  ADO model kind={ado_model.get('kind', 'unknown')}",
        format_jacobian_report_text(report["jacobian"]["delta_sigma"]),
        format_pdf_model_report("P", p_model),
        "-" * 72,
        format_pdf_model_report("ADO", ado_model),
        "-" * 72,
        "Runtime",
        f"  fit_models = {runtime['fit_models']:.4f} s",
        f"  build P-ADO points = {runtime['build_pado_points']:.4f} s",
        f"  transform_pdf = {runtime['transform_pdf']:.4f} s",
        f"  total = {runtime['total']:.4f} s",
        "=" * 72,
    ]
    return "\n".join(lines)


def _escape_markdown_table_cell(value) -> str:
    return str(value).replace("|", "\\|")


def _markdown_table(rows) -> str:
    lines = ["| Setting | Value |", "| --- | ---: |"]
    lines.extend(
        f"| {_escape_markdown_table_cell(name)} | "
        f"{_escape_markdown_table_cell(value)} |"
        for name, value in rows
    )
    return "\n".join(lines)


def _format_pdf_model_markdown(name: str, model: dict) -> str:
    report_lines = format_pdf_model_report(name, model).splitlines()
    body = "\n".join(report_lines[1:])
    return f"### {name} PDF\n\n```text\n{body}\n```"


def _format_file_size(size) -> str:
    if size is None:
        return "unknown"
    return f"{int(size)} bytes"


def format_csv_export_text(export_manifest: dict) -> str:
    datasets = {
        entry["dataset"]: entry for entry in export_manifest.get("datasets", [])
    }
    lines = ["CSV export"]
    for dataset in ("all", "detJ_singular", "detJ_regular"):
        entry = datasets.get(dataset)
        if entry is not None:
            lines.append(
                f"  {dataset} = {format_display_path(entry['path'])} "
                f"({entry['row_count']} rows)"
            )
    lines.extend(
        [
            (
                "  detJ split criterion = "
                f"{export_manifest['detj_split_criterion']}"
            ),
            (
                "  detJ near-zero threshold = "
                f"{export_manifest['detj_near_zero_threshold']:.1e}"
            ),
        ]
    )
    all_entry = datasets.get("all")
    if all_entry is not None and all_entry["compressed"]:
        lines.append(
            f"  compression = gzip level {all_entry['compression_level']}"
        )
    else:
        lines.append("  compression = none")
    return "\n".join(lines)


def _format_csv_export_markdown(export_manifest: dict) -> str:
    lines = [
        "### CSV export",
        "",
        (
            "| dataset | file | rule | rows | fraction | compressed | "
            "file size | write time |"
        ),
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for entry in export_manifest.get("datasets", []):
        if entry["compressed"]:
            compression = f"yes, gzip level {entry['compression_level']}"
        else:
            compression = "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown_table_cell(entry["dataset"]),
                    f"`{_escape_markdown_table_cell(format_display_path(entry['path']))}`",
                    _escape_markdown_table_cell(entry["rule"]),
                    str(entry["row_count"]),
                    f"{float(entry['fraction']):.6g}",
                    compression,
                    _format_file_size(entry["file_size_bytes"]),
                    f"{float(entry['write_time_seconds']):.4f} s",
                ]
            )
            + " |"
        )

    datasets = {
        entry["dataset"]: entry for entry in export_manifest.get("datasets", [])
    }
    all_file = datasets.get("all", {}).get("file", "not written")
    singular_file = datasets.get("detJ_singular", {}).get("file", "not written")
    regular_file = datasets.get("detJ_regular", {}).get("file", "not written")
    lines.extend(
        [
            "",
            f"- Official benchmark file: `{all_file}`",
            f"- Diagnostic singular-response file: `{singular_file}`",
            f"- Diagnostic regular-only file: `{regular_file}`",
            (
                "- detJ split criterion: `"
                f"{export_manifest['detj_split_criterion']}`"
            ),
            (
                "- detJ near-zero threshold: `"
                f"{export_manifest['detj_near_zero_threshold']:.1e}`"
            ),
            "- Non-finite detJ rows are included in the singular export.",
            (
                "- Density-relative thresholds are diagnostic-only and do not "
                "affect split exports."
            ),
            "- The official benchmark uses the all-data file.",
            (
                "- CSV files were written in one pass; per-dataset write times "
                "report the shared export wall time."
            ),
        ]
    )
    return "\n".join(lines)


def format_transition_markdown_report(report: dict) -> str:
    values = report["input"]
    grid = report["grid"]
    runtime = report["runtime"]
    theta_min, theta_max, theta_step = grid["theta_range"]
    sigma_i_min, sigma_i_max, sigma_i_step = grid["sigma_i_range"]
    padded_theta_min, padded_theta_max = grid["padded_theta_range"]
    padded_sigma_min, padded_sigma_max = grid["padded_sigma_i_range"]

    input_table = _markdown_table(
        [
            ("Ji", f"{values['Ji']:.1f}"),
            ("Jf", f"{values['Jf']:.1f}"),
            ("P", f"{values['P']:.6g}"),
            ("P error left", f"{values['P_errL']:.6g}"),
            ("P error right", f"{values['P_errR']:.6g}"),
            ("ADO", f"{values['ADO']:.6g}"),
            ("ADO error left", f"{values['ADO_errL']:.6g}"),
            ("ADO error right", f"{values['ADO_errR']:.6g}"),
        ]
    )
    grid_table = _markdown_table(
        [
            ("grid mode", grid["mode"]),
            (
                "theta range",
                f"[{theta_min:.6g}, {theta_max:.6g}], step={theta_step:.6g}",
            ),
            (
                "sigma/I range",
                f"[{sigma_i_min:.6g}, {sigma_i_max:.6g}], step={sigma_i_step:.6g}",
            ),
            (
                "padded theta range",
                f"[{padded_theta_min:.6g}, {padded_theta_max:.6g}]",
            ),
            (
                "padded sigma/I range",
                f"[{padded_sigma_min:.6g}, {padded_sigma_max:.6g}]",
            ),
            ("P-ADO points shape", grid["pado_points_shape"]),
            ("interior grid shape", grid["interior_grid_shape"]),
            ("P model kind", report["p_model"].get("kind", "unknown")),
            ("ADO model kind", report["ado_model"].get("kind", "unknown")),
        ]
    )
    runtime_table = _markdown_table(
        [
            ("fit_models", f"{runtime['fit_models']:.4f} s"),
            ("build P-ADO points", f"{runtime['build_pado_points']:.4f} s"),
            ("transform_pdf", f"{runtime['transform_pdf']:.4f} s"),
            ("total", f"{runtime['total']:.4f} s"),
        ]
    )

    sections = [
        f"## Transition {report['label']}",
        "### Input data\n\n" + input_table,
        "### Grid setup\n\n" + grid_table,
    ]
    if "csv_export" in report:
        sections.append(_format_csv_export_markdown(report["csv_export"]))
    sections.extend(
        [
            "### Jacobian check",
            format_jacobian_report_markdown(
                report["jacobian"]["delta_sigma"],
                "#### Jacobian(delta,sigma/I)",
            ),
            format_jacobian_report_markdown(
                report["jacobian"]["theta_sigma"],
                "#### Jacobian(AT,sigma/I)",
            ),
            _format_pdf_model_markdown("P", report["p_model"]),
            _format_pdf_model_markdown("ADO", report["ado_model"]),
            "### Runtime\n\n" + runtime_table,
        ]
    )
    return "\n\n".join(sections)


def format_markdown_runtime_report(
    input_path,
    output_dir,
    mode,
    transition_reports,
    started_at,
    completed_at,
) -> str:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    heading = f"# {input_path.stem}_export"
    generated_files = []
    for report in transition_reports:
        for entry in report.get("csv_export", {}).get("datasets", []):
            file_name = entry.get("file")
            if file_name:
                generated_files.append(file_name)
    generated_files.append(f"{input_path.stem}_reports.md")
    generated_file_lines = "\n".join(
        f"  - {file_name}" for file_name in generated_files
    )
    run_information = "\n".join(
        [
            "## Run information",
            "",
            "| Setting | Value |",
            "| --- | --- |",
            f"| Input file name | {input_path.name} |",
            f"| Input path | {format_display_path(input_path)} |",
            f"| Output directory | {format_display_path(output_dir, trailing_slash=True)} |",
            f"| Grid mode | {mode} |",
            f"| Transition count | {len(transition_reports)} |",
            f"| Started | {started_at.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| Completed | {completed_at.strftime('%Y-%m-%d %H:%M:%S')} |",
            "",
            "Generated files:",
            generated_file_lines,
        ]
    )
    global_settings = "\n".join(
        [
            "## Global settings",
            "",
            "| Setting | Value |",
            "| --- | ---: |",
            f"| L1 | {L1} |",
            f"| L2 | {L2} |",
            f"| ADO_THETA_L_DEG | {ADO_THETA_L_DEG:.6g} degrees |",
            f"| ADO_THETA_S_DEG | {ADO_THETA_S_DEG:.6g} degrees |",
            f"| P_THETA_DEG | {P_THETA_DEG:.6g} degrees |",
        ]
    )
    transition_sections = [
        format_transition_markdown_report(report) for report in transition_reports
    ]
    return "\n\n".join([heading, run_information, global_settings, *transition_sections]) + "\n"
