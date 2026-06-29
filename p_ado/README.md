# P-ADO Calculation Package

## Purpose

The `p_ado` package runs the P-ADO calculation workflow for mixing-ratio analysis using polarization and angular-distribution observables. It reads transition input from a `.dat` file and writes CSV exports plus a markdown runtime report.

## Supported environment

Python 3.8 or later is required. Ubuntu 20.04 or later is recommended. On Ubuntu 18.04, please install Python 3.8 explicitly; the default Python 3.6 environment is not supported.

## Input file format

Input files are whitespace-delimited `.dat` files, not CSV files. Empty lines and lines beginning with `#` are ignored. The program reads at least eight columns and uses the absolute value of the four uncertainty columns.

## Input columns

```text
Ji Jf Pvalue PValueERRL PValueERRR ADOValue ADOValueERRL ADOValueERRR
```

## Example input

```text
# Ji  Jf  Pvalue  PValueERRL  PValueERRR  ADOValue  ADOValueERRL  ADOValueERRR
10.5 9.5 -0.03 0.13 0.05 0.48 0.06 0.06
```

The repository includes `example_input.dat` and `input.dat`.

## Simplest run from the repository root

From the repository root, install the calculation dependencies:

```bash
python -m pip install -r p_ado/requirements.txt
```

Then run the example input file:

```bash
python -m p_ado.main --input p_ado/example_input.dat --output-dir outputs/example_input --mode test
```

This command does not require installing P-ADO itself. It only requires that the command is executed from the repository root.

## Optional editable installation

For long-term use, or if you want to run P-ADO from another directory, install the project in editable mode from the repository root:

```bash
python -m pip install -e .
python -m p_ado --help
```

## How to run test mode

```bash
python -m p_ado.main --input p_ado/example_input.dat --output-dir outputs/example_input --output-stem example_input --mode test
```

`test` mode uses a smaller grid and is intended for quick verification.

## How to run full mode

```bash
python -m p_ado.main --input p_ado/input.dat --output-dir outputs/input --output-stem input --mode full
```

`full` mode uses the production grid settings defined in `config.py` and can be computationally expensive.

## Output files

For each transition, P-ADO writes:

```text
<stem>_<Ji>_<Jf>_all.csv.gz
<stem>_<Ji>_<Jf>_detJ_singular.csv.gz
<stem>_<Ji>_<Jf>_detJ_regular.csv.gz
```

The markdown runtime report is written to the same output directory:

```text
<input_stem>_reports.md
```

Compression is enabled by default. Use `--no-compress-csv` to write plain `.csv` files.

## CSV column definitions

The all-data and regular diagnostic files use these columns:

```text
delta
sigma/I
pdf density in delta and sigma/I dimensions
ArcTan[delta](AT)
pdf density in AT and sigma/I dimensions
p
ado
jacobian(delta,sigma/I)
gaus2d
jacobian(ArcTan[delta](AT),sigma/I)
```

The singular diagnostic file appends signed-Jacobian diagnostic columns, including signed and absolute determinant values, determinant sign labels, `detJ_class`, `detJ_near_zero_threshold`, `detJ_split_rule`, and `observable_density_rel_to_max`.

## Runtime report

The runtime report records the input file name, relative input path, output directory, generated file names, grid mode, transition count, global settings, fitted PDF summaries, Jacobian diagnostics, CSV export metadata, and runtime timings. Paths are displayed as relative paths or file names so reports are portable.

## Numerical Jacobian evaluation and regular/singular diagnostics

The Jacobian determinant used in the density transformation is evaluated numerically on the response-function grid by centered finite differences in `transform_pdf_with_central_difference()`. For the `(delta, sigma/I)` coordinates, the code estimates the local derivatives of `(P, R_ADO)` and forms the signed determinant using:

```text
detJ_signed(delta,sigma/I) = dP/ddelta * dR_ADO/d(sigma/I) - dP/d(sigma/I) * dR_ADO/ddelta
```

The transformed density uses the absolute determinant, so the sign is mainly used for diagnostics. The code also evaluates the analogous signed determinant in `(ArcTan[delta](AT), sigma/I)` coordinates for diagnostic reporting and export.

Boundary points cannot support centered finite differences and are therefore excluded from the Jacobian-transformed output. The output corresponds to the interior response grid, `pado_points[1:-1, 1:-1, :]`.

The diagnostic split uses `detJ_signed(delta,sigma/I)`, not the AT-space determinant. Rows are written to the singular diagnostic file when:

```text
nonfinite or abs(detJ_signed(delta,sigma/I)) <= detJ_near_zero_threshold
```

Rows with finite `detJ_signed(delta,sigma/I)` and an absolute value above the threshold are written to the regular diagnostic file when regular export is enabled. The default threshold is `1e-12`, defined by `DETJ_NEAR_ZERO_THRESHOLD` in `p_ado/config.py`; use `--detj-near-zero-threshold` to override it.

The singular diagnostic export appends these columns:

```text
detJ_signed(delta,sigma/I)
detJ_abs(delta,sigma/I)
detJ_sign(delta,sigma/I)
detJ_signed(ArcTan[delta](AT),sigma/I)
detJ_abs(ArcTan[delta](AT),sigma/I)
detJ_sign(ArcTan[delta](AT),sigma/I)
detJ_class
detJ_near_zero_threshold
detJ_split_rule
observable_density_rel_to_max
```

The all-data CSV remains the official benchmark dataset. The `detJ_singular` and `detJ_regular` files are diagnostic split exports.

## Notes on computational cost

`full` mode uses a fine grid and can produce large compressed CSV files. Start with `--mode test` to confirm the input and output setup, then run `--mode full` for production calculations.
