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

## Numerical Jacobian evaluation and lower-boundary diagnostics

The probability-density transformation in P-ADO uses a numerical Jacobian evaluated on the response-function grid. This section documents the implementation used by the code, the centered finite-difference scheme, and the diagnostic treatment of the lower-boundary region. The diagnostic is intended to record the adopted numerical boundary and response behavior; it does not assign the lower-boundary behavior to either the numerical implementation or the physical formalism.

For the density transformation, the code evaluates the signed Jacobian determinant in the observable ordering used internally by the implementation:

```math
\det J_{\delta,\sigma/I}
=
\det
\frac{\partial(P,R_{\mathrm{ADO}})}
{\partial(\delta,\sigma/I)} .
```

At an interior grid point $(i,j)$, with

```math
R_{i,j}=R_{\mathrm{ADO}}(\delta_i,(\sigma/I)_j),
\qquad
P_{i,j}=P(\delta_i,(\sigma/I)_j),
```

the centered finite-difference derivatives are

```math
\left.
\frac{\partial R_{\mathrm{ADO}}}{\partial \delta}
\right|_{i,j}
\approx
\frac{R_{i+1,j}-R_{i-1,j}}
{\delta_{i+1}-\delta_{i-1}},
\qquad
\left.
\frac{\partial R_{\mathrm{ADO}}}{\partial(\sigma/I)}
\right|_{i,j}
\approx
\frac{R_{i,j+1}-R_{i,j-1}}
{(\sigma/I)_{j+1}-(\sigma/I)_{j-1}},
```

```math
\left.
\frac{\partial P}{\partial \delta}
\right|_{i,j}
\approx
\frac{P_{i+1,j}-P_{i-1,j}}
{\delta_{i+1}-\delta_{i-1}},
\qquad
\left.
\frac{\partial P}{\partial(\sigma/I)}
\right|_{i,j}
\approx
\frac{P_{i,j+1}-P_{i,j-1}}
{(\sigma/I)_{j+1}-(\sigma/I)_{j-1}}.
```

The signed determinant used for diagnostics is therefore

```math
\det J_{\delta,\sigma/I}\big|_{i,j}
\approx
\frac{P_{i+1,j}-P_{i-1,j}}
{\delta_{i+1}-\delta_{i-1}}
\frac{R_{i,j+1}-R_{i,j-1}}
{(\sigma/I)_{j+1}-(\sigma/I)_{j-1}}
-
\frac{P_{i,j+1}-P_{i,j-1}}
{(\sigma/I)_{j+1}-(\sigma/I)_{j-1}}
\frac{R_{i+1,j}-R_{i-1,j}}
{\delta_{i+1}-\delta_{i-1}} .
```

The transformed density uses the absolute value $|\det J_{\delta,\sigma/I}|$. Therefore, using the alternative observable ordering $(R_{\mathrm{ADO}},P)$ changes only the global sign of the signed determinant and does not change the transformed density. The sign is retained only for diagnostic output.

Boundary points cannot support centered finite differences and are excluded from the Jacobian-transformed output. The transformed output corresponds to the interior response grid,

```text
pado_points[1:-1, 1:-1, :]
```

The all-data and regular diagnostic files include the absolute determinant used in the transformation. The code also evaluates the analogous determinant in $(AT,\sigma/I)$ coordinates, where $AT=\arctan(\delta)$, for diagnostic reporting and export.

Rows are written to the singular diagnostic file when

```text
nonfinite or abs(detJ_signed(delta,sigma/I)) <= detJ_near_zero_threshold
```

Rows with finite `detJ_signed(delta,sigma/I)` and an absolute value above the threshold are written to the regular diagnostic file when regular export is enabled. The default near-zero threshold is `1e-12`, defined by `DETJ_NEAR_ZERO_THRESHOLD` in `p_ado/config.py`; it can be overridden with `--detj-near-zero-threshold`.

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

For the representative $^{187}$Au transition, the response grid was configured over

```math
AT=\arctan(\delta)\in[-89^\circ,89^\circ]
```

with a step of $0.01^\circ$, and over

```math
\sigma/I\in[0.039,1.0]
```

with a step of 0.001. Because centered finite differences require neighboring grid points, the first transformed point used in the Jacobian evaluation was $\sigma/I=0.04$.

The transformed interior grid contained 17086080 points. All 17086080 Jacobian values were finite, with zero non-finite entries, zero exactly vanishing entries, and zero near-zero entries under the threshold $1.0\times10^{-12}$. The minimum value of $|\det J|$ was $9.26148\times10^{-11}$, and no sign-change edges were found along either the $\delta$ or $\sigma/I$ direction. The diagnostic was therefore accepted. Additional density-threshold checks from $10^{-6}$ to $10^{-2}$ also gave zero near-zero entries and accepted status.

In the adopted alignment convention, $\sigma=0$ corresponds to complete alignment. The very-low-$\sigma/I$ side is therefore the complete-alignment-side limiting region of the response calculation. P-ADO treats this as a documented lower-boundary behavior of the numerical response calculation, rather than assigning it specifically to either the implementation or the formalism. Additional numerical checks for both integer and half-integer spins showed that this lower-boundary behavior shifts to smaller $\sigma/I$ as the initial spin $I$ increases, consistent with its dependence on the absolute width parameter $\sigma=(\sigma/I)I$.

## Notes on computational cost

`full` mode uses a fine grid and can produce large compressed CSV files. Start with `--mode test` to confirm the input and output setup, then run `--mode full` for production calculations.
