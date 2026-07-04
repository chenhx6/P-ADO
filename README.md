# P-ADO

## Project overview

P-ADO is a Python research codebase for mixing-ratio analysis using polarization and angular-distribution observables. The repository contains the calculation package, plotting and analysis notebooks, tests, example `.dat` input files, and release metadata for GitHub and Science Data Bank.

## What is P-ADO?

P-ADO maps transition-level input observables into probability-density output tables in delta-sigma/I, AT-sigma/I, and observable spaces. The all-data CSV is the primary calculation output and official benchmark dataset; optional `detJ_singular` and `detJ_regular` files provide diagnostic splits based on the Jacobian determinant.

## Supported environment

Python 3.8 or later is required. Ubuntu 20.04 or later is recommended. On Ubuntu 18.04, please install Python 3.8 explicitly; the default Python 3.6 environment is not supported.

## Repository structure

```text
p_ado/      Calculation package, CLI entry point, example input, and core requirements.
analysis/   Analysis helpers, plotting helpers, and notebooks.
tests/      Pytest tests for the calculation package.
```

## Installation

### Simplest no-install run

For a simple first test, you do not need to install P-ADO as a Python package. Install only the required Python dependencies and run the program from the repository root:

```bash
python -m pip install -r p_ado/requirements.txt
python -m p_ado.main --help
python -m p_ado.main --input p_ado/example_input.dat --output-dir outputs/example_input --mode test
```

This method is recommended for users who only want to run the example or reproduce the calculation from the repository directory.

### Optional editable installation

For long-term use, or if you want to run P-ADO from other directories, you may install the project in editable mode:

```bash
python -m pip install -e .
python -m p_ado --help
```

For notebooks and plotting workflows, install the analysis dependencies separately:

```bash
python -m pip install -r analysis/requirements_analysis.txt
```

## Quick start

From the repository root, the simplest calculation check is:

```bash
python -m p_ado.main --help
python -m p_ado.main --input p_ado/example_input.dat --output-dir outputs/example_input --mode test
```

The command above writes CSV files and a markdown runtime report under `outputs/example_input/`.

## Calculation workflow

Prepare a whitespace-delimited `.dat` input file with transition rows, then run the P-ADO CLI in `test` mode for a small verification run or `full` mode for the full production grid. The calculation package writes an all-data CSV file, optional Jacobian diagnostic split files, and a markdown runtime report in the selected output directory.

See `p_ado/README.md` for detailed input/output definitions and Jacobian diagnostic rules.

## Analysis workflow

Use the ordinary analysis helpers and `YExtract.ipynb` with `*_all.csv` or `*_all.csv.gz` files. Use `analysis/YAll_transition_results.ipynb` for batch projection summaries across multiple ordinary P-ADO transition output files. Use the singular diagnostic helpers and `YExtract_singular.ipynb` with `*_detJ_singular.csv` or `*_detJ_singular.csv.gz` files.

See `analysis/README.md` before applying grid-based plotting routines to regular diagnostic files.

## AI assistance disclosure

Parts of this project were developed with assistance from OpenAI Codex. The repository author reviewed, tested, and curated the generated or suggested code before public release. All scientific assumptions, implementation choices, validation, and final responsibility for the released software remain with the repository author.

## Citation

Citation metadata is provided in `CITATION.cff`. The author list, repository URL, and any future Science Data Bank DOI must be completed before formal release.

## License

P-ADO is distributed under the MIT License. See `LICENSE`.

## Notes for GitHub and Science Data Bank release

Before release, confirm the copyright holder, complete `CITATION.cff`, add the final repository URL, add the Science Data Bank DOI after it is assigned, and verify the version number is consistent across release records.
