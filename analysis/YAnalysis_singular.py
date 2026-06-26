from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DELTA_COL = "delta"
SIGMA_I_COL = "sigma/I"
THETA_COL = "ArcTan[delta](AT)"

PDF_DELTA_SIGMA_COL = "pdf density in delta and sigma/I dimensions"
PDF_THETA_SIGMA_COL = "pdf density in AT and sigma/I dimensions"

P_COL = "p"
ADO_COL = "ado"
GAUS2D_COL = "gaus2d"
JAC_DELTA_SIGMA_COL = "jacobian(delta,sigma/I)"

DETJ_SIGNED_DELTA_SIGMA_COL = "detJ_signed(delta,sigma/I)"
DETJ_ABS_DELTA_SIGMA_COL = "detJ_abs(delta,sigma/I)"
DETJ_SIGN_DELTA_SIGMA_COL = "detJ_sign(delta,sigma/I)"

DETJ_SIGNED_THETA_SIGMA_COL = "detJ_signed(ArcTan[delta](AT),sigma/I)"
DETJ_ABS_THETA_SIGMA_COL = "detJ_abs(ArcTan[delta](AT),sigma/I)"
DETJ_SIGN_THETA_SIGMA_COL = "detJ_sign(ArcTan[delta](AT),sigma/I)"

REQUIRED_COLUMNS = [
    DELTA_COL,
    SIGMA_I_COL,
    PDF_DELTA_SIGMA_COL,
    THETA_COL,
    PDF_THETA_SIGMA_COL,
    P_COL,
    ADO_COL,
    JAC_DELTA_SIGMA_COL,
    GAUS2D_COL,
]

OPTIONAL_DIAGNOSTIC_COLUMNS = [
    DETJ_SIGNED_DELTA_SIGMA_COL,
    DETJ_ABS_DELTA_SIGMA_COL,
    DETJ_SIGN_DELTA_SIGMA_COL,
    DETJ_SIGNED_THETA_SIGMA_COL,
    DETJ_ABS_THETA_SIGMA_COL,
    DETJ_SIGN_THETA_SIGMA_COL,
    "detJ_class",
    "detJ_near_zero_threshold",
    "detJ_split_rule",
    "observable_density_rel_to_max",
]

SIGN_COLUMN = DETJ_SIGN_DELTA_SIGMA_COL
SIGNED_DETJ_COLUMN = DETJ_SIGNED_DELTA_SIGMA_COL
DENSITY_REL_COLUMN = "observable_density_rel_to_max"


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def _with_density_relative_to_max(df: pd.DataFrame) -> pd.DataFrame:
    if DENSITY_REL_COLUMN in df.columns:
        df[DENSITY_REL_COLUMN] = pd.to_numeric(
            df[DENSITY_REL_COLUMN], errors="coerce"
        )
        return df

    gaus2d = pd.to_numeric(df["gaus2d"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(gaus2d)
    finite_max = float(np.max(gaus2d[finite])) if finite.any() else np.nan
    relative = np.full(gaus2d.shape, np.nan, dtype=float)
    if np.isfinite(finite_max) and finite_max > 0:
        relative[finite] = gaus2d[finite] / finite_max
    elif finite.any():
        relative[finite] = 0.0
    df[DENSITY_REL_COLUMN] = relative
    return df


def default_sign_col_for_xcol(xcol: str) -> str:
    if xcol == THETA_COL:
        return DETJ_SIGN_THETA_SIGMA_COL
    return DETJ_SIGN_DELTA_SIGMA_COL


def _signed_col_for_sign_col(sign_col: str) -> str:
    if sign_col == DETJ_SIGN_THETA_SIGMA_COL:
        return DETJ_SIGNED_THETA_SIGMA_COL
    return DETJ_SIGNED_DELTA_SIGMA_COL


def _sign_values(series: pd.Series) -> np.ndarray:
    text = series.astype("string").str.strip().str.lower().fillna("")
    text_values = text.to_numpy(dtype=str)
    sign = pd.to_numeric(text, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    aliases = {
        "+": 1.0,
        "+1": 1.0,
        "plus": 1.0,
        "positive": 1.0,
        "pos": 1.0,
        "-": -1.0,
        "-1": -1.0,
        "minus": -1.0,
        "negative": -1.0,
        "neg": -1.0,
        "0": 0.0,
        "+0": 0.0,
        "-0": 0.0,
        "zero": 0.0,
    }
    for label, value in aliases.items():
        sign[text_values == label] = value
    finite = np.isfinite(sign)
    sign[finite] = np.sign(sign[finite])
    return sign


def _derive_sign_column(
    df: pd.DataFrame, *, signed_col: str, sign_col: str
) -> pd.DataFrame:
    if sign_col in df.columns or signed_col not in df.columns:
        return df
    signed = pd.to_numeric(df[signed_col], errors="coerce").to_numpy(dtype=float)
    sign = np.full(signed.shape, np.nan, dtype=float)
    finite = np.isfinite(signed)
    sign[finite] = np.sign(signed[finite])
    df[sign_col] = sign
    return df


def _with_derived_sign(df: pd.DataFrame) -> pd.DataFrame:
    df = _derive_sign_column(
        df,
        signed_col=DETJ_SIGNED_DELTA_SIGMA_COL,
        sign_col=DETJ_SIGN_DELTA_SIGMA_COL,
    )
    df = _derive_sign_column(
        df,
        signed_col=DETJ_SIGNED_THETA_SIGMA_COL,
        sign_col=DETJ_SIGN_THETA_SIGMA_COL,
    )
    return df


def load_singular_csv(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    sample: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Load a sparse singular-Jacobian CSV and validate its original columns.

    When ``columns`` is supplied, the required original columns are retained as
    well so validation and derived-density calculation remain available.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Singular CSV not found: {path}")

    usecols = None
    if columns is not None:
        requested = list(columns)
        usecols = list(dict.fromkeys([*REQUIRED_COLUMNS, *requested]))

    df = pd.read_csv(path, usecols=usecols)
    _require_columns(df, REQUIRED_COLUMNS)
    df = _with_density_relative_to_max(df)
    df = _with_derived_sign(df)

    if sample is not None:
        sample = int(sample)
        if sample < 0:
            raise ValueError("sample must be non-negative or None.")
        if sample < len(df):
            df = df.sample(n=sample, random_state=random_state).reset_index(drop=True)
    return df


def _normalize_selection(values: Any) -> list[Any]:
    if isinstance(values, (str, bytes)) or np.isscalar(values):
        return [values]
    return list(values)


def _validate_range(value: Any, *, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{name} must contain finite values.")
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    return low, high


def _range_mask(series: pd.Series, value: Any, *, name: str) -> np.ndarray:
    checked = _validate_range(value, name=name)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if checked is None:
        return np.isfinite(numeric)
    low, high = checked
    return np.isfinite(numeric) & (numeric >= low) & (numeric <= high)


def _sign_masks(
    df: pd.DataFrame, sign_col: str = SIGN_COLUMN
) -> dict[str, np.ndarray]:
    signed_col = _signed_col_for_sign_col(sign_col)
    if sign_col in df.columns:
        sign = _sign_values(df[sign_col])
    elif signed_col in df.columns:
        signed = pd.to_numeric(df[signed_col], errors="coerce").to_numpy(dtype=float)
        sign = np.sign(signed)
        sign[~np.isfinite(signed)] = np.nan
    else:
        sign = np.full(len(df), np.nan, dtype=float)

    nonfinite = ~np.isfinite(sign)
    if signed_col in df.columns:
        signed = pd.to_numeric(df[signed_col], errors="coerce").to_numpy(dtype=float)
        nonfinite |= ~np.isfinite(signed)
    return {
        "plus": (sign > 0) & ~nonfinite,
        "minus": (sign < 0) & ~nonfinite,
        "zero": (sign == 0) & ~nonfinite,
        "nonfinite": nonfinite,
    }


class SingularTransitionData:
    def __init__(
        self,
        path_or_df: str | Path | pd.DataFrame,
        *,
        name: str | None = None,
        validate: bool = True,
    ) -> None:
        self.path: Path | None = None
        if isinstance(path_or_df, pd.DataFrame):
            self.df = path_or_df.copy()
            self.name = name or "singular_transition"
            if validate:
                _require_columns(self.df, REQUIRED_COLUMNS)
            self.df = _with_density_relative_to_max(self.df)
            self.df = _with_derived_sign(self.df)
        else:
            self.path = Path(path_or_df)
            self.df = load_singular_csv(self.path)
            filename = self.path.name
            for suffix in (".csv.gz", ".csv", ".gz"):
                if filename.endswith(suffix):
                    filename = filename[: -len(suffix)]
                    break
            self.name = name or filename
        if validate:
            _require_columns(self.df, REQUIRED_COLUMNS)

        self.diagnostic_columns = [
            column
            for column in OPTIONAL_DIAGNOSTIC_COLUMNS
            if column in self.df.columns
        ]

    def col(self, name: str) -> np.ndarray:
        _require_columns(self.df, [name])
        return self.df[name].to_numpy(dtype=float)

    def has_column(self, name: str) -> bool:
        return name in self.df.columns

    def optional_col(self, name: str) -> np.ndarray | None:
        if not self.has_column(name):
            return None
        return self.col(name)

    def text_col(self, name: str) -> pd.Series:
        _require_columns(self.df, [name])
        return self.df[name].astype("string")

    def detj_signed_delta_sigma(self) -> np.ndarray:
        return self.col(DETJ_SIGNED_DELTA_SIGMA_COL)

    def detj_abs_delta_sigma(self) -> np.ndarray:
        return self.col(DETJ_ABS_DELTA_SIGMA_COL)

    def detj_sign_delta_sigma(self) -> pd.Series:
        return self.text_col(DETJ_SIGN_DELTA_SIGMA_COL)

    def detj_signed_theta_sigma(self) -> np.ndarray:
        return self.col(DETJ_SIGNED_THETA_SIGMA_COL)

    def detj_abs_theta_sigma(self) -> np.ndarray:
        return self.col(DETJ_ABS_THETA_SIGMA_COL)

    def detj_sign_theta_sigma(self) -> pd.Series:
        return self.text_col(DETJ_SIGN_THETA_SIGMA_COL)

    def summary(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        columns = list(dict.fromkeys([*REQUIRED_COLUMNS, *self.diagnostic_columns]))
        for column in columns:
            series = self.df[column]
            row: dict[str, Any] = {
                "column": column,
                "dtype": str(series.dtype),
                "count": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            finite = numeric[np.isfinite(numeric)]
            row["finite"] = int(finite.size)
            row["min"] = float(np.min(finite)) if finite.size else np.nan
            row["max"] = float(np.max(finite)) if finite.size else np.nan
            row["mean"] = float(np.mean(finite)) if finite.size else np.nan
            row["std"] = float(np.std(finite)) if finite.size else np.nan
            rows.append(row)
        return pd.DataFrame(rows).set_index("column")

    def class_counts(self) -> pd.DataFrame:
        column = "detJ_class"
        if column not in self.df.columns:
            return pd.DataFrame(
                [{"detJ_class": "unavailable", "count": len(self.df), "fraction": 1.0}]
            )
        counts = self.df[column].fillna("nonfinite").astype(str).value_counts(dropna=False)
        result = counts.rename_axis("detJ_class").reset_index(name="count")
        result["fraction"] = result["count"] / max(len(self.df), 1)
        return result

    def sign_counts(self) -> pd.DataFrame:
        masks = _sign_masks(self.df)
        rows = [
            {"sign": label, "count": int(mask.sum())}
            for label, mask in masks.items()
        ]
        result = pd.DataFrame(rows)
        result["fraction"] = result["count"] / max(len(self.df), 1)
        return result

    def density_quantiles(self) -> pd.DataFrame:
        values = pd.to_numeric(
            self.df[DENSITY_REL_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        quantiles = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0])
        if values.size:
            quantile_values = np.quantile(values, quantiles)
        else:
            quantile_values = np.full(quantiles.shape, np.nan)
        return pd.DataFrame(
            {"quantile": quantiles, DENSITY_REL_COLUMN: quantile_values}
        )

    def filter_by_class(self, classes: Any) -> pd.DataFrame:
        if "detJ_class" not in self.df.columns:
            return self.df.iloc[0:0].copy()
        wanted = {str(value) for value in _normalize_selection(classes)}
        mask = self.df["detJ_class"].astype(str).isin(wanted)
        return self.df.loc[mask].copy()

    def filter_by_sign(self, signs: Any) -> pd.DataFrame:
        wanted = _normalize_selection(signs)
        masks = _sign_masks(self.df)
        aliases = {
            1: "plus",
            1.0: "plus",
            "+1": "plus",
            "+": "plus",
            "positive": "plus",
            "plus": "plus",
            -1: "minus",
            -1.0: "minus",
            "-1": "minus",
            "-": "minus",
            "negative": "minus",
            "minus": "minus",
            0: "zero",
            0.0: "zero",
            "0": "zero",
            "zero": "zero",
            "nonfinite": "nonfinite",
            "nan": "nonfinite",
        }
        selected = np.zeros(len(self.df), dtype=bool)
        for value in wanted:
            key = aliases.get(value, aliases.get(str(value).lower()))
            if key is None:
                raise ValueError(f"Unsupported sign selector: {value!r}")
            selected |= masks[key]
        return self.df.loc[selected].copy()

    def filter_by_density_rel(self, min_rel: float) -> pd.DataFrame:
        min_rel = float(min_rel)
        values = pd.to_numeric(
            self.df[DENSITY_REL_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
        mask = np.isfinite(values) & (values >= min_rel)
        return self.df.loc[mask].copy()

    def filter_by_sigma_range(self, value: Any) -> pd.DataFrame:
        return self.df.loc[
            _range_mask(self.df[SIGMA_I_COL], value, name="sigma range")
        ].copy()

    def filter_by_at_range(self, value: Any) -> pd.DataFrame:
        return self.df.loc[
            _range_mask(self.df[THETA_COL], value, name="AT range")
        ].copy()

    def filter_by_delta_range(self, value: Any) -> pd.DataFrame:
        return self.df.loc[
            _range_mask(self.df[DELTA_COL], value, name="delta range")
        ].copy()

    def filter_by_pado_range(
        self, ado_range: Any = None, p_range: Any = None
    ) -> pd.DataFrame:
        mask = np.ones(len(self.df), dtype=bool)
        if ado_range is not None:
            mask &= _range_mask(self.df[ADO_COL], ado_range, name="ADO range")
        if p_range is not None:
            mask &= _range_mask(self.df[P_COL], p_range, name="P range")
        return self.df.loc[mask].copy()

    def sample(self, n: int, random_state: int = 0) -> pd.DataFrame:
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n >= len(self.df):
            return self.df.copy()
        return self.df.sample(n=n, random_state=random_state).copy()


def suggest_bins(
    n_rows: int, *, min_bins: int = 150, max_bins: int = 500
) -> int:
    n_rows = max(int(n_rows), 0)
    min_bins = int(min_bins)
    max_bins = int(max_bins)
    if min_bins <= 0 or max_bins < min_bins:
        raise ValueError("Require 0 < min_bins <= max_bins.")
    estimate = int(round(np.sqrt(max(n_rows, 1))))
    return int(np.clip(estimate, min_bins, max_bins))


def compute_hist2d(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    bins: int | tuple[int, int] = 300,
    range: Any = None,
    weights: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_columns(df, [xcol, ycol])
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    weight_values = None
    if weights is not None:
        if isinstance(weights, str):
            _require_columns(df, [weights])
            weight_values = pd.to_numeric(df[weights], errors="coerce").to_numpy(
                dtype=float
            )
        else:
            weight_values = np.asarray(weights, dtype=float)
            if weight_values.shape != x.shape:
                raise ValueError(
                    f"weights must have shape {x.shape}; got {weight_values.shape}."
                )
        valid &= np.isfinite(weight_values)
        weight_values = weight_values[valid]

    return np.histogram2d(
        x[valid], y[valid], bins=bins, range=range, weights=weight_values
    )

def compute_sign_hist2d(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    bins: int | tuple[int, int] = 220,
    range: Any = None,
    sign_col: str | None = None,
) -> dict[str, np.ndarray]:
    _require_columns(df, [xcol, ycol])
    if sign_col is None:
        sign_col = default_sign_col_for_xcol(xcol)

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    coordinate_valid = np.isfinite(x) & np.isfinite(y)

    if not coordinate_valid.any():
        raise ValueError(f"No finite coordinates are available for {xcol!r} and {ycol!r}.")

    # Use one common range for all sign classes.
    # This is essential when range=None; otherwise each sign group would get
    # its own bin edges, and an empty group such as nonfinite would return 0..1.
    common_range = range
    if common_range is None:
        x_valid = x[coordinate_valid]
        y_valid = y[coordinate_valid]

        x_min, x_max = float(np.min(x_valid)), float(np.max(x_valid))
        y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))

        # Avoid zero-width ranges.
        if x_min == x_max:
            pad = max(abs(x_min) * 1e-6, 1e-12)
            x_min -= pad
            x_max += pad
        if y_min == y_max:
            pad = max(abs(y_min) * 1e-6, 1e-12)
            y_min -= pad
            y_max += pad

        common_range = ((x_min, x_max), (y_min, y_max))

    masks = _sign_masks(df, sign_col=sign_col)

    histograms: dict[str, np.ndarray] = {}
    xedges: np.ndarray | None = None
    yedges: np.ndarray | None = None

    for label in ("plus", "minus", "zero", "nonfinite"):
        selected = coordinate_valid & masks[label]
        hist, xedges, yedges = np.histogram2d(
            x[selected],
            y[selected],
            bins=bins,
            range=common_range,
        )
        histograms[label] = hist

    plus = histograms["plus"]
    minus = histograms["minus"]
    denominator = plus + minus

    mixed_fraction = np.zeros_like(denominator, dtype=float)
    np.divide(
        2.0 * np.minimum(plus, minus),
        denominator,
        out=mixed_fraction,
        where=denominator > 0,
    )

    histograms["total"] = sum(
        histograms[label] for label in ("plus", "minus", "zero", "nonfinite")
    )
    histograms["mixed_fraction"] = mixed_fraction
    histograms["xedges"] = xedges
    histograms["yedges"] = yedges
    return histograms