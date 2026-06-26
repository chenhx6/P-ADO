"""
Standalone P-ADO single-sigma plotting script.

This file is intentionally self-contained. It does not import p_ado.
It scans theta = arctan(delta) at a fixed sigma/I and plots the corresponding
P-ADO curve for comparison with experimental data.

Edit the configuration constants at the top of this file to change sigma/I,
theta range, detector angles, output paths, and plotting options.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# User configuration
# =========================

JI = 5.5
JF = 4.5

SIGMA_I = 0.003

THETA_MIN = -89.0
THETA_MAX = 89.0
THETA_STEP = 0.1

ADO_SMALL_ANGLE_DEG = 26.0
ADO_LARGE_ANGLE_DEG = 90.0
P_ANGLE_DEG = 90.0

OUTPUT_FIGURE = "P-ADO_single.png"
SAVE_FIGURE = True
SHOW_FIGURE = True
DPI = 300

SAVE_CSV = False
CSV_OUTPUT = "P-ADO_single_points.csv"

MARKER_SIZE = 6.0
DRAW_LINE = False

SHOW_COLORBAR = False
COLOR_BY_THETA = True


# =========================
# Physics constants
# =========================

# From p_ado/config.py
L1 = 1
L2 = 2
UNDERFLOW_CUTOFF = -745.0

TOL = 1e-10


# =========================
# Small angular-momentum helpers
# =========================

# These functions replace the scipy/sympy calls used by p_ado/physics.
# They are deliberately plain and local so this script can run by itself.


def is_close_to_integer(value):
    return abs(value - round(value)) < TOL


def phase_from_integer_exponent(exponent):
    """Return (-1)**exponent for exponents that should be integers."""
    rounded = round(exponent)
    if abs(exponent - rounded) > 1e-8:
        raise ValueError(f"Expected an integer exponent, got {exponent!r}")
    return -1.0 if int(rounded) % 2 else 1.0


def ceil_int(value):
    return int(math.ceil(value - 1e-10))


def floor_int(value):
    return int(math.floor(value + 1e-10))


def triangle_delta_sqrt(a, b, c):
    """Square-root triangle factor used by Wigner 3j and 6j symbols."""
    terms = (a + b - c, a - b + c, -a + b + c)
    if any(term < -TOL for term in terms):
        return 0.0

    log_value = (
        math.lgamma(terms[0] + 1.0)
        + math.lgamma(terms[1] + 1.0)
        + math.lgamma(terms[2] + 1.0)
        - math.lgamma(a + b + c + 2.0)
    )
    return math.exp(0.5 * log_value)


def wigner_3j(j1, j2, j3, m1, m2, m3):
    """Compute a Wigner 3j symbol with the Racah formula."""
    if abs(m1 + m2 + m3) > 1e-8:
        return 0.0
    if any(j < -TOL for j in (j1, j2, j3)):
        return 0.0
    if abs(m1) > j1 + TOL or abs(m2) > j2 + TOL or abs(m3) > j3 + TOL:
        return 0.0

    delta = triangle_delta_sqrt(j1, j2, j3)
    if delta == 0.0:
        return 0.0

    prefactor_log = 0.5 * (
        math.lgamma(j1 + m1 + 1.0)
        + math.lgamma(j1 - m1 + 1.0)
        + math.lgamma(j2 + m2 + 1.0)
        + math.lgamma(j2 - m2 + 1.0)
        + math.lgamma(j3 + m3 + 1.0)
        + math.lgamma(j3 - m3 + 1.0)
    )
    prefactor = phase_from_integer_exponent(j1 - j2 - m3)
    prefactor *= delta * math.exp(prefactor_log)

    z_min = ceil_int(max(0.0, j2 - j3 - m1, j1 - j3 + m2))
    z_max = floor_int(min(j1 + j2 - j3, j1 - m1, j2 + m2))
    if z_min > z_max:
        return 0.0

    total = 0.0
    for z in range(z_min, z_max + 1):
        denom_args = (
            z,
            j1 + j2 - j3 - z,
            j1 - m1 - z,
            j2 + m2 - z,
            j3 - j2 + m1 + z,
            j3 - j1 - m2 + z,
        )
        if any(arg < -TOL for arg in denom_args):
            continue
        denom_log = sum(math.lgamma(arg + 1.0) for arg in denom_args)
        total += phase_from_integer_exponent(z) * math.exp(-denom_log)

    return prefactor * total


def clebsch_gordan_float(j1, j2, j3, m1, m2, m3):
    """Float version of sympy.physics.wigner.clebsch_gordan."""
    if abs(m1 + m2 - m3) > 1e-8:
        return 0.0
    return (
        phase_from_integer_exponent(j1 - j2 + m3)
        * math.sqrt(2.0 * j3 + 1.0)
        * wigner_3j(j1, j2, j3, m1, m2, -m3)
    )


def wigner_6j(j1, j2, j3, j4, j5, j6):
    """Compute a Wigner 6j symbol with the Racah formula."""
    delta_product = (
        triangle_delta_sqrt(j1, j2, j3)
        * triangle_delta_sqrt(j1, j5, j6)
        * triangle_delta_sqrt(j4, j2, j6)
        * triangle_delta_sqrt(j4, j5, j3)
    )
    if delta_product == 0.0:
        return 0.0

    lower = (
        j1 + j2 + j3,
        j1 + j5 + j6,
        j4 + j2 + j6,
        j4 + j5 + j3,
    )
    upper = (
        j1 + j2 + j4 + j5,
        j2 + j3 + j5 + j6,
        j3 + j1 + j6 + j4,
    )
    z_min = ceil_int(max(lower))
    z_max = floor_int(min(upper))
    if z_min > z_max:
        return 0.0

    total = 0.0
    for z in range(z_min, z_max + 1):
        denom_args = (
            z - lower[0],
            z - lower[1],
            z - lower[2],
            z - lower[3],
            upper[0] - z,
            upper[1] - z,
            upper[2] - z,
        )
        if any(arg < -TOL for arg in denom_args):
            continue
        term_log = math.lgamma(z + 2.0) - sum(
            math.lgamma(arg + 1.0) for arg in denom_args
        )
        total += phase_from_integer_exponent(z) * math.exp(term_log)

    return delta_product * total


# =========================
# Independent physics formulas
# =========================


def theta_deg_to_delta(theta_deg):
    """From p_ado/physics/delta.py: delta = tan(theta)."""
    return math.tan(math.radians(theta_deg))


def sigma_from_sigma_over_i(ji, sigma_i):
    """From p_ado/physics/sigmaOverI.py."""
    return float(ji) * float(sigma_i)


def pop_dist(m, sigma):
    """Unnormalized magnetic-substate population distribution."""
    if sigma <= 0.0:
        return 0.0
    arg = -(float(m) ** 2) / (2.0 * (float(sigma) ** 2))
    if arg <= UNDERFLOW_CUTOFF:
        return 0.0
    return math.exp(arg)


def pop_dist_norm_coeff(j, sigma):
    """Normalization of the substate population distribution."""
    if is_close_to_integer(j):
        j_int = int(round(j))
        return 2.0 * sum(pop_dist(i, sigma) for i in range(1, j_int + 1)) + pop_dist(
            0.0, sigma
        )

    total = 0.0
    m = 0.5
    while m <= j + TOL:
        total += pop_dist(m, sigma)
        m += 1.0
    return 2.0 * total


def calculate_alignment(j, k, sigma):
    """Alignment parameter A_k from p_ado/physics/sigmaOverI.py."""
    pop_norm = pop_dist_norm_coeff(j, sigma)
    if pop_norm == 0.0:
        return 0.0

    coeff = math.sqrt(2.0 * j + 1.0) / pop_norm

    def align_term(m):
        return (
            phase_from_integer_exponent(j - m)
            * clebsch_gordan_float(j, j, k, m, -m, 0.0)
            * pop_dist(m, sigma)
        )

    m = 1.0 if is_close_to_integer(j) else 0.5
    total = 0.0
    while m <= j + TOL:
        total += align_term(m)
        m += 1.0

    if is_close_to_integer(j):
        total += 0.5 * align_term(0.0)

    return 2.0 * coeff * total


def legendre_p2(x):
    """Replacement for scipy.special.eval_legendre(2, x)."""
    return 0.5 * (3.0 * x * x - 1.0)


def legendre_p4(x):
    """Replacement for scipy.special.eval_legendre(4, x)."""
    x2 = x * x
    return (35.0 * x2 * x2 - 30.0 * x2 + 3.0) / 8.0


def associated_p22(x):
    """The associated Legendre term used in p_ado/physics/pado.py."""
    return 3.0 * (1.0 - x * x)


def associated_p42(x):
    """The associated Legendre term used in p_ado/physics/pado.py."""
    return 7.5 * (1.0 - x * x) * (7.0 * x * x - 1.0)


def racah_coeff_like_pado(ji, l1, l2, k, jf):
    """Racah coefficient arranged exactly as in p_ado/physics/pado.py."""
    phase = phase_from_integer_exponent(ji + ji + l1 + l2)
    return phase * wigner_6j(ji, ji, k, l2, l1, jf)


def f_coeff(jf, l1, l2, ji, k):
    """F_k coefficient from p_ado/physics/pado.py."""
    phase = phase_from_integer_exponent(jf - ji - 1.0)
    prefactor = math.sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) * (2.0 * ji + 1.0))
    cg = clebsch_gordan_float(l1, l2, k, 1.0, -1.0, 0.0)
    rac = racah_coeff_like_pado(ji, l1, l2, k, jf)
    return phase * prefactor * cg * rac


def prepare_transition(
    ji,
    jf,
    ado_small_angle_deg=ADO_SMALL_ANGLE_DEG,
    ado_large_angle_deg=ADO_LARGE_ANGLE_DEG,
    p_angle_deg=P_ANGLE_DEG,
):
    """Precompute angle and transition constants for one Ji -> Jf transition."""
    x_s = math.cos(math.radians(ado_small_angle_deg))
    x_l = math.cos(math.radians(ado_large_angle_deg))
    x_p = math.cos(math.radians(p_angle_deg))

    return {
        "legS2": legendre_p2(x_s),
        "legS4": legendre_p4(x_s),
        "legL2": legendre_p2(x_l),
        "legL4": legendre_p4(x_l),
        "legP2": legendre_p2(x_p),
        "legP4": legendre_p4(x_p),
        "assocP22": associated_p22(x_p),
        "assocP42": associated_p42(x_p),
        "f11k2": f_coeff(jf, L1, L1, ji, 2),
        "f12k2": f_coeff(jf, L1, L2, ji, 2),
        "f22k2": f_coeff(jf, L2, L2, ji, 2),
        "f11k4": f_coeff(jf, L1, L1, ji, 4),
        "f12k4": f_coeff(jf, L1, L2, ji, 4),
        "f22k4": f_coeff(jf, L2, L2, ji, 4),
    }


def calculate_angular_terms(delta, align2, align4, prepared):
    """Shared numerator terms for W(theta), P, and R_ADO."""
    d2 = delta * delta
    denom = 1.0 + d2

    leg2 = align2 * (
        (prepared["f11k2"] + 2.0 * delta * prepared["f12k2"] + d2 * prepared["f22k2"])
        / denom
    )
    leg4 = align4 * (
        (prepared["f11k4"] + 2.0 * delta * prepared["f12k4"] + d2 * prepared["f22k4"])
        / denom
    )

    assoc22 = align2 * (
        (
            0.5 * prepared["f11k2"]
            - (1.0 / 3.0) * delta * prepared["f12k2"]
            + 0.5 * d2 * prepared["f22k2"]
        )
        / denom
    )
    assoc42 = align4 * (((-1.0 / 12.0) * d2 * prepared["f22k4"]) / denom)

    return leg2, leg4, assoc22, assoc42


def calculate_ws(leg2, leg4, prepared):
    """Angular distribution W at the small ADO detector angle."""
    return 1.0 + leg2 * prepared["legS2"] + leg4 * prepared["legS4"]


def calculate_wl(leg2, leg4, prepared):
    """Angular distribution W at the large ADO detector angle."""
    return 1.0 + leg2 * prepared["legL2"] + leg4 * prepared["legL4"]


def calculate_p(leg2, leg4, assoc22, assoc42, prepared):
    """Linear polarization P from p_ado/physics/pado.py."""
    numerator = assoc22 * prepared["assocP22"] + assoc42 * prepared["assocP42"]
    denominator = 1.0 + leg2 * prepared["legP2"] + leg4 * prepared["legP4"]
    return numerator / denominator


def calculate_rado(ws, wl):
    """ADO/R_ADO ratio."""
    return ws / wl


def calculate_pado_point(theta_deg, sigma_i, ji, jf, prepared, align2, align4):
    """Calculate one row: theta, delta, sigma/I, P, and R_ADO."""
    del jf  # Kept in the signature so the call mirrors the transition setup.
    delta = theta_deg_to_delta(theta_deg)
    leg2, leg4, assoc22, assoc42 = calculate_angular_terms(
        delta, align2, align4, prepared
    )
    p_value = calculate_p(leg2, leg4, assoc22, assoc42, prepared)
    ws = calculate_ws(leg2, leg4, prepared)
    wl = calculate_wl(leg2, leg4, prepared)
    rado = calculate_rado(ws, wl)

    return {
        "theta_deg": float(theta_deg),
        "delta": float(delta),
        "sigma/I": float(sigma_i),
        "p": float(p_value),
        "ado": float(rado),
    }


def generate_pado_points(
    ji,
    jf,
    sigma_i,
    theta_min,
    theta_max,
    theta_step,
    ado_small_angle_deg=ADO_SMALL_ANGLE_DEG,
    ado_large_angle_deg=ADO_LARGE_ANGLE_DEG,
    p_angle_deg=P_ANGLE_DEG,
):
    """Generate a list of P-ADO points at fixed sigma/I."""
    if theta_step <= 0.0:
        raise ValueError("THETA_STEP must be positive.")
    if theta_max < theta_min:
        raise ValueError("THETA_MAX must be greater than or equal to THETA_MIN.")

    theta_values = np.arange(
        theta_min,
        theta_max + theta_step / 2.0,
        theta_step,
        dtype=float,
    )
    theta_values = theta_values[theta_values <= theta_max + abs(theta_step) * 1e-9]

    sigma = sigma_from_sigma_over_i(ji, sigma_i)
    align2 = calculate_alignment(ji, 2, sigma)
    align4 = calculate_alignment(ji, 4, sigma)
    prepared = prepare_transition(
        ji,
        jf,
        ado_small_angle_deg=ado_small_angle_deg,
        ado_large_angle_deg=ado_large_angle_deg,
        p_angle_deg=p_angle_deg,
    )

    return [
        calculate_pado_point(theta, sigma_i, ji, jf, prepared, align2, align4)
        for theta in theta_values
    ]


# =========================
# Output helpers
# =========================


def save_points_csv(points, csv_output):
    """Save theta, delta, sigma/I, P, and R_ADO values to CSV."""
    output_path = Path(csv_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["theta_deg", "delta", "sigma/I", "p", "ado"],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "theta_deg": point["theta_deg"],
                    "delta": point["delta"],
                    "sigma/I": point["sigma/I"],
                    "p": point["p"],
                    "ado": point["ado"],
                }
            )


def plot_pado_points(
    points,
    sigma_i=SIGMA_I,
    theta_min=THETA_MIN,
    theta_max=THETA_MAX,
    ado_small_angle_deg=ADO_SMALL_ANGLE_DEG,
    ado_large_angle_deg=ADO_LARGE_ANGLE_DEG,
    p_angle_deg=P_ANGLE_DEG,
    marker_size=MARKER_SIZE,
    draw_line=DRAW_LINE,
    color_by_theta=COLOR_BY_THETA,
    show_colorbar=SHOW_COLORBAR,
):
    """Plot R_ADO on the x axis and P on the y axis."""
    rado_values = np.array([point["ado"] for point in points], dtype=float)
    p_values = np.array([point["p"] for point in points], dtype=float)
    theta_values = np.array([point["theta_deg"] for point in points], dtype=float)

    try:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    except Exception as exc:
        if "TclError" not in type(exc).__name__ and "tk" not in str(exc).lower():
            raise
        plt.switch_backend("Agg")
        print("Matplotlib GUI backend is unavailable; using Agg backend.")
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)

    if draw_line:
        ax.plot(rado_values, p_values, color="0.45", linewidth=0.8, alpha=0.75)

    scatter = None
    if color_by_theta:
        scatter = ax.scatter(
            rado_values,
            p_values,
            c=theta_values,
            s=marker_size,
            cmap="viridis",
            linewidths=0.0,
        )
    else:
        ax.scatter(
            rado_values,
            p_values,
            s=marker_size,
            color="tab:blue",
            linewidths=0.0,
        )

    if show_colorbar:
        if scatter is None:
            norm = plt.Normalize(float(theta_values.min()), float(theta_values.max()))
            scatter = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
            scatter.set_array(theta_values)
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("theta (deg)")

    ax.set_xlabel("R_ADO")
    ax.set_ylabel("P")
    ax.set_title(
        f"sigma/I = {sigma_i:g} | theta = {theta_min:g} to {theta_max:g} deg\n"
        f"RADO: {ado_small_angle_deg:g}/{ado_large_angle_deg:g} deg, "
        f"P: {p_angle_deg:g} deg"
    )
    ax.grid(True, color="0.88", linewidth=0.8)

    return fig, ax


def main():
    points = generate_pado_points(
        JI,
        JF,
        SIGMA_I,
        THETA_MIN,
        THETA_MAX,
        THETA_STEP,
        ado_small_angle_deg=ADO_SMALL_ANGLE_DEG,
        ado_large_angle_deg=ADO_LARGE_ANGLE_DEG,
        p_angle_deg=P_ANGLE_DEG,
    )
    fig, _ = plot_pado_points(
        points,
        sigma_i=SIGMA_I,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        ado_small_angle_deg=ADO_SMALL_ANGLE_DEG,
        ado_large_angle_deg=ADO_LARGE_ANGLE_DEG,
        p_angle_deg=P_ANGLE_DEG,
        marker_size=MARKER_SIZE,
        draw_line=DRAW_LINE,
        color_by_theta=COLOR_BY_THETA,
        show_colorbar=SHOW_COLORBAR,
    )

    if SAVE_FIGURE:
        figure_path = Path(OUTPUT_FIGURE)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=DPI)
        print(f"Saved figure: {figure_path}")

    if SAVE_CSV:
        save_points_csv(points, CSV_OUTPUT)
        print(f"Saved CSV: {CSV_OUTPUT}")

    p_values = np.array([point["p"] for point in points], dtype=float)
    rado_values = np.array([point["ado"] for point in points], dtype=float)
    finite = bool(np.all(np.isfinite(p_values)) and np.all(np.isfinite(rado_values)))
    print(f"Generated points: {len(points)}")
    print(f"Finite P and R_ADO values: {finite}")
    print(f"Colorbar shown: {SHOW_COLORBAR}")

    backend = plt.get_backend().lower()
    if SHOW_FIGURE and "agg" not in backend:
        plt.show()
    else:
        if SHOW_FIGURE:
            print("SHOW_FIGURE is True, but the current backend cannot open a window.")
        plt.close(fig)


if __name__ == "__main__":
    main()
