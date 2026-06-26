import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.special import eval_legendre
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_6j

from ..config import L1, L2, ADO_THETA_L_DEG, ADO_THETA_S_DEG, P_THETA_DEG
from .sigmaOverI import align_par


# Convert a float into a stable Fraction representation for half-integer spins.
def _to_fraction(x: float) -> Fraction:
    return Fraction(str(float(x))).limit_denominator()


# Convert a numeric value into a SymPy Rational for exact angular-momentum algebra.
def _to_sympy_rational(x: float):
    frac = _to_fraction(x)
    return Rational(frac.numerator, frac.denominator)


# Compute the Racah coefficient used in the angular-correlation terms.
@lru_cache(maxsize=None)
def racah_coeff(
    j1: float, j2: float, l2: float, l1: float, j3: float, l3: float
) -> float:
    jj1, jj2, ll2, ll1, jj3, ll3 = map(_to_sympy_rational, [j1, j2, l2, l1, j3, l3])
    phase = (-1.0) ** float(jj1 + jj2 + ll1 + ll2)
    return float(phase * wigner_6j(jj1, jj2, jj3, ll1, ll2, ll3).evalf())


# Build the F_k coefficient for a given transition and multipole combination.
@lru_cache(maxsize=None)
def f_coeff(Jf: float, L1_: int, L2_: int, Ji: float, k: int) -> float:
    jjf, ll1, ll2, jji, kk = map(_to_sympy_rational, [Jf, L1_, L2_, Ji, k])
    phase = (-1.0) ** float(jjf - jji - 1)
    pref = math.sqrt(
        (2.0 * float(ll1) + 1.0) * (2.0 * float(ll2) + 1.0) * (2.0 * float(jji) + 1.0)
    )
    cg = float(
        clebsch_gordan(
            ll1,
            ll2,
            kk,
            Rational(1, 1),
            Rational(-1, 1),
            Rational(0, 1),
        ).evalf()
    )
    rac = racah_coeff(
        float(jji),
        float(jji),
        float(ll1),
        float(ll2),
        float(kk),
        float(jjf),
    )
    return float(phase * pref * cg * rac)


# Precompute angle- and transition-dependent constants reused across the grid.
def prepare_transition(ji: float, jf: float):
    x_s = math.cos(math.radians(ADO_THETA_S_DEG))
    x_l = math.cos(math.radians(ADO_THETA_L_DEG))
    x_p = math.cos(math.radians(P_THETA_DEG))

    return {
        "legS2": float(eval_legendre(2, x_s)),
        "legS4": float(eval_legendre(4, x_s)),
        "legL2": float(eval_legendre(2, x_l)),
        "legL4": float(eval_legendre(4, x_l)),
        "legP2": float(eval_legendre(2, x_p)),
        "legP4": float(eval_legendre(4, x_p)),
        "assocP22": float(3.0 * (1.0 - x_p * x_p)),
        "assocP42": float(7.5 * (1.0 - x_p * x_p) * (7.0 * x_p * x_p - 1.0)),
        "f11k2": f_coeff(jf, L1, L1, ji, 2),
        "f12k2": f_coeff(jf, L1, L2, ji, 2),
        "f22k2": f_coeff(jf, L2, L2, ji, 2),
        "f11k4": f_coeff(jf, L1, L1, ji, 4),
        "f12k4": f_coeff(jf, L1, L2, ji, 4),
        "f22k4": f_coeff(jf, L2, L2, ji, 4),
    }


# Compute the P value and ADO ratio for one (delta, alignment) point.
def fast_pr(delta, align2, align4, pre: dict):
    delta_arr = np.asarray(delta, dtype=float)
    align2_arr = np.asarray(align2, dtype=float)
    align4_arr = np.asarray(align4, dtype=float)

    d2 = delta_arr * delta_arr
    denom = 1.0 + d2

    leg2 = align2_arr * (
        (pre["f11k2"] + 2.0 * delta_arr * pre["f12k2"] + d2 * pre["f22k2"]) / denom
    )
    leg4 = align4_arr * (
        (pre["f11k4"] + 2.0 * delta_arr * pre["f12k4"] + d2 * pre["f22k4"]) / denom
    )

    assoc22 = align2_arr * (
        (
            0.5 * pre["f11k2"]
            - (1.0 / 3.0) * delta_arr * pre["f12k2"]
            + 0.5 * d2 * pre["f22k2"]
        )
        / denom
    )
    assoc42 = align4_arr * (((-1.0 / 12.0) * d2 * pre["f22k4"]) / denom)

    p = (assoc22 * pre["assocP22"] + assoc42 * pre["assocP42"]) / (
        1.0 + leg2 * pre["legP2"] + leg4 * pre["legP4"]
    )
    wS = 1.0 + leg2 * pre["legS2"] + leg4 * pre["legS4"]
    wL = 1.0 + leg2 * pre["legL2"] + leg4 * pre["legL4"]
    ado = wS / wL

    if p.ndim == 0 and ado.ndim == 0:
        return float(p), float(ado)
    return p, ado


# Build the full observable grid: delta, theta, sigma/I, P, and ADO.
def build_pado_points(
    ji: float,
    jf: float,
    delta_list,
    theta_deg_list,
    sigma_list,
    sigma_i_list,
):
    pre = prepare_transition(ji, jf)
    sigma_arr = np.asarray(sigma_list, dtype=float)
    delta_arr = np.asarray(delta_list, dtype=float)
    theta_arr = np.asarray(theta_deg_list, dtype=float)
    sigma_i_arr = np.asarray(sigma_i_list, dtype=float)

    align2_list = np.fromiter(
        (align_par(ji, 2, sigma) for sigma in sigma_arr),
        dtype=float,
    )
    align4_list = np.fromiter(
        (align_par(ji, 4, sigma) for sigma in sigma_arr),
        dtype=float,
    )

    # Benchmark note:
    # the old implementation called fast_pr once per (delta, sigma/I) point in
    # nested Python loops. The vectorized path below pushes the full grid
    # arithmetic into NumPy broadcasting, which removes most per-point Python
    # overhead and is noticeably faster on dense production grids.
    delta_grid = delta_arr[:, None]
    theta_grid = theta_arr[:, None]
    sigma_i_grid = sigma_i_arr[None, :]
    align2_grid = align2_list[None, :]
    align4_grid = align4_list[None, :]

    p_grid, ado_grid = fast_pr(delta_grid, align2_grid, align4_grid, pre)

    n_delta = delta_arr.size
    n_sigma = sigma_arr.size
    points = np.empty((n_delta, n_sigma, 5), dtype=float)
    points[:, :, 0] = delta_grid
    points[:, :, 1] = theta_grid
    points[:, :, 2] = sigma_i_grid
    points[:, :, 3] = p_grid
    points[:, :, 4] = ado_grid
    return points
