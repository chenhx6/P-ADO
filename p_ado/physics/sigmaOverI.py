import math
from fractions import Fraction
from functools import lru_cache

from sympy import Rational
from sympy.physics.wigner import clebsch_gordan

from ..config import UNDERFLOW_CUTOFF


# Convert a float into a stable Fraction representation for half-integer spins.
def _to_fraction(x: float) -> Fraction:
    return Fraction(str(float(x))).limit_denominator()


# Convert a numeric value into a SymPy Rational for exact angular-momentum algebra.
def _to_sympy_rational(x: float):
    frac = _to_fraction(x)
    return Rational(frac.numerator, frac.denominator)


# Unnormalized population distribution over magnetic substates for a given sigma.
def pop_dist(m: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    arg = -(float(m) ** 2) / (2.0 * (float(sigma) ** 2))
    if arg <= UNDERFLOW_CUTOFF:
        return 0.0
    return math.exp(arg)


# Normalization factor for the substate population distribution at spin J.
@lru_cache(maxsize=None)
def pop_dist_norm_coeff(J: float, sigma: float) -> float:
    fracJ = _to_fraction(J)
    if fracJ.denominator == 1:
        j_int = fracJ.numerator
        return (
            2.0 * sum(pop_dist(i, sigma) for i in range(1, j_int + 1))
            + pop_dist(0.0, sigma)
        )

    # half-integer J
    values = []
    m = Fraction(1, 2)
    while m <= fracJ:
        values.append(float(m))
        m += 1

    return 2.0 * sum(pop_dist(v, sigma) for v in values)


# Cached Clebsch-Gordan coefficient evaluated as a float.
@lru_cache(maxsize=None)
def clebsch_gordan_float(
    j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
) -> float:
    return float(
        clebsch_gordan(
            _to_sympy_rational(j1),
            _to_sympy_rational(j2),
            _to_sympy_rational(j3),
            _to_sympy_rational(m1),
            _to_sympy_rational(m2),
            _to_sympy_rational(m3),
        ).evalf()
    )


# Calculate the alignment parameter A_k for spin J and width sigma.
@lru_cache(maxsize=None)
def align_par(J: float, k: int, sigma: float) -> float:
    JJ = _to_fraction(J)
    pop_norm = pop_dist_norm_coeff(float(JJ), float(sigma))
    if pop_norm == 0.0:
        return 0.0

    sqrt2J = math.sqrt(2.0 * float(JJ) + 1.0)
    coeff = sqrt2J / pop_norm

    def align_term(m: Fraction) -> float:
        phase = (-1.0) ** float(JJ - m)
        cg = clebsch_gordan_float(
            float(JJ), float(JJ), float(k), float(m), float(-m), 0.0
        )
        return phase * cg * pop_dist(float(m), float(sigma))

    m_start = Fraction(1, 1) if JJ.denominator == 1 else Fraction(1, 2)
    m = m_start
    sum_part = 0.0
    while m <= JJ:
        sum_part += align_term(m)
        m += 1

    if JJ.denominator == 1:
        sum_part += 0.5 * align_term(Fraction(0, 1))

    return float(2.0 * coeff * sum_part)


# Convert the dimensionless sigma/I value into sigma using Ji.
def sigma_from_sigma_over_i(ji: float, sigma_i: float) -> float:
    return float(ji) * float(sigma_i)
