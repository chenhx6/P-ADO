"""
Probability-density primitives used by the observable-space fitting stage.

What this module computes
-------------------------
This module provides the actual 1D PDF formulas used for measured observables.
In the current workflow, the observables are:

1. P
2. ADO

For each observable, we build a 1D probability density in "observable space".
That means the PDF here is not yet a PDF in (delta, sigma/I). Instead, it is a
PDF of the measured value itself, such as:

- p_pdf(P)
- ado_pdf(ADO)

These 1D PDFs are later combined in `compute/jacobian.py` as:

    gaus2d = p_pdf(p0) * ado_pdf(ado0)

and then mapped from observable space to parameter space through the Jacobian.

Why there are three models
--------------------------
The experimental input gives a central value plus left/right errors. Depending
on whether those errors are symmetric or asymmetric, we choose one of three
models:

1. Gaussian:
   Used when left/right errors are effectively the same.

2. Skew-normal:
   Used when the uncertainty is asymmetric, but a smooth single-piece skewed
   distribution can still match the requested quantiles well.

3. Continuous split-normal:
   Used as a fallback when the uncertainty is asymmetric and skew-normal cannot
   reproduce the target quantiles well enough. It keeps continuity at the
   center and allows different left/right widths.

This file only defines the formulas themselves.
The model-selection logic lives in `pdf/fitters.py`.
"""

import numpy as np
from scipy.special import ndtr

from ..config import INV_SQRT_2PI


def _as_float_or_array(values):
    """
    Return a Python float for scalar inputs and a NumPy array for array inputs.

    This keeps the PDF helpers convenient for both:
    - scalar calls during ad-hoc checks
    - bulk array evaluation in the Jacobian hot path
    """

    if np.ndim(values) == 0:
        return float(values)
    return values


def gaussian_pdf(x, mu, sigma):
    """
    Evaluate a standard Gaussian PDF.

    Use case:
    - Left and right errors are symmetric or nearly symmetric.

    Parameters
    ----------
    x:
        Observable value at which the density is evaluated.
    mu:
        Central value of the observable.
    sigma:
        Standard deviation.

    Returns
    -------
    float or ndarray
        Probability density of the Gaussian model at x.
    """

    x_arr = np.asarray(x, dtype=float)
    z = (x_arr - mu) / sigma
    values = (INV_SQRT_2PI / sigma) * np.exp(-0.5 * z * z)
    return _as_float_or_array(values)


def skew_normal_pdf(x, xi, omega, alpha):
    """
    Evaluate a skew-normal PDF.

    Why this model is used
    ----------------------
    When the reported left/right uncertainties are not equal, the target
    distribution is asymmetric. A skew-normal is a natural first choice because:

    - it is smooth
    - it has one continuous closed-form density
    - it can represent both symmetric and skewed shapes

    Parameter meaning
    -----------------
    xi:
        Location parameter.
    omega:
        Scale parameter.
    alpha:
        Shape parameter controlling skewness.

    Implementation note
    -------------------
    This uses the direct skew-normal formula

        2 / omega * phi(z) * Phi(alpha * z)

    with:
        z = (x - xi) / omega

    where phi is the standard normal PDF and Phi is the standard normal CDF.
    Using `scipy.special.ndtr` keeps the evaluator naturally vectorized for
    NumPy arrays and avoids extra overhead from scalar-oriented stats wrappers.

    Returns
    -------
    float or ndarray
        Probability density of the skew-normal model at x.
    """

    x_arr = np.asarray(x, dtype=float)
    z = (x_arr - xi) / omega
    phi_z = INV_SQRT_2PI * np.exp(-0.5 * z * z)
    values = (2.0 / omega) * phi_z * ndtr(alpha * z)
    return _as_float_or_array(values)


def split_normal_continuous_pdf(x, mu, sigma_l, sigma_r):
    """
    Evaluate a continuous split-normal PDF.

    Why this fallback exists
    ------------------------
    Some asymmetric uncertainties are hard to match well with a skew-normal
    under the parameter bounds used by this project. In those cases we fall back
    to a split-normal-like model:

    - left side uses sigma_l
    - right side uses sigma_r
    - the density is continuous at x = mu

    This model is less "globally smooth" than skew-normal in the modeling
    sense, but it is robust and directly reflects asymmetric widths.

    Important property
    ------------------
    This implementation uses a normalization coefficient based on
    `sigma_l + sigma_r`, so the full piecewise density integrates to 1 and
    remains continuous at the center.

    Returns
    -------
    float or ndarray
        Probability density of the continuous split-normal model at x.
    """

    coeff = (2.0 * INV_SQRT_2PI) / (sigma_l + sigma_r)
    x_arr = np.asarray(x, dtype=float)
    values = np.where(
        x_arr < mu,
        coeff * np.exp(-0.5 * ((x_arr - mu) / sigma_l) ** 2),
        coeff * np.exp(-0.5 * ((x_arr - mu) / sigma_r) ** 2),
    )
    return _as_float_or_array(values)
