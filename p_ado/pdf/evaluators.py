"""
Turn fitted PDF model dictionaries into callable density evaluators.

How this file fits into the workflow
------------------------------------
The PDF pipeline across this project is:

1. `fit_pdf_model(...)` in `pdf/fitters.py`
   Input:
   - center value
   - left error
   - right error

   Output:
   - a model dictionary describing which PDF family to use and with what
     parameters

2. `build_pdf_evaluator(model)` in this file
   Input:
   - the model dictionary

   Output:
   - a callable function `pdf(x)` that evaluates the 1D density at x

3. Downstream usage in `compute/solver.py`
   - one evaluator is built for P
   - one evaluator is built for ADO

4. Downstream usage in `compute/jacobian.py`
   - at each grid point, the code computes the theoretical observable pair
     (P, ADO)
   - the 1D densities are evaluated there
   - their product forms the observable-space weight:

         p_pdf(P) * ado_pdf(ADO)

Important conceptual point
--------------------------
This module does not compute a PDF in model-parameter space directly.
It builds PDFs in measurement space:

- one PDF over P
- one PDF over ADO

Those observable-space PDFs are later transported to parameter space through the
Jacobian transformation.
"""

from .models import gaussian_pdf, skew_normal_pdf, split_normal_continuous_pdf


def build_pdf_evaluator(model):
    """
    Build a callable 1D PDF evaluator from a fitted model dictionary.

    Parameters
    ----------
    model:
        Dictionary returned by `fit_pdf_model`.

    Returns
    -------
    callable
        A function of one variable:

            pdf(x) -> density

    Supported model kinds
    ---------------------
    gaussian:
        Returns a Gaussian density evaluator.

    skew_normal:
        Returns a skew-normal density evaluator.

    split_normal:
        Returns a continuous split-normal density evaluator.

    Why return a lambda
    -------------------
    The later Jacobian code only needs a simple callable object. Returning a
    bound lambda keeps the downstream code clean:

        gaus2d = p_pdf(p0) * ado_pdf(ado0)

    where `p_pdf` and `ado_pdf` already remember their fitted parameters.

    Performance note
    ----------------
    These evaluators are designed to accept whole NumPy arrays, because the
    Jacobian stage evaluates PDF values over large interior grids. Bulk array
    evaluation avoids Python-level scalar loops in that hot path.
    """

    kind = model["kind"]
    if kind == "gaussian":
        mu = model["mu"]
        sigma = model["sigma"]
        return lambda x: gaussian_pdf(x, mu, sigma)
    if kind == "skew_normal":
        xi = model["xi"]
        omega = model["omega"]
        alpha = model["alpha"]
        return lambda x: skew_normal_pdf(x, xi, omega, alpha)
    if kind == "split_normal":
        mu = model["mu"]
        sigma_l = model["sigma_l"]
        sigma_r = model["sigma_r"]
        return lambda x: split_normal_continuous_pdf(x, mu, sigma_l, sigma_r)

    raise ValueError(f"Unknown PDF model kind: {kind}")
