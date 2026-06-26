import numpy as np

from ..physics.delta import theta_deg_to_delta
from ..physics.sigmaOverI import sigma_from_sigma_over_i


def build_delta_grid(grid_cfg):
    theta_min, theta_max, theta_step = grid_cfg.theta_range()

    theta_deg = np.arange(
        theta_min,
        theta_max + theta_step / 2.0,
        theta_step,
        dtype=float,
    )

    tol = np.abs(theta_step) * 1e-9
    theta_deg = theta_deg[theta_deg <= theta_max + tol]

    delta = theta_deg_to_delta(theta_deg)
    return theta_deg, delta


def build_sigma_grid(ji, grid_cfg):
    sigma_i_min, sigma_i_max, sigma_i_step = grid_cfg.sigma_i_range()
    sigma_min = sigma_from_sigma_over_i(ji, sigma_i_min)
    sigma_max = sigma_from_sigma_over_i(ji, sigma_i_max)
    step_sigma = sigma_from_sigma_over_i(ji, sigma_i_step)

    sigma = np.arange(
        sigma_min,
        sigma_max + step_sigma / 2.0,
        step_sigma,
        dtype=float,
    )
    tol = np.abs(step_sigma) * 1e-9
    sigma = sigma[sigma <= sigma_max + tol]

    sigma_i = sigma / ji
    return sigma, sigma_i
