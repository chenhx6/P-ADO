import numpy as np

from p_ado.physics.delta import theta_deg_to_delta, delta_to_theta_deg
from p_ado.physics import pado as pado_module
from p_ado.physics.pado import build_pado_points, fast_pr, prepare_transition
from p_ado.physics.sigmaOverI import align_par


def test_delta_theta_round_trip():
    theta = 20.0
    delta = theta_deg_to_delta(theta)
    theta2 = delta_to_theta_deg(delta)
    assert abs(theta - theta2) < 1e-12


def _old_hard_coded_p(delta, align2, align4, pre):
    d2 = delta * delta
    denom = 1.0 + d2

    leg2 = align2 * (
        (pre["f11k2"] + 2.0 * delta * pre["f12k2"] + d2 * pre["f22k2"]) / denom
    )
    leg4 = align4 * (
        (pre["f11k4"] + 2.0 * delta * pre["f12k4"] + d2 * pre["f22k4"]) / denom
    )
    assoc22 = align2 * (
        (
            0.5 * pre["f11k2"]
            - (1.0 / 3.0) * delta * pre["f12k2"]
            + 0.5 * d2 * pre["f22k2"]
        )
        / denom
    )
    assoc42 = align4 * (((-1.0 / 12.0) * d2 * pre["f22k4"]) / denom)

    return (3.0 * assoc22 - 7.5 * assoc42) / (1.0 - 0.5 * leg2 + 0.375 * leg4)


def test_prepare_transition_includes_default_p_angle_factors():
    pre = prepare_transition(10.5, 9.5)

    for key in ("legP2", "legP4", "assocP22", "assocP42"):
        assert key in pre

    assert np.isclose(pre["legP2"], -0.5)
    assert np.isclose(pre["legP4"], 0.375)
    assert np.isclose(pre["assocP22"], 3.0)
    assert np.isclose(pre["assocP42"], -7.5)


def test_fast_pr_default_p_angle_matches_old_hard_coded_expression():
    pre = prepare_transition(10.5, 9.5)
    delta = 0.25
    align2 = 0.3
    align4 = 0.07

    p_new, _ = fast_pr(delta, align2, align4, pre)
    p_old = _old_hard_coded_p(delta, align2, align4, pre)

    assert np.isclose(p_new, p_old)


def test_prepare_transition_uses_configurable_p_angle(monkeypatch):
    monkeypatch.setattr(pado_module, "P_THETA_DEG", 45.0)

    pre = prepare_transition(10.5, 9.5)

    assert not np.isclose(pre["legP2"], -0.5)
    assert not np.isclose(pre["legP4"], 0.375)
    assert not np.isclose(pre["assocP22"], 3.0)
    assert not np.isclose(pre["assocP42"], -7.5)


def test_fast_pr_vectorized_matches_scalar_calls():
    pre = prepare_transition(10.5, 9.5)
    delta = np.array([-0.3, 0.1, 0.5], dtype=float)[:, None]
    align2 = np.array([0.2, 0.4], dtype=float)[None, :]
    align4 = np.array([0.05, 0.08], dtype=float)[None, :]

    p_grid, ado_grid = fast_pr(delta, align2, align4, pre)

    expected_p = np.empty((delta.shape[0], align2.shape[1]), dtype=float)
    expected_ado = np.empty_like(expected_p)
    for i, delta_value in enumerate(delta[:, 0]):
        for j, (a2, a4) in enumerate(zip(align2[0], align4[0])):
            p_scalar, ado_scalar = fast_pr(
                float(delta_value),
                float(a2),
                float(a4),
                pre,
            )
            expected_p[i, j] = p_scalar
            expected_ado[i, j] = ado_scalar

    assert np.allclose(p_grid, expected_p)
    assert np.allclose(ado_grid, expected_ado)


def test_build_pado_points_vectorized_matches_pointwise_reference():
    ji = 10.5
    jf = 9.5
    delta_list = np.array([-0.5, 0.0, 0.5], dtype=float)
    theta_deg_list = np.array([-26.56505118, 0.0, 26.56505118], dtype=float)
    sigma_list = np.array([1.05, 2.10, 3.15], dtype=float)
    sigma_i_list = sigma_list / ji

    points = build_pado_points(
        ji,
        jf,
        delta_list,
        theta_deg_list,
        sigma_list,
        sigma_i_list,
    )

    pre = prepare_transition(ji, jf)
    align2_list = np.array(
        [align_par(ji, 2, sigma) for sigma in sigma_list],
        dtype=float,
    )
    align4_list = np.array(
        [align_par(ji, 4, sigma) for sigma in sigma_list],
        dtype=float,
    )

    expected = np.empty((delta_list.size, sigma_list.size, 5), dtype=float)
    for i, (delta_value, theta_value) in enumerate(zip(delta_list, theta_deg_list)):
        for j, sigma_i_value in enumerate(sigma_i_list):
            p_scalar, ado_scalar = fast_pr(
                float(delta_value),
                float(align2_list[j]),
                float(align4_list[j]),
                pre,
            )
            expected[i, j, 0] = delta_value
            expected[i, j, 1] = theta_value
            expected[i, j, 2] = sigma_i_value
            expected[i, j, 3] = p_scalar
            expected[i, j, 4] = ado_scalar

    assert points.shape == (delta_list.size, sigma_list.size, 5)
    assert np.allclose(points, expected)
