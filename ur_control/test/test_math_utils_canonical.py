"""Canonical rotation conversion tests for ur_control.math_utils."""

import numpy as np
import quaternion

from ur_control.math_utils import (
    axis_angle_from_quaternion,
    matrix_to_axis_angle,
    quaternion_from_axis_angle,
    quaternion_from_matrix,
    relative_axis_angle,
    rotation_matrix_from_quaternion,
)


def _random_rotation_matrix(rng: np.random.Generator, max_angle: float = np.pi) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, max_angle)
    return quaternion.as_rotation_matrix(quaternion.from_rotation_vector(axis * angle))


def test_quaternion_from_matrix_has_nonnegative_w():
    rng = np.random.default_rng(0)
    for _ in range(100):
        R = _random_rotation_matrix(rng)
        q = quaternion_from_matrix(R)
        assert q[3] >= 0.0


def test_axis_angle_from_quaternion_is_shortest_arc():
    rng = np.random.default_rng(1)
    theta = 0.03
    mags = []
    for _ in range(100):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(axis * theta))
        q = quaternion_from_matrix(R)
        aa = axis_angle_from_quaternion(q)
        mags.append(np.linalg.norm(aa))
    assert max(mags) <= np.pi + 1e-10
    assert np.allclose(mags, theta, atol=1e-6)


def test_matrix_to_axis_angle_matches_shortest_arc():
    rng = np.random.default_rng(2)
    theta = 0.025
    for _ in range(50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R_offset = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(axis * theta))
        for _ in range(5):
            R_base = _random_rotation_matrix(rng, max_angle=2 * np.pi)
            R = R_base @ R_offset @ R_base.T
            aa = matrix_to_axis_angle(R)
            assert np.linalg.norm(aa) <= np.pi + 1e-10
            assert abs(np.linalg.norm(aa) - theta) < 1e-6


def test_quaternion_from_axis_angle_wraps_aliased_input():
    aa_good = np.array([0.0, 0.0, 0.03])
    aa_bad = np.array([0.0, 0.0, -(2.0 * np.pi - 0.03)])
    R_good = rotation_matrix_from_quaternion(quaternion_from_axis_angle(aa_good))[:3, :3]
    R_bad = rotation_matrix_from_quaternion(quaternion_from_axis_angle(aa_bad))[:3, :3]
    np.testing.assert_allclose(R_good, R_bad, atol=1e-10)


def test_relative_axis_angle_shortest_arc():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = quaternion_from_axis_angle(np.array([0.0, 0.0, 0.2]))
    err = relative_axis_angle(q1, q0)
    assert abs(np.linalg.norm(err) - 0.2) < 1e-6
