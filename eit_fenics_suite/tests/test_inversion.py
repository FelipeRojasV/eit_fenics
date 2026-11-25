import numpy as np

from eit_lib import (
    extract_measurement_vector,
    solve_least_squares,
    solve_svd_linearized,
)


PAIRS = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
A_DEMO = np.array([
    [-1.25, -0.15, -0.05, -0.35],
    [-0.45, -0.40, -0.08, -0.12],
    [-0.30, -0.08, -0.08, -0.28],
    [-0.15, -1.35, -0.32, -0.06],
    [-0.08, -0.42, -0.35, -0.10],
    [-0.05, -0.10, -1.40, -0.25],
])


def test_extract_symmetrize_flag():
    F = np.array([
        [-2.00, -1.30, -1.10],
        [-1.31, -2.30, -1.40],
        [-1.10, -1.40, -2.50],
    ])
    b_sym = extract_measurement_vector(F, PAIRS, symmetrize=True)
    b_raw = extract_measurement_vector(F, PAIRS, symmetrize=False)

    # Diagonal: igual en ambos modos
    assert b_sym[0] == b_raw[0]
    assert b_sym[3] == b_raw[3]
    assert b_sym[5] == b_raw[5]

    # Off-diagonal: sym promedia, raw toma F[j-1, k-1]
    assert b_sym[1] != b_raw[1]
    assert b_raw[1] == F[0, 1]
    assert b_sym[1] == 0.5 * (F[0, 1] + F[1, 0])


def test_least_squares_recovers_noiseless():
    x_true = np.array([0.7, -0.3, 0.1, 0.4])
    b = A_DEMO @ x_true
    x_rec = solve_least_squares(A_DEMO, b, alpha=0.0)
    assert np.allclose(x_rec, x_true, atol=1e-10)


def test_svd_returns_correct_shapes():
    delta_b = A_DEMO @ np.array([0.7, -0.3, 0.1, 0.4])
    x_rec, S, U, Vt = solve_svd_linearized(A_DEMO, delta_b, n_singular_values=4)
    assert x_rec.shape == (4,)
    assert S.shape == (4,)
    assert U.shape == (6, 4)
    assert Vt.shape == (4, 4)