"""
Algoritmos de reconstrucción: Tikhonov/Gauss-Newton y TSVD.
"""

import numpy as np


def extract_measurement_vector(F_matrix, pairs, symmetrize=True):
    """
    Extrae el vector de mediciones independientes.
    symmetrize=True promedia F[j,k] y F[k,j].
    symmetrize=False toma F[j-1, k-1] directamente.
    """
    b = np.zeros(len(pairs))
    for idx, (j, k) in enumerate(pairs):
        if symmetrize:
            b[idx] = 0.5 * (F_matrix[j - 1, k - 1] + F_matrix[k - 1, j - 1])
        else:
            b[idx] = F_matrix[j - 1, k - 1]
    return b


def solve_least_squares(A, delta_b, alpha=1e-4):
    AtA = A.T @ A
    Atb = A.T @ delta_b
    return np.linalg.solve(AtA + alpha * np.eye(A.shape[1]), Atb)


def solve_svd_linearized(A, delta_b, n_singular_values=4):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = min(n_singular_values, len(S))
    delta_sigma = np.zeros(A.shape[1])
    for i in range(r):
        if S[i] > 1e-12:
            delta_sigma += ((U[:, i].dot(delta_b)) / S[i]) * Vt[i, :]
    return delta_sigma, S, U, Vt
