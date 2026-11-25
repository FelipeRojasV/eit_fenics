"""
eit_lib/inversion.py
Algoritmos de reconstrucción de conductividad:
- Mínimos Cuadrados regularizados (Tikhonov / Gauss-Newton)
- Descomposición en Valores Singulares Truncada (TSVD)

Nota: la construcción de una Function DG(0) a partir de valores por
cuadrante vive en eit_lib/representation.py, no aquí.
"""

import numpy as np


def extract_measurement_vector(F_matrix, pairs, symmetrize=True):
    """
    Extrae el vector de mediciones independientes a partir de la matriz NtD.

    Parámetros
    ----------
    F_matrix : np.ndarray, shape (3, 3)
        Matriz NtD completa.
    pairs : list[tuple[int, int]]
        Pares (j, k) de mediciones independientes.
    symmetrize : bool, default=True
        Si True, promedia F[j,k] y F[k,j] (0.5*(F + F.T)), lo cual
        enmascara cualquier asimetría numérica. Si False, toma F[j-1, k-1]
        directamente, lo que permite verificar la reciprocidad cruda
        (usado en ex2 y en los tests).
    """
    b = np.zeros(len(pairs))
    for idx, (j, k) in enumerate(pairs):
        if symmetrize:
            b[idx] = 0.5 * (F_matrix[j - 1, k - 1] + F_matrix[k - 1, j - 1])
        else:
            b[idx] = F_matrix[j - 1, k - 1]
    return b


def solve_least_squares(A, delta_b, alpha=1e-4):
    """
    Resuelve el problema de mínimos cuadrados regularizado (Tikhonov):
    delta_sigma = argmin ||A * x - delta_b||^2 + alpha * ||x||^2
    """
    AtA = A.T @ A
    Atb = A.T @ delta_b
    delta_sigma = np.linalg.solve(AtA + alpha * np.eye(A.shape[1]), Atb)
    return delta_sigma


def solve_svd_linearized(A, delta_b, n_singular_values=4):
    """
    Resuelve mediante Truncated SVD (TSVD):
    A = U * S * V^T, usando r valores singulares principales.
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = min(n_singular_values, len(S))

    delta_sigma = np.zeros(A.shape[1])
    for i in range(r):
        if S[i] > 1e-12:
            coeff = (U[:, i].dot(delta_b)) / S[i]
            delta_sigma += coeff * Vt[i, :]

    return delta_sigma, S, U, Vt