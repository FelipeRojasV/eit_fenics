"""
Módulo de Utilidades y Verificaciones de Principios Físicos.
"""

import numpy as np


def verify_reciprocity_matrix(F_matrix, tol=1e-2):
    """
    Verifica si la matriz de transferencia Neumann-to-Dirichlet cumple
    con la simetría exigida por el teorema de reciprocidad de Lorentz:
        |F - F^T| < tol
    """
    diff = np.abs(F_matrix - F_matrix.T)
    max_diff = float(np.max(diff))
    is_symmetric = max_diff < tol
    return is_symmetric, max_diff
