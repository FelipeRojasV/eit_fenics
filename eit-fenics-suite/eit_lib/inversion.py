"""
Algoritmos de reconstrucción: Tikhonov/Gauss-Newton y TSVD.
"""

import numpy as np

# Intentar importar petsc4py; si no está disponible, LUSolver no podrá usarse
try:
    from petsc4py import PETSc
    from dolfin import PETScMatrix, PETScVector, LUSolver
    _HAS_PETSC = True
except ImportError:
    _HAS_PETSC = False


def extract_measurement_vector(F_matrix, pairs, symmetrize=True):
    b = np.zeros(len(pairs))
    for idx, (j, k) in enumerate(pairs):
        if symmetrize:
            b[idx] = 0.5 * (F_matrix[j - 1, k - 1] + F_matrix[k - 1, j - 1])
        else:
            b[idx] = F_matrix[j - 1, k - 1]
    return b


def solve_least_squares(A, delta_b, alpha=1e-4):
    """
    Resuelve (A^T A + alpha I) x = A^T delta_b.

    Si petsc4py está disponible: construye una matriz densa de PETSc,
    la envuelve como PETScMatrix y usa LUSolver para resolver.
    Si no: recurre a np.linalg.solve.
    """
    AtA = A.T @ A + alpha * np.eye(A.shape[1])
    Atb = A.T @ delta_b
    n = AtA.shape[0]

    if not _HAS_PETSC:
        return np.linalg.solve(AtA, Atb)

    try:
        # 1. Crear matriz densa de PETSc con petsc4py
        pet_mat = PETSc.Mat()
        pet_mat.createDense((n, n), array=AtA.astype(np.float64))
        pet_mat.setUp()

        # 2. Envolver como PETScMatrix de DOLFIN
        M = PETScMatrix(pet_mat)

        # 3. Crear vector b (también usando petsc4py para asegurar compatibilidad)
        pet_b = PETSc.Vec().createWithArray(Atb.astype(np.float64))
        b = PETScVector(pet_b)

        # 4. Crear vector solución
        x = PETScVector()
        x.resize(n)

        # 5. Resolver con LUSolver
        LUSolver().solve(M, x, b)

        return np.array([x.getitem(i) for i in range(n)])
    except Exception:
        # Si petsc4py existe pero la construcción de matriz densa falla
        # (por ejemplo, el backend no es PETSc), recurrir a NumPy
        return np.linalg.solve(AtA, Atb)


def solve_svd_linearized(A, delta_b, n_singular_values=4):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = min(n_singular_values, len(S))
    delta_sigma = np.zeros(A.shape[1])
    for i in range(r):
        if S[i] > 1e-12:
            delta_sigma += ((U[:, i].dot(delta_b)) / S[i]) * Vt[i, :]
    return delta_sigma, S, U, Vt
