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


"""
Algoritmos de reconstrucción: Tikhonov/Gauss-Newton y TSVD.

solve_least_squares usa exclusivamente LUSolver de DOLFIN con el
factorizador 'umfpack'. La matriz (A^T A + alpha I) se construye como
AIJ (dispersa llena) con petsc4py, porque PETSc no tiene factorizador
LU para MATDENSE en esta compilación.

No hay fallback a numpy.linalg.solve: si petsc4py no está disponible
o LUSolver falla, se lanza una excepción con diagnóstico.
"""

import numpy as np

try:
    from petsc4py import PETSc
    from dolfin import PETScMatrix, PETScVector, LUSolver
except ImportError as exc:
    raise ImportError(
        "\n"
        "==============================================================\n"
        " solve_least_squares requiere petsc4py y DOLFIN con backend PETSc.\n"
        " Esta suite no incluye fallback a numpy.linalg.solve.\n"
        "\n"
        " Verifica que tu entorno FEniCS Legacy tenga petsc4py instalado:\n"
        "   python -c \"from petsc4py import PETSc; print('ok')\"\n"
        "==============================================================\n"
    ) from exc


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
    """
    Resuelve (A^T A + alpha I) x = A^T delta_b usando LUSolver('umfpack').

    La matriz se construye como AIJ (dispersa llena) con petsc4py, porque
    PETSc no tiene factorizador LU para MATDENSE en esta compilación
    (error 92 = MISSING_FACTOR). Se usa 'umfpack' explícitamente porque
    es el factorizador LU más adecuado para matrices pequeñas y llenas.

    Si LUSolver falla, lanza RuntimeError con diagnóstico.
    """
    
    """
    Se añade comentada una alternativa a solve_least_squares mejor en cada ámbito pero no de fenics
    Resuelve el problema de mínimos cuadrados regularizado (Tikhonov):
    delta_sigma = argmin ||A * x - delta_b||^2 + alpha * ||x||^2
    """

    """
    def solve_least_squares(A, delta_b, alpha=1e-4):
    
    AtA = A.T @ A
    Atb = A.T @ delta_b
    return np.linalg.solve(AtA + alpha * np.eye(A.shape[1]), Atb)
    """
    
    AtA = A.T @ A + alpha * np.eye(A.shape[1])
    Atb = A.T @ delta_b
    n = AtA.shape[0]

    # 1. Matriz AIJ llena con petsc4py
    pet_mat = PETSc.Mat()
    pet_mat.createAIJ((n, n), nnz=n * n)
    pet_mat.setUp()
    for i in range(n):
        for j in range(n):
            pet_mat.setValue(i, j, float(AtA[i, j]))
    pet_mat.assemblyBegin()
    pet_mat.assemblyEnd()

    # 2. Envolver como PETScMatrix de DOLFIN
    M = PETScMatrix(pet_mat)

    # 3. Vector b como PETScVector
    pet_b = PETSc.Vec().createWithArray(
        np.ascontiguousarray(Atb, dtype=np.float64)
    )
    b = PETScVector(pet_b)

    # 4. Vector solución
    pet_x = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    pet_x.setSizes(n)
    pet_x.setUp()
    x = PETScVector(pet_x)

    # 5. Resolver con LUSolver('umfpack')
    try:
        LUSolver("umfpack").solve(M, x, b)
    except RuntimeError as exc:
        raise RuntimeError(
            "\n"
            "==============================================================\n"
            " LUSolver('umfpack') falló al resolver el sistema 4x4.\n"
            f" Error original: {exc}\n"
            "\n"
            " Factorizadores disponibles en este PETSc:\n"
            "   default, mumps, petsc, superlu, superlu_dist, umfpack\n"
            "\n"
            " Prueba LUSolver('mumps') o LUSolver('superlu') como alternativa.\n"
            "==============================================================\n"
        ) from exc

    return np.array(x.get_local())


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
            delta_sigma += ((U[:, i].dot(delta_b)) / S[i]) * Vt[i, :]
    return delta_sigma, S, U, Vt


def solve_svd_linearized(A, delta_b, n_singular_values=4):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    r = min(n_singular_values, len(S))
    delta_sigma = np.zeros(A.shape[1])
    for i in range(r):
        if S[i] > 1e-12:
            delta_sigma += ((U[:, i].dot(delta_b)) / S[i]) * Vt[i, :]
    return delta_sigma, S, U, Vt
