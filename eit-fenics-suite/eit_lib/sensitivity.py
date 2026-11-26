"""
Ensamblaje de la matriz de sensibilidad Jacobiana A.
A_{(j,k), m} = - (1/I) * int_{Omega_m} grad(u_j) . grad(u_k) dx
"""

from ._fenics_check import require_fenics
require_fenics()

from fenics import dot, grad, assemble
import numpy as np


class EITSensitivity:
    def __init__(self, geometry, current_intensity=1.0):
        self.geom = geometry
        self.I = current_intensity
        self.pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    def build_jacobian(self, potentials, num_params=4):
        A = np.zeros((len(self.pairs), num_params))
        for row_idx, (j, k) in enumerate(self.pairs):
            u_j = potentials[j - 1]
            u_k = potentials[k - 1]
            integrand = dot(grad(u_j), grad(u_k))
            for m in range(1, num_params + 1):
                val = assemble(integrand * self.geom.dx(m))
                A[row_idx, m - 1] = -(1.0 / self.I) * val
        return A
