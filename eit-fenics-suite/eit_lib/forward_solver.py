"""
Resolvedor del problema directo para EIT en FEniCS Legacy.
"""

from ._fenics_check import require_fenics
require_fenics()

from fenics import (
    FunctionSpace, DirichletBC, Constant, TrialFunction, TestFunction,
    dot, grad, dx, solve, Function, assemble,
)
import numpy as np


class EITForwardSolver:
    def __init__(self, geometry, current_intensity=1.0):
        self.geom = geometry
        self.I = current_intensity
        self.V = FunctionSpace(self.geom.mesh, "Lagrange", 1)
        self.bc = DirichletBC(self.V, Constant(0.0), self.geom.boundaries, 10)

    def solve(self, sigma_func, injection_electrode_idx):
        tag_ej = self.geom.electrode_ids[injection_electrode_idx - 1]
        area_ej = self.geom.area_Ek[injection_electrode_idx - 1]
        if area_ej <= 0.0:
            raise ValueError(
                f"Electrodo E{injection_electrode_idx} tiene área 0 en esta malla. "
                f"Aumenta nx, ny o electrode_width."
            )

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        g_j = Constant(self.I / area_ej)

        a = sigma_func * dot(grad(u), grad(v)) * dx
        L = g_j * v * self.geom.ds(tag_ej)

        u_sol = Function(self.V)
        solve(a == L, u_sol, self.bc)
        return u_sol

    def measure_voltages(self, u_sol):
        voltages = np.zeros(3)
        for k_idx, tag in enumerate(self.geom.electrode_ids):
            area_k = self.geom.area_Ek[k_idx]
            v_val = assemble(u_sol * self.geom.ds(tag)) / area_k
            voltages[k_idx] = v_val
        return voltages

    def compute_ntd_matrix(self, sigma_func):
        F = np.zeros((3, 3))
        potentials = []
        for j in range(1, 4):
            u_j = self.solve(sigma_func, j)
            potentials.append(u_j)
            F[j - 1, :] = self.measure_voltages(u_j)
        return F, potentials
