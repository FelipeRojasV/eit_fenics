"""
eit_lib/forward_solver.py
Resolvedor de Problema Directo para EIT en FEniCS Legacy.
Resuelve -div(sigma grad(u)) = 0 sujeto a condiciones de electrodo mixto.
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
        # Electrodo de referencia E0 puesto a tierra (0V)
        self.bc = DirichletBC(self.V, Constant(0.0), self.geom.boundaries, 10)

    def solve(self, sigma_func, injection_electrode_idx):
        """
        Resuelve P_j[sigma]: corriente inyectada por E_j (j in {1, 2, 3})
        y saliendo por E_0 (referencia).
        """
        tag_ej = self.geom.electrode_ids[injection_electrode_idx - 1]
        area_ej = self.geom.area_Ek[injection_electrode_idx - 1]

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        g_j = Constant(self.I / area_ej)

        a = sigma_func * dot(grad(u), grad(v)) * dx
        L = g_j * v * self.geom.ds(tag_ej)

        u_sol = Function(self.V)
        solve(a == L, u_sol, self.bc)
        return u_sol

    def measure_voltages(self, u_sol):
        """
        Calcula el potencial medio medido en cada uno de los electrodos E1, E2, E3:
        f_j(k) = (1 / |E_k|) int_{E_k} u_j ds
        """
        voltages = np.zeros(3)
        for k_idx, tag in enumerate(self.geom.electrode_ids):
            area_k = self.geom.area_Ek[k_idx]
            v_val = assemble(u_sol * self.geom.ds(tag)) / area_k
            voltages[k_idx] = v_val
        return voltages

    def compute_ntd_matrix(self, sigma_func):
        """
        Calcula la matriz completa F_sigma de 3x3 para las 3 inyecciones j=1,2,3
        """
        F = np.zeros((3, 3))
        potentials = []
        for j in range(1, 4):
            u_j = self.solve(sigma_func, j)
            potentials.append(u_j)
            F[j - 1, :] = self.measure_voltages(u_j)
        return F, potentials