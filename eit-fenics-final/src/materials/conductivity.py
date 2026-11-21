"""
Módulo de Campo de Conductividad por Subdominios (FEniCS DG0).
"""

import numpy as np
from dolfin import Function


class SubdomainConductivity:
    def __init__(self, domain, initial_val=1.0):
        self.domain = domain
        self.gamma_vals = np.full(4, initial_val, dtype=float)
        self.sigma_fenics = Function(self.domain.DG0)
        self.update_gamma(self.gamma_vals)

    def update_gamma(self, new_gamma):
        """
        Asigna el vector de parámetros gamma_m a los elementos finitos de cada cuadrante.
        """
        self.gamma_vals = np.array(new_gamma, dtype=float)
        sigma_arr = self.sigma_fenics.vector().get_local()
        for m in range(4):
            sub_mask = (self.domain.subdomains.array() == (m + 1))
            sigma_arr[sub_mask] = self.gamma_vals[m]
        self.sigma_fenics.vector().set_local(sigma_arr)
        self.sigma_fenics.vector().apply("insert")

    def get_fenics_function(self):
        return self.sigma_fenics
