r"""
Módulo de Extracción de Mediciones y Operador Neumann-to-Dirichlet (NtD).

Notas Metodológicas:
-------------------
1. Medición de Potencial Promediado:
   En los electrodos reales, el potencial medido corresponde a la media integral
   de superficie V_k = (1 / |E_k|) * \int_{E_k} u ds. Esta formulación es
   consistente con el modelo Shunt/Complete Electrode Model y evita la dependencia
   de valores nodales puntuales sensibles al tamaño de malla local.

2. Principio de Reciprocidad de Lorentz:
   Para cualquier medio lineal y pasivo div(sigma grad u) = 0, la matriz de
   transferencia de impedancia F es simétrica (F_{j,k} = F_{k,j}). En una
   configuración de 3 canales activos independientes, existen exactamente:
       N_med = (3 * 4) / 2 = 6 mediciones linealmente independientes.
   Los 6 pares ordenados se eligen como:
       (0,0), (0,1), (0,2), (1,1), (1,2), (2,2).
"""

import numpy as np
from dolfin import assemble


class NtDMeasurementManager:
    """
    Administra la extracción de voltajes en el contorno y la construcción
    de la matriz Baby Neumann-to-Dirichlet (NtD) de 3x3 (6 mediciones independientes).
    """
    def __init__(self, domain, forward_solver):
        self.domain = domain
        self.forward_solver = forward_solver
        self.pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    def read_voltage(self, u_sol, k_elec):
        r"""
        Calcula el potencial medio en el electrodo k_elec mediante integración variacional:
            V_k = (1 / |E_k|) * \int_{E_k} u ds
        """
        area_k = self.domain.area_elec[k_elec - 1]
        return float(assemble(u_sol * self.domain.ds_elec(k_elec)) / area_k)

    def get_simulated_vector(self, conductivity):
        """
        Calcula las 6 lecturas sintéticas independientes asociadas al campo conductivity.
        """
        u_solutions = [self.forward_solver.solve_channel(conductivity, j + 1) for j in range(3)]
        sim_data = [self.read_voltage(u_solutions[j], k + 1) for (j, k) in self.pairs]
        return np.array(sim_data, dtype=float), u_solutions

    def get_simulated_matrix(self, conductivity):
        """
        Construye la matriz completa 3x3 de voltajes simulados F_{j,k} = V_k(I_j).
        """
        u_solutions = [self.forward_solver.solve_channel(conductivity, j + 1) for j in range(3)]
        F_sim = np.zeros((3, 3), dtype=float)
        for j in range(3):
            for k in range(3):
                F_sim[j, k] = self.read_voltage(u_solutions[j], k + 1)
        return F_sim

    @staticmethod
    def get_problem_1_1_reference_matrix():
        """
        Matriz de voltajes medidos para el Problema 1.1 (Seo & Woo / Ammari):
        Valores de referencia obtenidos con conductividad anómala en los cuadrantes.
        """
        return np.array([
            [-2.0285, -1.3025, -1.0962],
            [-1.3068, -2.3413, -1.3633],
            [-1.1053, -1.3724, -2.5987]
        ], dtype=float)

    @classmethod
    def get_problem_1_1_reference_data(cls):
        """
        Vector de las 6 mediciones independientes de referencia f_meas.
        """
        F_meas = cls.get_problem_1_1_reference_matrix()
        pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        return np.array([F_meas[j, k] for (j, k) in pairs], dtype=float)
