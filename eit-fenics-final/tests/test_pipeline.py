"""
Pruebas Unitarias Automatizadas para el Pipeline EIT.
"""

import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.domain import Rectangular4ChannelDomain
from src.materials.conductivity import SubdomainConductivity
from src.forward.solver import ForwardSolver
from src.measurements.ntd_map import NtDMeasurementManager
from src.reconstruction.least_squares import GaussNewtonReconstructor
from src.utils.helpers import verify_reciprocity_matrix


class TestEITPipeline(unittest.TestCase):
    def test_domain_and_materials(self):
        """Verifica la inicialización del dominio y el campo de conductividad unitario."""
        domain = Rectangular4ChannelDomain(mesh_density=16)
        cond = SubdomainConductivity(domain, initial_val=1.0)
        np.testing.assert_array_equal(cond.gamma_vals, np.ones(4))
        self.assertEqual(len(domain.area_elec), 3)

    def test_forward_and_measurements(self):
        """Verifica el cálculo del problema directo y la simetría física de reciprocidad."""
        domain = Rectangular4ChannelDomain(mesh_density=16)
        cond = SubdomainConductivity(domain, initial_val=1.0)
        forward = ForwardSolver(domain)
        meas = NtDMeasurementManager(domain, forward)
        
        f_vec, u_sols = meas.get_simulated_vector(cond)
        self.assertEqual(len(f_vec), 6)
        self.assertEqual(len(u_sols), 3)
        
        F_mat = meas.get_simulated_matrix(cond)
        self.assertEqual(F_mat.shape, (3, 3))
        is_sym, max_diff = verify_reciprocity_matrix(F_mat)
        self.assertTrue(is_sym, f"La matriz NtD calculada no es simétrica. Máx diff: {max_diff}")

    def test_reconstruction_step(self):
        """Verifica que el bucle de Gauss-Newton ejecute pasos de descenso válidos."""
        domain = Rectangular4ChannelDomain(mesh_density=16)
        cond = SubdomainConductivity(domain, initial_val=1.0)
        forward = ForwardSolver(domain)
        meas = NtDMeasurementManager(domain, forward)
        f_ref = meas.get_problem_1_1_reference_data()

        reconstructor = GaussNewtonReconstructor(domain, cond, meas)
        gamma_res, hist = reconstructor.solve(
            f_ref, max_iter=2, alpha=1e-4, verbose=False, return_history=True
        )
        self.assertEqual(len(gamma_res), 4)
        self.assertIn("residuals", hist)
        self.assertIn("gamma_history", hist)
        self.assertEqual(len(hist["residuals"]), 2)
        # Verificar que los valores de conductividad se mantengan estrictamente positivos
        self.assertTrue(np.all(gamma_res > 0.0))


if __name__ == "__main__":
    unittest.main()
