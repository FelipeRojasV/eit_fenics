"""Pruebas unitarias para la suite v3 de variantes controladas (5 figuras con 3D + 5 tablas)."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiments import (
    ControlledVariant,
    export_variant_tables,
    get_default_variants,
    plot_variant_results,
    run_lcurve_study,
    run_mesh_refinement_study,
    run_single_variant,
)


class TestControlledVariants(unittest.TestCase):
    def test_default_variants_list(self):
        variants = get_default_variants()
        self.assertEqual(len(variants), 8)
        names = [v.name for v in variants]
        self.assertEqual(len(names), len(set(names)))

    def test_homogeneous_variant_recovery(self):
        v = ControlledVariant(
            name="test_homog",
            description="Prueba unitaria medio homogéneo",
            target_gamma=(1.0, 1.0, 1.0, 1.0),
            mesh_density=16,
            alpha=1e-4,
            max_iter=8,
        )
        res = run_single_variant(v, verbose=False)
        self.assertEqual(len(res["estimated_gamma"]), 4)
        self.assertLess(res["rel_err_gamma_pct"], 1.0)
        self.assertTrue(res["reciprocity_ok"])
        self.assertIsNotNone(res["potential_snapshot"])

    def test_five_figures_3d_and_five_tables(self):
        v = ControlledVariant(
            name="V0_base_referencia",
            description="Prueba unitaria contraste en Q2",
            target_gamma=(1.0, 2.0, 1.0, 1.0),
            mesh_density=16,
            alpha=1e-4,
            max_iter=12,
        )
        res = run_single_variant(v, verbose=False)
        self.assertLess(res["rel_err_gamma_pct"], 2.0)

        lc = run_lcurve_study(alphas=(1e-5, 1e-3), mesh_density=16, noise_std=1e-2, verbose=False)
        mr = run_mesh_refinement_study(densities=(16, 24), verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            t_files = export_variant_tables([res], base / "tables", lcurve_results=lc, mesh_results=mr)
            f_files = plot_variant_results([res], base / "figures", lcurve_results=lc, mesh_results=mr)
            # 5 tablas x 3 formatos (.md, .tex, .png) + 3 csv = 18 archivos
            self.assertEqual(len(t_files), 18)
            # 5 figuras esenciales (incluyendo 3D)
            self.assertEqual(len(f_files), 5)
            for p in t_files + f_files:
                self.assertTrue(p.exists() and p.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
