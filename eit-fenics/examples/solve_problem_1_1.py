"""
Script Principal de Reconstrucción de EIT y Generación de Entregables.

Tesis:
    Tomografía de Impedancia Eléctrica con FEniCS
Caso de Estudio:
    Problema 1.1 de Referencia Cuadrante (Seo & Woo, 2012 / Ammari et al.).
"""

import os
import sys
import logging
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.geometry.domain import Rectangular4ChannelDomain
from src.materials.conductivity import SubdomainConductivity
from src.forward.solver import ForwardSolver
from src.measurements.ntd_map import NtDMeasurementManager
from src.reconstruction.least_squares import GaussNewtonReconstructor
from src.visualization.plotting import EITVisualizer
from src.utils.helpers import verify_reciprocity_matrix


def setup_logger(log_dir):
    """
    Configura el sistema de registro (logging) para emitir a la consola
    y almacenar un archivo persistente .log para auditoría de tesis.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "eit_experiment.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_sensitivity_studies(f_measured):
    logger = logging.getLogger("sensitivity")
    logger.info("Iniciando barrido de hiperparámetros (alpha de Tikhonov y densidad de malla)...")

    alphas = [1e-2, 1e-4, 1e-6, 1e-8]
    gamma_by_alpha = []
    for a in alphas:
        dom = Rectangular4ChannelDomain(mesh_density=32)
        cond = SubdomainConductivity(dom, initial_val=1.0)
        fwd = ForwardSolver(dom)
        meas = NtDMeasurementManager(dom, fwd)
        rec = GaussNewtonReconstructor(dom, cond, meas)
        g_res = rec.solve(f_measured, max_iter=15, alpha=a, verbose=False)
        gamma_by_alpha.append(g_res)

    densities = [16, 32, 48]
    gamma_by_mesh = []
    for d in densities:
        dom = Rectangular4ChannelDomain(mesh_density=d)
        cond = SubdomainConductivity(dom, initial_val=1.0)
        fwd = ForwardSolver(dom)
        meas = NtDMeasurementManager(dom, fwd)
        rec = GaussNewtonReconstructor(dom, cond, meas)
        g_res = rec.solve(f_measured, max_iter=15, alpha=1e-4, verbose=False)
        gamma_by_mesh.append(g_res)

    return alphas, np.array(gamma_by_alpha), densities, np.array(gamma_by_mesh)


def main():
    results_dir = os.path.join(ROOT_DIR, "results")
    fig_dir = os.path.join(results_dir, "figures")
    log_dir = os.path.join(results_dir, "logs")

    os.makedirs(fig_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    logger.info("=================================================================")
    logger.info("  RECONSTRUCCIÓN MODULAR DE EIT - PROBLEMA 1.1 (SEO & WOO)")
    logger.info("=================================================================")

    # 1. Geometría y discretización
    domain = Rectangular4ChannelDomain(mesh_density=32, electrode_width=0.35)
    logger.info("Dominio FEniCS inicializado. Densidad: %d | Celdas: %d", 
                domain.mesh_density, domain.mesh.num_cells())
    EITVisualizer.plot_domain_setup(
        domain, save_path=os.path.join(fig_dir, "dominio_malla_electrodos.png")
    )

    # 2. Material y operadores directos
    cond_mgr = SubdomainConductivity(domain, initial_val=1.0)
    forward_solver = ForwardSolver(domain)
    meas_mgr = NtDMeasurementManager(domain, forward_solver)

    # 3. Carga de datos sintéticos / experimentales de referencia
    f_measured = meas_mgr.get_problem_1_1_reference_data()
    F_meas_mat = meas_mgr.get_problem_1_1_reference_matrix()
    logger.info("Datos de voltaje medidos (vector f_meas): %s", np.array2string(f_measured, precision=4))

    # 4. Inversión numérica
    reconstructor = GaussNewtonReconstructor(domain, cond_mgr, meas_mgr)
    final_gamma, history = reconstructor.solve(
        f_measured, max_iter=15, alpha=1e-4, verbose=True, return_history=True
    )

    logger.info("Conductividad reconstruida final [Omega_1, Omega_2, Omega_3, Omega_4]: %s",
                np.array2string(final_gamma, precision=4))

    # 5. Verificación física de reciprocidad de Lorentz
    F_sim_mat = meas_mgr.get_simulated_matrix(cond_mgr)
    is_sym, max_diff = verify_reciprocity_matrix(F_sim_mat)
    logger.info("Consistencia física de la matriz NtD | Simétrica: %s | Desviación máxima: %.2e",
                is_sym, max_diff)

    # 6. Generación de entregables visuales
    logger.info("Renderizando figuras científicas...")
    EITVisualizer.plot_forward_potentials(
        domain, history["u_initial"],
        save_path=os.path.join(fig_dir, "potenciales_directos_homogeneos.png")
    )
    EITVisualizer.plot_sensitivity_maps(
        domain, history["u_initial"], meas_mgr.pairs,
        save_path=os.path.join(fig_dir, "mapas_sensibilidad_jacobiano.png")
    )
    EITVisualizer.plot_convergence(
        history, save_path=os.path.join(fig_dir, "convergencia_residuo_gamma.png")
    )
    EITVisualizer.plot_ntd_comparison(
        F_meas_mat, F_sim_mat,
        save_path=os.path.join(fig_dir, "matriz_ntd_reciprocidad.png")
    )

    # 7. Estudios de sensibilidad
    alphas, g_alpha, densities, g_mesh = run_sensitivity_studies(f_measured)
    EITVisualizer.plot_parameter_sensitivity(
        alphas, g_alpha, densities, g_mesh,
        save_path=os.path.join(fig_dir, "estudio_regularizacion_malla.png")
    )
    EITVisualizer.plot_iteration_strip(
        domain, history["gamma_history"],
        save_path=os.path.join(fig_dir, "tira_iteraciones_reconstruccion.png")
    )
    EITVisualizer.plot_potential_perturbation(
        domain, history["u_initial"][0], history["u_final"][0],
        save_path=os.path.join(fig_dir, "comparativa_potencial_inicial_vs_final.png")
    )
    EITVisualizer.plot_reconstruction(
        cond_mgr, save_path=os.path.join(fig_dir, "reconstructed_gamma_final.png")
    )

    logger.info("Ejecución completada con éxito. Figuras en: %s | Log en: %s", fig_dir, log_dir)


if __name__ == "__main__":
    main()
