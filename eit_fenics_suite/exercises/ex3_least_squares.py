"""
exercises/ex3_least_squares.py
Ejercicio 3: Mínimos Cuadrados no lineales (Gauss-Newton) para reconstruir
la conductividad en el sistema EIT de cuatro electrodos.

Requiere FEniCS Legacy. Sin fallback analítico.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eit_lib import (
    EITGeometry,
    EITForwardSolver,
    EITSensitivity,
    extract_measurement_vector,
    solve_least_squares,
    reconstruct_quadrants,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_exercise_3():
    print("=" * 65)
    print("EJERCICIO 3: MÍNIMOS CUADRADOS NO LINEAL (GAUSS-NEWTON)")
    print("=" * 65)

    sigma_true = np.array([2.5, 1.0, 1.0, 1.0])  # Anomalía en cuadrante 1
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    # Datos sintéticos verdaderos
    sigma_exact = reconstruct_quadrants(geom.mesh, geom.subdomains, sigma_true)
    F_true, _ = solver.compute_ntd_matrix(sigma_exact)
    b_true = extract_measurement_vector(F_true, pairs, symmetrize=False)

    np.random.seed(42)
    b_meas = b_true + 0.005 * np.random.randn(len(b_true))

    # Gauss-Newton
    sigma_k = np.array([1.0, 1.0, 1.0, 1.0])
    history_res = []

    for it in range(6):
        sigma_func = reconstruct_quadrants(geom.mesh, geom.subdomains, sigma_k)
        F_sim, pots = solver.compute_ntd_matrix(sigma_func)
        b_sim = extract_measurement_vector(F_sim, pairs, symmetrize=False)
        res = b_meas - b_sim
        norm_r = np.linalg.norm(res)
        history_res.append(norm_r)

        A = sens.build_jacobian(pots, num_params=4)
        delta_sigma = solve_least_squares(A, res, alpha=1e-4)
        sigma_k += delta_sigma
        sigma_k = np.clip(sigma_k, 0.2, 5.0)
        print(f"   Iteración {it + 1}: Norma Residuo = {norm_r:.4e} | sigma_k = {np.round(sigma_k, 3)}")
        if norm_r < 1e-4:
            break

    print(f"\nValores Reales de Conductividad:       {sigma_true}")
    print(f"Valores Finales Reconstruidos (GN):    {np.round(sigma_k, 4)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].semilogy(range(1, len(history_res) + 1), history_res, "o-", color="#d62728", lw=2)
    axes[0].set_xlabel("Iteración Gauss-Newton")
    axes[0].set_ylabel(r"Norma del Residuo $\|f_{\mathrm{meas}} - f_k\|$")
    axes[0].set_title("Convergencia del Algoritmo de Mínimos Cuadrados")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    x_idx = np.arange(1, 5)
    w = 0.35
    axes[1].bar(x_idx - w / 2, sigma_true, w, label=r"Verdadero $\sigma_{\mathrm{true}}$", color="#2ca02c")
    axes[1].bar(x_idx + w / 2, sigma_k, w, label=r"Reconstruido $\sigma_{\mathrm{recon}}$", color="#1f77b4")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels([r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"])
    axes[1].set_ylabel("Conductividad [S/m]")
    axes[1].set_title("Reconstrucción por Mínimos Cuadrados")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex3_least_squares_convergence.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Guardado] Imagen guardada en: {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_3()