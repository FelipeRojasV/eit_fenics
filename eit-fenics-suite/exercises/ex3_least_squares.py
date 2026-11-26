"""
Ejercicio 3: mínimos cuadrados no lineales (Gauss-Newton).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eit_lib.results_io import save_raw
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
    print("\n[Ej3] Gauss-Newton")

    sigma_true = np.array([2.5, 1.0, 1.0, 1.0])
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    sigma_exact = reconstruct_quadrants(geom.mesh, geom.subdomains, sigma_true)
    F_true, _ = solver.compute_ntd_matrix(sigma_exact)
    b_true = extract_measurement_vector(F_true, pairs, symmetrize=False)

    np.random.seed(42)
    b_meas = b_true + 0.005 * np.random.randn(len(b_true))

    sigma_k = np.ones(4)
    history_res = []

    for it in range(6):
        sf = reconstruct_quadrants(geom.mesh, geom.subdomains, sigma_k)
        F_sim, pots = solver.compute_ntd_matrix(sf)
        b_sim = extract_measurement_vector(F_sim, pairs, symmetrize=False)
        res = b_meas - b_sim
        norm_r = np.linalg.norm(res)
        history_res.append(norm_r)
        A = sens.build_jacobian(pots, num_params=4)
        sigma_k = np.clip(sigma_k + solve_least_squares(A, res, alpha=1e-4), 0.2, 5.0)
        print(f"  it {it + 1}: res = {norm_r:.4e} | sigma = {np.round(sigma_k, 3)}")
        if norm_r < 1e-4:
            break

    print(f"  true    = {sigma_true}")
    print(f"  recon   = {np.round(sigma_k, 4)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].semilogy(range(1, len(history_res) + 1), history_res, "o-", color="#d62728")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("Residuo")
    axes[0].set_title("Convergencia GN")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    x_idx = np.arange(1, 5)
    w = 0.35
    axes[1].bar(x_idx - w / 2, sigma_true, w, label="True", color="#2ca02c")
    axes[1].bar(x_idx + w / 2, sigma_k, w, label="Recon", color="#1f77b4")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels([r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"])
    axes[1].set_title("Reconstrucción")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex3_least_squares_convergence.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()

    save_raw("ex3", {
        "sigma_true": sigma_true, "sigma_recon": sigma_k,
        "history_res": np.array(history_res),
        "b_meas": b_meas, "b_true": b_true, "pairs": np.array(pairs),
    })
    print(f"  [Guardado] {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_3()
