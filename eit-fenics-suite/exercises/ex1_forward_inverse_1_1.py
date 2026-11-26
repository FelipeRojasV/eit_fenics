"""
Ejercicio 1: Problema 1.1 — forward e inverso en 4 electrodos.
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
)

from fenics import Function, FunctionSpace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


F_meas_table = np.array([
    [-2.0285, -1.3025, -1.0962],
    [-1.3068, -2.3413, -1.3633],
    [-1.1053, -1.3724, -2.5987],
])


def run_exercise_1():
    print("\n[Ej1] forward e inverso, problema 1.1")

    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    V_dg = FunctionSpace(geom.mesh, "DG", 0)
    sigma_0 = Function(V_dg)
    sigma_0.vector()[:] = 1.0

    F_sim, potentials = solver.compute_ntd_matrix(sigma_0)
    A = sens.build_jacobian(potentials, num_params=4)
    pairs = sens.pairs

    b_meas = extract_measurement_vector(F_meas_table, pairs, symmetrize=True)
    b_sim = extract_measurement_vector(F_sim, pairs, symmetrize=False)
    delta_b = b_meas - b_sim
    delta_sigma = solve_least_squares(A, delta_b, alpha=1e-3)
    sigma_recon = 1.0 + delta_sigma

    print("  F_sim =\n", np.round(F_sim, 4))
    print("  sigma_recon =", np.round(sigma_recon, 4))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    grid = np.array([[sigma_recon[1], sigma_recon[0]],
                     [sigma_recon[2], sigma_recon[3]]])
    im = axes[0].imshow(grid, cmap="viridis", extent=[-1, 1, -1, 1], origin="lower")
    axes[0].set_title(r"Conductividad reconstruida $\sigma(\Omega_m)$")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    labels = [f"f_{j}({k})" for j, k in pairs]
    x_pos = np.arange(len(labels))
    w = 0.35
    axes[1].bar(x_pos - w / 2, b_meas, w, label="Medido", color="#1f77b4")
    axes[1].bar(x_pos + w / 2, b_sim, w, label="Simulado", color="#ff7f0e")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Mediciones independientes")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex1_baby_ntd.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()

    save_raw("ex1", {
        "F_sim": F_sim, "F_meas": F_meas_table, "A": A,
        "b_meas": b_meas, "b_sim": b_sim,
        "delta_b": delta_b, "delta_sigma": delta_sigma,
        "sigma_recon": sigma_recon, "pairs": np.array(pairs),
    })
    print(f"  [Guardado] {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_1()
