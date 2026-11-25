"""
exercises/ex1_forward_inverse_1_1.py
Ejercicio 1: Problema 1.1 (Ammari, pág. 15-16).
Resuelve el problema directo con conductividad homogénea sigma0 = 1.0,
compara con la matriz baby NtD dada en la tabla y reconstruye la
conductividad por cuadrantes.

Requiere FEniCS Legacy. Sin fallback analítico.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fenics import Function, FunctionSpace

from eit_lib import (
    EITGeometry,
    EITForwardSolver,
    EITSensitivity,
    extract_measurement_vector,
    solve_least_squares,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Matriz baby NtD experimental dada en el Problema 1.1 (pág. 16)
F_meas_table = np.array([
    [-2.0285, -1.3025, -1.0962],
    [-1.3068, -2.3413, -1.3633],
    [-1.1053, -1.3724, -2.5987],
])


def run_exercise_1():
    print("=" * 65)
    print("EJERCICIO 1: PROBLEMA 1.1 - FORWARD E INVERSO EN 4 ELECTRODOS")
    print("=" * 65)

    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    # Forward con sigma_0 = 1.0
    V_dg = FunctionSpace(geom.mesh, "DG", 0)
    sigma_0 = Function(V_dg)
    sigma_0.vector()[:] = 1.0

    F_sim, potentials = solver.compute_ntd_matrix(sigma_0)
    A = sens.build_jacobian(potentials, num_params=4)
    pairs = sens.pairs

    # b_meas se simetriza (tabla con ruido), b_sim ya es simétrica por FEM
    b_meas = extract_measurement_vector(F_meas_table, pairs, symmetrize=True)
    b_sim = extract_measurement_vector(F_sim, pairs, symmetrize=False)
    delta_b = b_meas - b_sim
    delta_sigma = solve_least_squares(A, delta_b, alpha=1e-3)
    sigma_recon = 1.0 + delta_sigma

    print("\n1. Matriz F_sigma simulada (sigma = 1.0):")
    print(np.round(F_sim, 4))
    print("\n2. Matriz F_meas dada en Problema 1.1:")
    print(F_meas_table)
    print("\n3. Conductividad reconstruida en los 4 cuadrantes:")
    for m in range(4):
        print(f"   Cuadrante {m + 1} (Omega_{m + 1}): sigma = {sigma_recon[m]:.4f}")

    # Guardar gráfico de diagnóstico
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    grid = np.array([[sigma_recon[1], sigma_recon[0]],
                     [sigma_recon[2], sigma_recon[3]]])
    im = axes[0].imshow(grid, cmap="viridis", extent=[-1, 1, -1, 1], origin="lower")
    axes[0].set_title(r"Conductividad Reconstruida $\sigma(\Omega_m)$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    for (i, j), val in np.ndenumerate(grid):
        axes[0].text(j * 2 - 1 + 0.5, i * 2 - 1 + 0.5,
                     f"$\\Omega_{{{[2, 1, 3, 4][i * 2 + j]}}}$\n{val:.3f}",
                     ha="center", va="center", color="white", fontweight="bold")

    labels = [f"f_{j}({k})" for j, k in pairs]
    x_pos = np.arange(len(labels))
    w = 0.35
    axes[1].bar(x_pos - w / 2, b_meas, w, label="Medido (Problema 1.1)", color="#1f77b4")
    axes[1].bar(x_pos + w / 2, b_sim, w, label=r"Simulado ($\sigma_0=1.0$)", color="#ff7f0e")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Voltaje [V]")
    axes[1].set_title("Comparación de Mediciones Independientes")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex1_baby_ntd.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Guardado] Imagen guardada en: {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_1()