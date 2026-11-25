"""
exercises/ex4_linearized_svd.py
Ejercicio 4: Reconstrucción linealizada mediante descomposición
en valores singulares (SVD) truncada.

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
    solve_svd_linearized,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_exercise_4():
    print("=" * 65)
    print("EJERCICIO 4: RECONSTRUCCIÓN LINEALIZADA CON SVD (TSVD)")
    print("=" * 65)

    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    V_dg = FunctionSpace(geom.mesh, "DG", 0)
    sigma_0 = Function(V_dg)
    sigma_0.vector()[:] = 1.0

    _, pots_0 = solver.compute_ntd_matrix(sigma_0)
    A = sens.build_jacobian(pots_0, num_params=4)

    delta_true = np.array([0.7, -0.3, 0.1, 0.4])
    np.random.seed(10)
    delta_b = A @ delta_true + 0.002 * np.random.randn(6)

    delta_recon, S, U, Vt = solve_svd_linearized(A, delta_b, n_singular_values=4)

    print("\n1. Espectro de Valores Singulares de la Sensibilidad A:")
    for idx, s in enumerate(S):
        print(f"   s_{idx + 1} = {s:.5e}")
    cond_A = S[0] / S[-1]
    print(f"\n   Número de condición cond(A) = {cond_A:.2f}")

    print("\n2. Comparación de Perturbaciones delta_sigma:")
    for m in range(4):
        print(f"   Cuadrante {m + 1}: Real = {delta_true[m]:+.4f} | Reconstruido = {delta_recon[m]:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(range(1, len(S) + 1), S, "s-", color="#9467bd", lw=2, markersize=8)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(1, len(S) + 1))
    axes[0].set_xlabel("Índice de Valor Singular $i$")
    axes[0].set_ylabel(r"Valor Singular $s_i$ (escala log)")
    axes[0].set_title(f"Decaimiento Espectral y Mal Condicionamiento (cond={cond_A:.1f})")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    x_pos = np.arange(1, 5)
    w = 0.35
    axes[1].bar(x_pos - w / 2, delta_true, w, label=r"$\delta\sigma_{\mathrm{true}}$", color="#17becf")
    axes[1].bar(x_pos + w / 2, delta_recon, w, label=r"$\delta\sigma_{\mathrm{TSVD}}$", color="#ff7f0e")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"])
    axes[1].set_ylabel(r"$\delta\sigma$")
    axes[1].set_title("Reconstrucción Linealizada con TSVD")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex4_svd_spectrum_modes.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Guardado] Imagen guardada en: {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_4()