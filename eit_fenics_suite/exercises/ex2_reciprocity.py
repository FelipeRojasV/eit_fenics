"""
exercises/ex2_reciprocity.py
Ejercicio 2: Demostración de reciprocidad f_j(k) = f_k(j) y conteo
de mediciones independientes.

Requiere FEniCS Legacy. Sin fallback analítico.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fenics import FunctionSpace, interpolate, Expression

from eit_lib import EITGeometry, EITForwardSolver

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_exercise_2():
    print("=" * 65)
    print("EJERCICIO 2: RELACIÓN DE RECIPROCIDAD Y CONTEO DE MEDICIONES")
    print("=" * 65)

    geom = EITGeometry(nx=36, ny=36)
    solver = EITForwardSolver(geom, current_intensity=1.0)

    V_sigma = FunctionSpace(geom.mesh, "Lagrange", 1)
    sigma_expr = Expression(
        "1.0 + 0.9 * exp(-4.0*(pow(x[0]-0.2, 2) + pow(x[1]-0.1, 2)))",
        degree=2,
    )
    sigma = interpolate(sigma_expr, V_sigma)
    F, _ = solver.compute_ntd_matrix(sigma)

    diff_12 = abs(F[0, 1] - F[1, 0])
    diff_13 = abs(F[0, 2] - F[2, 0])
    diff_23 = abs(F[1, 2] - F[2, 1])

    print("\n1. Verificación numérica de la Relación de Reciprocidad:")
    print(f"   f_1(2) = {F[0, 1]:.6f} | f_2(1) = {F[1, 0]:.6f} | Discrepancia = {diff_12:.2e}")
    print(f"   f_1(3) = {F[0, 2]:.6f} | f_3(1) = {F[2, 0]:.6f} | Discrepancia = {diff_13:.2e}")
    print(f"   f_2(3) = {F[1, 2]:.6f} | f_3(2) = {F[2, 1]:.6f} | Discrepancia = {diff_23:.2e}")

    n_electrodes = 4
    n_independent_currents = n_electrodes - 1  # 3
    n_independent_measurements = (n_independent_currents * (n_independent_currents + 1)) // 2

    print("\n2. Determinación formal de mediciones independientes:")
    print(f"   - Electrodos totales en contorno: {n_electrodes}")
    print(f"   - Electrodo de tierra (referencia E0): 1")
    print(f"   - Inyecciones linealmente independientes (n): {n_independent_currents}")
    print(f"   - Tamaño total de la matriz NtD: {n_independent_currents} x {n_independent_currents} = {n_independent_currents ** 2}")
    print(f"   - Fórmula de simetría: N_indep = n*(n+1)/2 = {n_independent_measurements}")
    print(f"   ==> CONCLUSIÓN: En un sistema de 4 electrodos sólo existen {n_independent_measurements} mediciones independientes.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sym_error = np.abs(F - F.T)
    im = axes[0].imshow(sym_error, cmap="hot_r", origin="upper")
    axes[0].set_title(r"Matriz de Error de Reciprocidad $|F - F^T|$")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_xticklabels(["k=1", "k=2", "k=3"])
    axes[0].set_yticklabels(["j=1", "j=2", "j=3"])
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    vals_jk = [F[0, 1], F[0, 2], F[1, 2]]
    vals_kj = [F[1, 0], F[2, 0], F[2, 1]]
    axes[1].scatter(vals_jk, vals_kj, color="red", s=80, zorder=5,
                    label=r"Pares $(f_j(k), f_k(j))$")
    min_v, max_v = min(vals_jk) - 0.2, max(vals_jk) + 0.2
    axes[1].plot([min_v, max_v], [min_v, max_v], "k--",
                 label="Línea de Reciprocidad Exacta y=x")
    axes[1].set_xlabel(r"$f_j(k)$ [V]")
    axes[1].set_ylabel(r"$f_k(j)$ [V]")
    axes[1].set_title("Identidad de Reciprocidad de Green")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex2_reciprocity.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Guardado] Imagen guardada en: {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_2()