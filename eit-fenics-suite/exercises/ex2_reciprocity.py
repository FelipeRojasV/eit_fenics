"""
Ejercicio 2: reciprocidad f_j(k) = f_k(j) y conteo de mediciones independientes.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eit_lib.results_io import save_raw
from eit_lib import EITGeometry, EITForwardSolver

from fenics import FunctionSpace, interpolate, Expression
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_exercise_2():
    print("\n[Ej2] reciprocidad y mediciones independientes")

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

    print(f"  |f1(2)-f2(1)| = {diff_12:.2e}")
    print(f"  |f1(3)-f3(1)| = {diff_13:.2e}")
    print(f"  |f2(3)-f3(2)| = {diff_23:.2e}")

    n = 3
    n_indep = n * (n + 1) // 2
    print(f"  Mediciones independientes: {n_indep}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = axes[0].imshow(np.abs(F - F.T), cmap="hot_r", origin="upper")
    axes[0].set_title(r"$|F - F^T|$")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    vals_jk = [F[0, 1], F[0, 2], F[1, 2]]
    vals_kj = [F[1, 0], F[2, 0], F[2, 1]]
    axes[1].scatter(vals_jk, vals_kj, color="red", s=80)
    axes[1].plot(vals_jk, vals_jk, "k--")
    axes[1].set_xlabel(r"$f_j(k)$")
    axes[1].set_ylabel(r"$f_k(j)$")
    axes[1].set_title("Reciprocidad de Green")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex2_reciprocity.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()

    save_raw("ex2", {
        "F": F,
        "diff_12": np.array(diff_12),
        "diff_13": np.array(diff_13),
        "diff_23": np.array(diff_23),
        "n_indep": np.array(n_indep),
    })
    print(f"  [Guardado] {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_2()
