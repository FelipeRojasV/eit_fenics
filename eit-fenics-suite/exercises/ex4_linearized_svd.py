"""
Ejercicio 4: reconstrucción linealizada con TSVD.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eit_lib.results_io import save_raw
from eit_lib import (
    EITGeometry,
    EITForwardSolver,
    EITSensitivity,
    solve_svd_linearized,
)

from fenics import Function, FunctionSpace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_exercise_4():
    print("\n[Ej4] TSVD linealizado")

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
    cond_A = S[0] / S[-1]
    print(f"  cond(A) = {cond_A:.2f}")
    print(f"  delta_true  = {delta_true}")
    print(f"  delta_recon = {np.round(delta_recon, 4)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(range(1, len(S) + 1), S, "s-", color="#9467bd")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Espectro singular (cond={cond_A:.1f})")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    x_pos = np.arange(1, 5)
    w = 0.35
    axes[1].bar(x_pos - w / 2, delta_true, w, label="True", color="#17becf")
    axes[1].bar(x_pos + w / 2, delta_recon, w, label="TSVD", color="#ff7f0e")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"])
    axes[1].set_title("Reconstrucción TSVD")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "plots", "ex4_svd_spectrum_modes.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()

    save_raw("ex4", {
        "A": A, "delta_true": delta_true, "delta_recon": delta_recon,
        "delta_b": delta_b, "S": S, "U": U, "Vt": Vt,
        "cond_A": np.array(cond_A),
    })
    print(f"  [Guardado] {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    run_exercise_4()
