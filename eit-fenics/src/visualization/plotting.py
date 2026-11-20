r"""
Módulo de Visualización Científica para Tomografía de Impedancia Eléctrica.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from dolfin import plot as dolfin_plot, project, VectorFunctionSpace, grad


class EITVisualizer:
    """
    Herramientas de visualización científica y análisis gráfico para EIT.
    Proporciona métodos para graficar el dominio discretizado, potenciales directos,
    densidad de corriente J, mapas de sensibilidad adjunta y convergencia inversa.
    """

    @staticmethod
    def _mesh_to_triangulation(mesh):
        coords = mesh.coordinates()
        cells = mesh.cells()
        return tri.Triangulation(coords[:, 0], coords[:, 1], cells)

    @classmethod
    def plot_domain_setup(cls, domain, save_path=None):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        triang = cls._mesh_to_triangulation(domain.mesh)
        sub_vals = domain.subdomains.array()
        tpc = ax.tripcolor(triang, facecolors=sub_vals, cmap="Pastel1", edgecolors="gray", linewidth=0.35)
        cbar = plt.colorbar(tpc, ax=ax, ticks=[1, 2, 3, 4], shrink=0.8)
        cbar.ax.set_yticklabels([r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"])

        centers = [(-0.5, -0.5, r"$\Omega_1$"), (0.5, -0.5, r"$\Omega_2$"),
                   (-0.5, 0.5, r"$\Omega_3$"), (0.5, 0.5, r"$\Omega_4$")]
        for cx, cy, lbl in centers:
            ax.text(cx, cy, lbl, fontsize=14, fontweight="bold", ha="center", va="center", color="#222222")

        w = domain.tol_e
        ax.plot([-w, w], [-1, -1], color="black", linewidth=6, label=r"$E_0$ (Tierra: $u=0$)")
        ax.plot([1, 1], [-w, w], color="#d62728", linewidth=6, label=r"$E_1$ (Canal 1)")
        ax.plot([-w, w], [1, 1], color="#1f77b4", linewidth=6, label=r"$E_2$ (Canal 2)")
        ax.plot([-1, -1], [-w, w], color="#2ca02c", linewidth=6, label=r"$E_3$ (Canal 3)")

        ax.set_title("Discretización FEM, Subdominios ($L=4$) y Electrodos")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_aspect("equal")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @classmethod
    def plot_forward_potentials(cls, domain, u_solutions, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        triang = cls._mesh_to_triangulation(domain.mesh)
        coords = domain.mesh.coordinates()
        V_vec = VectorFunctionSpace(domain.mesh, "P", 1)

        for j in range(3):
            ax = axes[j]
            u_vals = u_solutions[j].compute_vertex_values(domain.mesh)
            tcf = ax.tricontourf(triang, u_vals, levels=20, cmap="viridis")
            ax.tricontour(triang, u_vals, levels=12, colors="white", linewidths=0.6, alpha=0.7)

            J_func = project(-grad(u_solutions[j]), V_vec)
            J_vals = J_func.compute_vertex_values(domain.mesh).reshape(2, -1)
            step = max(1, len(coords) // 180)
            ax.quiver(
                coords[::step, 0], coords[::step, 1],
                J_vals[0, ::step], J_vals[1, ::step],
                color="white", alpha=0.75, scale=18
            )
            plt.colorbar(tcf, ax=ax, shrink=0.82)
            ax.set_title(rf"Inyección Canal {{j+1}}$ ({{j+1}}$ y $\mathbf{{J}}$)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

        plt.suptitle(r"Problema Directo en Medio Homogéneo ($\gamma = 1.0$): Potenciales y Flujo de Corriente", y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @classmethod
    def plot_sensitivity_maps(cls, domain, u_solutions, pairs, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        triang = cls._mesh_to_triangulation(domain.mesh)
        for idx, (j, k) in enumerate(pairs):
            ax = axes[idx // 3, idx % 3]
            sens_expr = (
                u_solutions[j].dx(0) * u_solutions[k].dx(0) +
                u_solutions[j].dx(1) * u_solutions[k].dx(1)
            )
            sens_func = project(sens_expr, domain.DG0)
            cell_vals = sens_func.vector().get_local()

            vmax = np.percentile(np.abs(cell_vals), 95)
            tpc = ax.tripcolor(triang, facecolors=cell_vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            plt.colorbar(tpc, ax=ax, shrink=0.8)
            ax.set_title(f"Sensibilidad Par $(E_{j+1}, E_{k+1})$")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

        plt.suptitle(r"Mapas Espaciales de Sensibilidad Adjunta: $\nabla u_j \cdot \nabla u_k$", y=1.01)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_convergence(history, save_path=None):
        residuals = history["residuals"]
        gammas = np.array(history["gamma_history"])
        iters_res = np.arange(1, len(residuals) + 1)
        iters_gam = np.arange(0, len(gammas))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.semilogy(iters_res, residuals, "o-", color="#1f77b4", linewidth=2)
        ax1.set_title("Convergencia del Residuo de Frontera")
        ax1.set_xlabel("Iteración $k$")
        ax1.set_ylabel(r"Norma del residuo $\|\mathbf{u}^{\gamma} - \mathbf{f}_{\mathrm{meas}}\|_2$")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        labels = [r"$\gamma_1$ ($\Omega_1$)", r"$\gamma_2$ ($\Omega_2$)",
                  r"$\gamma_3$ ($\Omega_3$)", r"$\gamma_4$ ($\Omega_4$)"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for m in range(4):
            ax2.plot(iters_gam, gammas[:, m], "s-", label=labels[m], color=colors[m], linewidth=2)

        ax2.set_title("Evolución de Conductividad por Subdominio")
        ax2.set_xlabel("Iteración $k$")
        ax2.set_ylabel(r"Conductividad estimada $\gamma_m$")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_ntd_comparison(F_meas, F_sim, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
        recip_diff = np.abs(F_sim - F_sim.T)

        matrices = [
            (F_meas, r"Matriz Objetivo $\mathbb{F}_{\mathrm{meas}}$", "coolwarm", None, None),
            (F_sim, r"Matriz Reconstruida $\mathbb{F}_{\mathrm{sim}}$", "coolwarm", None, None),
            (recip_diff, r"Asimetría Reciprocidad $|\mathbb{F}_{\mathrm{sim}} - \mathbb{F}_{\mathrm{sim}}^T|$", "YlOrRd", 0, max(1e-3, float(np.max(recip_diff))))
        ]

        for ax, (mat, title, cmap, vmin, vmax) in zip(axes, matrices):
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title)
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels([r"$E_1$", r"$E_2$", r"$E_3$"])
            ax.set_yticklabels([r"$E_1$", r"$E_2$", r"$E_3$"])
            for i in range(3):
                for j in range(3):
                    fmt = f"{mat[i, j]:.4f}" if vmin is None else f"{mat[i, j]:.1e}"
                    ax.text(j, i, fmt, ha="center", va="center", color="black", fontsize=9, fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_parameter_sensitivity(alphas, gamma_by_alpha, densities, gamma_by_mesh, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

        x_a = np.arange(len(alphas))
        for m in range(4):
            ax1.plot(x_a, gamma_by_alpha[:, m], "o-", label=rf"$\gamma_{m+1}$")
        ax1.set_xticks(x_a)
        ax1.set_xticklabels([f"1e{int(np.log10(a))}" for a in alphas])
        ax1.set_xlabel(r"Parámetro de regularización $\alpha$")
        ax1.set_ylabel(r"Conductividad reconstruida $\gamma_m$")
        ax1.set_title(r"Sensibilidad frente a $\alpha$ de Tikhonov")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        x_d = np.arange(len(densities))
        for m in range(4):
            ax2.plot(x_d, gamma_by_mesh[:, m], "s-", label=rf"$\gamma_{m+1}$")
        ax2.set_xticks(x_d)
        ax2.set_xticklabels([f"{d}x{d}" for d in densities])
        ax2.set_xlabel("Densidad de malla FEM")
        ax2.set_ylabel(r"Conductividad reconstruida $\gamma_m$")
        ax2.set_title("Estudio de Convergencia por Refinamiento de Malla")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @classmethod
    def plot_iteration_strip(cls, domain, gamma_history, save_path=None):
        gammas = np.array(gamma_history)
        n_total = len(gammas)
        indices = np.linspace(0, n_total - 1, min(5, n_total), dtype=int)
        vmin, vmax = float(np.min(gammas)), float(np.max(gammas))

        fig, axes = plt.subplots(1, len(indices), figsize=(3.4 * len(indices), 3.6))
        if len(indices) == 1:
            axes = [axes]

        triang = cls._mesh_to_triangulation(domain.mesh)
        sub_arr = domain.subdomains.array()
        for ax, idx in zip(axes, indices):
            cell_sigma = np.zeros_like(sub_arr, dtype=float)
            for m in range(4):
                cell_sigma[sub_arr == (m + 1)] = gammas[idx, m]
            tpc = ax.tripcolor(triang, facecolors=cell_sigma, cmap="jet", vmin=vmin, vmax=vmax)
            ax.set_title(f"Iteración $k={idx}$")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
        fig.colorbar(tpc, ax=axes, shrink=0.75, label=r"Conductividad $\sigma$")

        plt.suptitle("Evolución Iterativa de la Reconstrucción (Gauss-Newton)", y=1.03)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @classmethod
    def plot_potential_perturbation(cls, domain, u_init_0, u_final_0, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
        triang = cls._mesh_to_triangulation(domain.mesh)
        u0 = u_init_0.compute_vertex_values(domain.mesh)
        uf = u_final_0.compute_vertex_values(domain.mesh)
        du = uf - u0

        panels = [
            (u0, r"Potencial Inicial $u_1^{\gamma_0}$ ($\gamma=1$)", "viridis"),
            (uf, r"Potencial Reconstruido $u_1^{\gamma^*}$", "viridis"),
            (du, r"Perturbación $\delta u_1 = u_1^{\gamma^*} - u_1^{\gamma_0}$", "RdBu_r"),
        ]
        for ax, (vals, title, cmap) in zip(axes, panels):
            tcf = ax.tricontourf(triang, vals, levels=20, cmap=cmap)
            ax.tricontour(triang, vals, levels=10, colors="black", linewidths=0.5, alpha=0.5)
            plt.colorbar(tcf, ax=ax, shrink=0.82)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_reconstruction(conductivity_manager, title=r"Conductividad Reconstruida $\gamma$", save_path=None):
        plt.figure(figsize=(6, 5))
        sigma_func = conductivity_manager.get_fenics_function()
        p = dolfin_plot(sigma_func, title=title, cmap="jet")
        plt.colorbar(p, label=r"Conductividad $\sigma$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
