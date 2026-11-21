r"""
Módulo de Visualización Científica para Tomografía de Impedancia Eléctrica (2D y 3D).
"""

import os
import matplotlib
if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from dolfin import plot as dolfin_plot, project, VectorFunctionSpace, grad


class EITVisualizer:
    """
    Herramientas de visualización científica y análisis gráfico (2D y 3D) para EIT.
    Proporciona métodos para graficar el dominio discretizado, potenciales directos,
    densidad de corriente J, mapas de sensibilidad adjunta, convergencia inversa y
    representaciones tridimensionales (superficies de potencial, prismas de conductividad
    y matrices Baby NtD en 3D).
    """

    @staticmethod
    def _get_cmap(cmap_name):
        """Obtiene un colormap compatible con Matplotlib 3.5+ y 3.9+."""
        if hasattr(matplotlib, "colormaps"):
            return matplotlib.colormaps[cmap_name]
        return cm.get_cmap(cmap_name)

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

        centers = [
            (-0.5, -0.5, r"$\Omega_1$"),
            (0.5, -0.5, r"$\Omega_2$"),
            (-0.5, 0.5, r"$\Omega_3$"),
            (0.5, 0.5, r"$\Omega_4$"),
        ]
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
            ax.set_title(rf"Inyección Canal {j+1} ($u_{j+1}$ y $\mathbf{{J}}$)")
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
            if vmax <= 0:
                vmax = 1.0
            tpc = ax.tripcolor(triang, facecolors=cell_vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            plt.colorbar(tpc, ax=ax, shrink=0.8)
            ax.set_title(rf"Sensibilidad Par $(E_{j+1}, E_{k+1})$")
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

        labels = [
            r"$\gamma_1$ ($\Omega_1$)",
            r"$\gamma_2$ ($\Omega_2$)",
            r"$\gamma_3$ ($\Omega_3$)",
            r"$\gamma_4$ ($\Omega_4$)",
        ]
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
            (
                recip_diff,
                r"Asimetría Reciprocidad $|\mathbb{F}_{\mathrm{sim}} - \mathbb{F}_{\mathrm{sim}}^T|$",
                "YlOrRd",
                0,
                max(1e-3, float(np.max(recip_diff))),
            ),
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

    # ==========================================================================
    # NUEVOS MÉTODOS DE VISUALIZACIÓN 3D PARA EL PROBLEMA 1.1
    # ==========================================================================

    @classmethod
    def plot_potentials_3d(
        cls,
        domain,
        u_solutions,
        suptitle=r"Problema 1.1 — Superficies 3D de Potenciales Eléctricos Reconstruidos ($u_1, u_2, u_3$)",
        save_path=None,
    ):
        """
        Grafica en 3D (plot_trisurf sobre la triangulación FEniCS) los tres campos de potencial
        correspondientes a las inyecciones en los canales activos E_1, E_2 y E_3, junto con
        proyecciones equipotenciales en el plano base.
        """
        triang = cls._mesh_to_triangulation(domain.mesh)
        fig = plt.figure(figsize=(16.5, 5.4), dpi=180)
        canal_desc = [
            r"Canal 1: Inyección $E_1$ ($u_1^{\hat{\gamma}}$)",
            r"Canal 2: Inyección $E_2$ ($u_2^{\hat{\gamma}}$)",
            r"Canal 3: Inyección $E_3$ ($u_3^{\hat{\gamma}}$)",
        ]

        for j in range(3):
            ax = fig.add_subplot(1, 3, j + 1, projection="3d")
            u_vals = u_solutions[j].compute_vertex_values(domain.mesh)
            u_min, u_max = float(np.min(u_vals)), float(np.max(u_vals))
            span = max(u_max - u_min, 1e-6)
            offset = u_min - 0.18 * span

            surf = ax.plot_trisurf(
                triang, u_vals, cmap="viridis", edgecolor="none", antialiased=True, alpha=0.95
            )
            ax.tricontour(triang, u_vals, zdir="z", offset=offset, levels=14, cmap="viridis", linewidths=0.8)

            fig.colorbar(surf, ax=ax, shrink=0.58, aspect=14, pad=0.08, label=rf"$u_{j+1}(x,y)$")
            ax.set_zlim(offset, u_max + 0.05 * span)
            ax.set_xlabel("x", fontsize=9)
            ax.set_ylabel("y", fontsize=9)
            ax.set_zlabel(rf"$u_{j+1}$", fontsize=9)
            ax.set_title(canal_desc[j], fontsize=10.5, pad=8)
            ax.view_init(elev=30, azim=225)

        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.88, wspace=0.14)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @classmethod
    def plot_potential_perturbation_3d(cls, domain, u_init_0, u_final_0, save_path=None):
        """
        Grafica en 3D la comparativa del potencial inicial homogéneo u_1^{gamma_0},
        el potencial final reconstruido u_1^{gamma_hat} y la perturbación delta u_1.
        """
        triang = cls._mesh_to_triangulation(domain.mesh)
        u0 = u_init_0.compute_vertex_values(domain.mesh)
        uf = u_final_0.compute_vertex_values(domain.mesh)
        du = uf - u0
        du_abs = max(float(np.max(np.abs(du))), 1e-6)

        specs = [
            (u0, r"Potencial Inicial 3D $u_1^{\gamma_0}$ ($\gamma=1$)", "viridis", None, None, r"$u_1^{\gamma_0}$"),
            (uf, r"Potencial Reconstruido 3D $u_1^{\hat{\gamma}}$", "viridis", None, None, r"$u_1^{\hat{\gamma}}$"),
            (du, r"Perturbación 3D $\delta u_1 = u_1^{\hat{\gamma}} - u_1^{\gamma_0}$", "RdBu_r", -du_abs, du_abs, r"$\delta u_1$"),
        ]

        fig = plt.figure(figsize=(16.5, 5.4), dpi=180)
        for idx, (vals, title, cmap_name, vmin, vmax, zlbl) in enumerate(specs, start=1):
            ax = fig.add_subplot(1, 3, idx, projection="3d")
            lo = float(np.min(vals)) if vmin is None else vmin
            hi = float(np.max(vals)) if vmax is None else vmax
            span = max(hi - lo, 1e-6)
            offset = lo - 0.18 * span

            kwargs = dict(cmap=cmap_name, edgecolor="none", antialiased=True, alpha=0.95)
            if vmin is not None and vmax is not None:
                kwargs["vmin"] = vmin
                kwargs["vmax"] = vmax
            surf = ax.plot_trisurf(triang, vals, **kwargs)
            ax.tricontour(triang, vals, zdir="z", offset=offset, levels=12, cmap=cmap_name, linewidths=0.8)

            fig.colorbar(surf, ax=ax, shrink=0.58, aspect=14, pad=0.08)
            ax.set_zlim(offset, hi + 0.06 * span)
            ax.set_xlabel("x", fontsize=9)
            ax.set_ylabel("y", fontsize=9)
            ax.set_zlabel(zlbl, fontsize=9)
            ax.set_title(title, fontsize=10, pad=8)
            ax.view_init(elev=30, azim=225)

        fig.suptitle(r"Problema 1.1 — Deformación 3D del Potencial Eléctrico por Heterogeneidad (Canal 1)", fontsize=13, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.88, wspace=0.14)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @classmethod
    def plot_reconstruction_3d(cls, domain, gamma_final, gamma_history=None, save_path=None):
        """
        Panel dual 3D para la conductividad reconstruida del Problema 1.1:
          - Izquierda: prismas 3D de conductividad final en los 4 cuadrantes sobre [-1, 1]^2
            con la ubicación de los electrodos E_0..E_3 marcada en el plano z=0.
          - Derecha: superficie/barras 3D de la evolución de gamma_m(k) a través de las
            iteraciones de Gauss-Newton.
        """
        g_fin = np.asarray(gamma_final, dtype=float)
        has_hist = gamma_history is not None and len(gamma_history) > 1
        fig = plt.figure(figsize=(14.0 if has_hist else 7.5, 5.8), dpi=180)

        # Panel 1: Prismas 3D sobre el dominio [-1, 1]^2
        ax1 = fig.add_subplot(1, 2 if has_hist else 1, 1, projection="3d")
        xs = np.array([-0.92, 0.08, -0.92, 0.08], dtype=float)
        ys = np.array([-0.92, -0.92, 0.08, 0.08], dtype=float)
        zs = np.zeros(4, dtype=float)
        dx = np.full(4, 0.84, dtype=float)
        dy = np.full(4, 0.84, dtype=float)

        zmax = max(float(np.max(g_fin)), 1.0)
        cmap = cls._get_cmap("viridis")
        norm = Normalize(vmin=0.0, vmax=zmax)
        colors = [cmap(norm(val)) for val in g_fin]

        ax1.bar3d(xs, ys, zs, dx, dy, g_fin, color=colors, edgecolor="#1a202c", linewidth=0.7, alpha=0.92, shade=True)

        # Etiquetas sobre cada bloque
        q_tags = [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"]
        for i in range(4):
            ax1.text(
                xs[i] + dx[i] / 2.0,
                ys[i] + dy[i] / 2.0,
                g_fin[i] + 0.05 * zmax,
                f"{q_tags[i]}\n{g_fin[i]:.3f}",
                ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1a202c",
            )

        # Electrodos en el borde del plano z=0
        w = getattr(domain, "tol_e", 0.35)
        ax1.plot([-w, w], [-1.0, -1.0], [0.0, 0.0], color="black", linewidth=4.5, label=r"$E_0$ (Tierra)")
        ax1.plot([1.0, 1.0], [-w, w], [0.0, 0.0], color="#d62728", linewidth=4.5, label=r"$E_1$")
        ax1.plot([-w, w], [1.0, 1.0], [0.0, 0.0], color="#1f77b4", linewidth=4.5, label=r"$E_2$")
        ax1.plot([-1.0, -1.0], [-w, w], [0.0, 0.0], color="#2ca02c", linewidth=4.5, label=r"$E_3$")

        ax1.set_xlim(-1.05, 1.05)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_zlim(0.0, zmax * 1.22)
        ax1.set_xticks([-0.5, 0.5])
        ax1.set_yticks([-0.5, 0.5])
        ax1.set_xlabel("x", fontsize=9)
        ax1.set_ylabel("y", fontsize=9)
        ax1.set_zlabel(r"Conductividad $\hat{\gamma}$", fontsize=9)
        ax1.set_title(r"(a) Distribución 3D Reconstruida $\hat{\gamma}$ por Cuadrante", fontsize=10.5, pad=8)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.view_init(elev=28, azim=-52)

        # Panel 2: Evolución 3D de gamma_m(k) por iteración
        if has_hist:
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            gh = np.asarray(gamma_history, dtype=float)
            n_it = len(gh)
            idx_sel = np.unique(np.linspace(0, n_it - 1, min(8, n_it), dtype=int))
            gh_sel = gh[idx_sel]
            zmax_h = max(float(np.max(gh_sel)), 1.0)
            norm_h = Normalize(vmin=0.0, vmax=zmax_h)

            q_pos, k_pos = np.meshgrid(np.arange(4), np.arange(len(idx_sel)), indexing="ij")
            x_f = q_pos.ravel() - 0.30
            y_f = k_pos.ravel() - 0.28
            z_f = np.zeros_like(x_f, dtype=float)
            dz_f = gh_sel.T.ravel()
            cols_h = [cmap(norm_h(v)) for v in dz_f]

            ax2.bar3d(x_f, y_f, z_f, 0.60, 0.56, dz_f, color=cols_h, edgecolor="#2d3748", linewidth=0.45, alpha=0.90, shade=True)
            ax2.set_xticks([0, 1, 2, 3])
            ax2.set_xticklabels([r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"], fontsize=9)
            ax2.set_yticks(np.arange(len(idx_sel)))
            ax2.set_yticklabels([f"k={k}" for k in idx_sel], fontsize=8)
            ax2.set_zlim(0.0, zmax_h * 1.15)
            ax2.set_xlabel("Subdominio", fontsize=9, labelpad=4)
            ax2.set_ylabel("Iteración", fontsize=9, labelpad=4)
            ax2.set_zlabel(r"$\gamma_m(k)$", fontsize=9)
            ax2.set_title("(b) Evolución 3D de la Reconstrucción (Gauss-Newton)", fontsize=10.5, pad=8)
            ax2.view_init(elev=26, azim=-55)

        fig.suptitle("Problema 1.1 (Seo & Woo) — Reconstrucción Tridimensional de Conductividad", fontsize=13, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.88, wspace=0.14)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @classmethod
    def plot_ntd_comparison_3d(cls, F_meas, F_sim, save_path=None):
        """
        Representación 3D en barras de las matrices Baby NtD 3x3 del Problema 1.1:
        magnitud observada |F_meas|, simulada |F_sim| y discrepancia absoluta |F_sim - F_meas|.
        """
        F_meas = np.asarray(F_meas, dtype=float)
        F_sim = np.asarray(F_sim, dtype=float)
        F_diff = np.abs(F_sim - F_meas)

        zmax_main = max(float(np.max(np.abs(F_meas))), float(np.max(np.abs(F_sim))), 1e-6)
        zmax_diff = max(float(np.max(F_diff)), 1e-3)

        panels = [
            (F_meas, np.abs(F_meas), r"(a) Objetivo $|\mathbb{F}_{\mathrm{meas}}|$ (Seo & Woo)", zmax_main, "coolwarm", "{:.3f}"),
            (F_sim, np.abs(F_sim), r"(b) Reconstruida $|\mathbb{F}_{\mathrm{sim}}|$ (FEniCS)", zmax_main, "coolwarm", "{:.3f}"),
            (F_diff, F_diff, r"(c) Discrepancia $|\mathbb{F}_{\mathrm{sim}} - \mathbb{F}_{\mathrm{meas}}|$", zmax_diff, "YlOrRd", "{:.1e}"),
        ]

        fig = plt.figure(figsize=(16.5, 5.4), dpi=180)
        xpos, ypos = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        x_f = xpos.ravel() - 0.30
        y_f = ypos.ravel() - 0.30
        z_f = np.zeros(9, dtype=float)

        for idx, (raw_mat, h_mat, title, zmax, cmap_name, fmt_s) in enumerate(panels, start=1):
            ax = fig.add_subplot(1, 3, idx, projection="3d")
            dz = h_mat.ravel()
            cmap = cls._get_cmap(cmap_name)
            norm = Normalize(vmin=0.0, vmax=zmax)
            cols = [cmap(norm(v)) for v in dz]

            ax.bar3d(x_f, y_f, z_f, 0.60, 0.60, dz, color=cols, edgecolor="#2d3748", linewidth=0.55, alpha=0.92, shade=True)
            for ii in range(3):
                for jj in range(3):
                    ax.text(
                        ii, jj, h_mat[ii, jj] + 0.05 * zmax,
                        fmt_s.format(raw_mat[ii, jj]),
                        ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#1a202c",
                    )

            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels([r"$E_1$", r"$E_2$", r"$E_3$"], fontsize=8.5)
            ax.set_yticklabels([r"$E_1$", r"$E_2$", r"$E_3$"], fontsize=8.5)
            ax.set_zlim(0.0, zmax * 1.22)
            ax.set_xlabel("Inyección $j$", fontsize=8.5)
            ax.set_ylabel("Medición $k$", fontsize=8.5)
            ax.set_zlabel("Magnitud", fontsize=8.5)
            ax.set_title(title, fontsize=10, pad=8)
            ax.view_init(elev=27, azim=-48)

        fig.suptitle("Problema 1.1 — Comparativa Tridimensional de la Matriz Baby NtD (3x3)", fontsize=13, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.88, wspace=0.14)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
