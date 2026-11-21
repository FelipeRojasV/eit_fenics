"""Exportación de 5 tablas científicas (.md, .tex, .png, .csv) y 5 figuras esenciales (con vistas 3D)."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def _get_cmap(cmap_name: str):
    """Obtiene un colormap de forma compatible con Matplotlib 3.5+ y 3.9+."""
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[cmap_name]
    return _get_cmap(cmap_name)


def _fmt_vec(vec: Any, decimals: int = 3) -> str:
    if vec is None:
        return "N/A"
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in vec) + "]"


def _render_table_png(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    save_path: Path,
    fontsize: float = 8.5,
) -> Path:
    """Renderiza una tabla científica estilizada en PNG de alta resolución."""
    n_rows = len(rows)
    n_cols = len(headers)
    fig_w = max(10.5, 1.25 * n_cols)
    fig_h = max(2.6, 0.45 * (n_rows + 2.4))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12, color="#1a202c")

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.44)

    for (r_idx, c_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e0")
        cell.set_linewidth(0.7)
        if r_idx == 0:
            cell.set_facecolor("#1a365d")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f7fafc" if r_idx % 2 == 1 else "#ffffff")
            if c_idx == 0:
                cell.set_text_props(fontweight="bold", color="#2d3748")

    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return save_path


def _write_md_tex_png_table(
    tables_dir: Path,
    stem: str,
    title_md: str,
    caption_tex: str,
    label_tex: str,
    headers_md: Sequence[str],
    headers_tex: Sequence[str],
    headers_png: Sequence[str],
    rows_md: Sequence[Sequence[str]],
    rows_tex: Sequence[Sequence[str]],
    rows_png: Sequence[Sequence[str]],
    col_spec_tex: str,
    export_png: bool = True,
) -> List[Path]:
    """Escribe de forma sincronizada las versiones .md, .tex y .png de una tabla."""
    out: List[Path] = []

    md_lines = [
        f"# {title_md}",
        "",
        "| " + " | ".join(headers_md) + " |",
        "| " + " | ".join(["---"] * len(headers_md)) + " |",
    ]
    for row in rows_md:
        md_lines.append("| " + " | ".join(row) + " |")
    p_md = tables_dir / f"{stem}.md"
    p_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    out.append(p_md)

    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption_tex}}}",
        f"\\label{{{label_tex}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec_tex}}}",
        "\\toprule",
        " & ".join(headers_tex) + " \\\\",
        "\\midrule",
    ]
    for row in rows_tex:
        tex_lines.append(" & ".join(row) + " \\\\")
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    p_tex = tables_dir / f"{stem}.tex"
    p_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    out.append(p_tex)

    if export_png:
        p_png = tables_dir / f"{stem}.png"
        _render_table_png(title_md, headers_png, rows_png, p_png)
        out.append(p_png)

    return out


def export_variant_tables(
    results: List[Dict[str, Any]],
    tables_dir: Path,
    lcurve_results: Optional[List[Dict[str, Any]]] = None,
    mesh_results: Optional[List[Dict[str, Any]]] = None,
    export_png_tables: bool = True,
) -> List[Path]:
    """Genera 5 tablas esenciales (.md, .tex, .png) y los datasets .csv de respaldo."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # TABLA 01: Resumen general de variantes controladas
    t1_md, t1_tex, t1_png = [], [], []
    for r in results:
        err_g = f"{r['rel_err_gamma_pct']:.2f}%" if r["rel_err_gamma_pct"] is not None else "N/A"
        err_g_tex = f"{r['rel_err_gamma_pct']:.2f}\\%" if r["rel_err_gamma_pct"] is not None else "N/A"
        mesh_str = f"{r['mesh_density']}x{r['mesh_density']}"
        tex_name = r["name"].replace("_", "\\_")
        t1_md.append([
            f"`{r['name']}`", r["factor"], mesh_str, f"{r['alpha']:.1e}", f"{r['noise_std']:.1e}",
            str(r["iterations"]), f"`{_fmt_vec(r['target_gamma'], 2)}`", f"`{_fmt_vec(r['estimated_gamma'], 3)}`",
            err_g, f"{r['ntd_rel_err_pct']:.3f}%", f"{r['reciprocity_max_diff']:.2e}", f"{r['elapsed_s']:.2f}",
        ])
        t1_tex.append([
            tex_name, mesh_str, f"{r['alpha']:.1e}", f"{r['noise_std']:.1e}",
            str(r["iterations"]), err_g_tex, f"{r['ntd_rel_err_pct']:.3f}\\%",
            f"{r['reciprocity_max_diff']:.2e}", f"{r['elapsed_s']:.2f}",
        ])
        t1_png.append([
            r["name"], mesh_str, f"{r['alpha']:.1e}", f"{r['noise_std']:.1e}",
            str(r["iterations"]), _fmt_vec(r["target_gamma"], 2), _fmt_vec(r["estimated_gamma"], 3),
            err_g, f"{r['ntd_rel_err_pct']:.3f}%", f"{r['reciprocity_max_diff']:.1e}", f"{r['elapsed_s']:.2f}s",
        ])

    generated.extend(_write_md_tex_png_table(
        tables_dir=tables_dir,
        stem="tabla_01_resumen_variantes",
        title_md="Tabla 1: Resumen Cuantitativo de Variantes Controladas (EIT 4 Canales)",
        caption_tex="Resumen cuantitativo de las variantes controladas del sistema EIT de 4 canales.",
        label_tex="tab:var_01_resumen",
        headers_md=["Variante", "Factor", "Malla", "$\\alpha$", "$\\sigma_{\\text{ruido}}$", "Iter.", "$\\gamma^*$", "$\\hat{\\gamma}$", "Error $\\gamma$ [%]", "Error NtD [%]", "$\\max|F-F^T|$", "Tiempo [s]"],
        headers_tex=["Variante", "Malla", "$\\alpha$", "$\\sigma_{\\text{ruido}}$", "Iter.", "Error $\\gamma$ [\\%]", "Error NtD [\\%]", "$\\max|F-F^T|$", "Tiempo [s]"],
        headers_png=["Variante", "Malla", "alpha", "Ruido", "Iter", "gamma*", "gamma_hat", "Err gamma", "Err NtD", "Asim. NtD", "Tiempo"],
        rows_md=t1_md,
        rows_tex=t1_tex,
        rows_png=t1_png,
        col_spec_tex="lcccccccc",
        export_png=export_png_tables,
    ))

    # TABLA 02: Detalle por cuadrante (Omega_1 .. Omega_4)
    t2_md, t2_tex, t2_png = [], [], []
    for r in results:
        est = r["estimated_gamma"]
        tgt = r["target_gamma"]
        rel_q = r["rel_err_per_quadrant_pct"]
        tex_name = r["name"].replace("_", "\\_")
        row_md, row_tex, row_png = [f"`{r['name']}`"], [tex_name], [r["name"]]
        for m in range(4):
            if tgt is not None and rel_q is not None:
                row_md.append(f"{tgt[m]:.2f} / {est[m]:.4f} ({rel_q[m]:.2f}%)")
                row_tex.append(f"{tgt[m]:.2f} / {est[m]:.4f} ({rel_q[m]:.2f}\\%)")
                row_png.append(f"{tgt[m]:.2f} -> {est[m]:.3f}\n({rel_q[m]:.2f}%)")
            else:
                row_md.append(f"N/A / {est[m]:.4f}")
                row_tex.append(f"N/A / {est[m]:.4f}")
                row_png.append(f"N/A -> {est[m]:.3f}")
        row_md.append(f"{r['final_residual']:.3e}")
        row_tex.append(f"{r['final_residual']:.3e}")
        row_png.append(f"{r['final_residual']:.2e}")
        t2_md.append(row_md)
        t2_tex.append(row_tex)
        t2_png.append(row_png)

    generated.extend(_write_md_tex_png_table(
        tables_dir=tables_dir,
        stem="tabla_02_detalle_cuadrantes",
        title_md="Tabla 2: Conductividad Objetivo vs. Reconstruida por Cuadrante",
        caption_tex="Conductividad objetivo ($\\gamma^*$), estimada ($\\hat{\\gamma}$) y error relativo por cuadrante.",
        label_tex="tab:var_02_cuadrantes",
        headers_md=["Variante", "$\\Omega_1$", "$\\Omega_2$", "$\\Omega_3$", "$\\Omega_4$", "Residuo $\\|b\\|_2$"],
        headers_tex=["Variante", "$\\Omega_1$", "$\\Omega_2$", "$\\Omega_3$", "$\\Omega_4$", "Residuo $\\|b\\|_2$"],
        headers_png=["Variante", "Omega_1 (Obj -> Est)", "Omega_2 (Obj -> Est)", "Omega_3 (Obj -> Est)", "Omega_4 (Obj -> Est)", "Residuo ||b||2"],
        rows_md=t2_md,
        rows_tex=t2_tex,
        rows_png=t2_png,
        col_spec_tex="lccccc",
        export_png=export_png_tables,
    ))

    # TABLA 03: Condicionamiento del Jacobiano y Reciprocidad NtD
    t3_md, t3_tex, t3_png = [], [], []
    for r in results:
        tex_name = r["name"].replace("_", "\\_")
        t3_md.append([
            f"`{r['name']}`", f"{r['jac_init_cond']:.2f}", f"{r['jac_final_cond']:.2f}",
            f"{r['ntd_max_abs_err']:.2e}", f"{r['reciprocity_max_diff']:.2e}",
            "Sí" if r["reciprocity_ok"] else "No", f"{r['residual_reduction']:.1f}x",
        ])
        t3_tex.append([
            tex_name, f"{r['jac_init_cond']:.2f}", f"{r['jac_final_cond']:.2f}",
            f"{r['ntd_max_abs_err']:.2e}", f"{r['reciprocity_max_diff']:.2e}",
            "Sí" if r["reciprocity_ok"] else "No", f"{r['residual_reduction']:.1f}$\\times$",
        ])
        t3_png.append([
            r["name"], f"{r['jac_init_cond']:.2f}", f"{r['jac_final_cond']:.2f}",
            f"{r['ntd_max_abs_err']:.2e}", f"{r['reciprocity_max_diff']:.2e}",
            "OK" if r["reciprocity_ok"] else "NO", f"{r['residual_reduction']:.1f}x",
        ])

    generated.extend(_write_md_tex_png_table(
        tables_dir=tables_dir,
        stem="tabla_03_condicionamiento_y_reciprocidad",
        title_md="Tabla 3: Condicionamiento del Jacobiano, Ajuste NtD y Reciprocidad Física",
        caption_tex="Número de condición del Jacobiano $\\kappa(A)$, error máximo en mediciones NtD, verificación de reciprocidad y reducción del residuo.",
        label_tex="tab:var_03_cond_recip",
        headers_md=["Variante", "$\\kappa(A_0)$", "$\\kappa(A_K)$", "$\\max|\\Delta f|$", "$\\max|F-F^T|$", "Reciprocidad", "Reducción"],
        headers_tex=["Variante", "$\\kappa(A_0)$", "$\\kappa(A_K)$", "$\\max|\\Delta f|$", "$\\max|F-F^T|$", "Recip.", "Reducción"],
        headers_png=["Variante", "cond(A0)", "cond(AK)", "Max |df|", "Max |F-FT|", "Recip.", "Reducción"],
        rows_md=t3_md,
        rows_tex=t3_tex,
        rows_png=t3_png,
        col_spec_tex="lcccccc",
        export_png=export_png_tables,
    ))

    # TABLA 04: Estudio paramétrico de Regularización Tikhonov (Curva L)
    if lcurve_results:
        t4_md, t4_tex, t4_png = [], [], []
        for r in lcurve_results:
            err_g = f"{r['rel_err_gamma_pct']:.2f}%" if r["rel_err_gamma_pct"] is not None else "N/A"
            err_g_tex = f"{r['rel_err_gamma_pct']:.2f}\\%" if r["rel_err_gamma_pct"] is not None else "N/A"
            t4_md.append([
                f"{r['alpha']:.1e}", str(r["iterations"]), f"`{_fmt_vec(r['estimated_gamma'], 3)}`",
                f"{r['final_residual']:.4e}", f"{r['gamma_norm']:.4f}", err_g, f"{r['ntd_rel_err_pct']:.3f}%",
            ])
            t4_tex.append([
                f"{r['alpha']:.1e}", str(r["iterations"]), _fmt_vec(r["estimated_gamma"], 3),
                f"{r['final_residual']:.4e}", f"{r['gamma_norm']:.4f}", err_g_tex, f"{r['ntd_rel_err_pct']:.3f}\\%",
            ])
            t4_png.append([
                f"{r['alpha']:.1e}", str(r["iterations"]), _fmt_vec(r["estimated_gamma"], 3),
                f"{r['final_residual']:.3e}", f"{r['gamma_norm']:.3f}", err_g, f"{r['ntd_rel_err_pct']:.3f}%",
            ])

        generated.extend(_write_md_tex_png_table(
            tables_dir=tables_dir,
            stem="tabla_04_estudio_regularizacion_lcurve",
            title_md="Tabla 4: Barrido de Regularización de Tikhonov (Curva L bajo Ruido)",
            caption_tex="Estudio paramétrico del hiperparámetro $\\alpha$ bajo mediciones perturbadas con ruido gaussiano.",
            label_tex="tab:var_04_lcurve",
            headers_md=["$\\alpha$", "Iter.", "$\\hat{\\gamma}$", "Residuo $\\|b\\|_2$", "$\\|\\hat{\\gamma}\\|_2$", "Error $\\gamma$ [%]", "Error NtD [%]"],
            headers_tex=["$\\alpha$", "Iter.", "$\\hat{\\gamma}$", "$\\|b\\|_2$", "$\\|\\hat{\\gamma}\\|_2$", "Error $\\gamma$ [\\%]", "Error NtD [\\%]"],
            headers_png=["alpha", "Iter", "gamma_hat", "Residuo ||b||2", "||gamma||2", "Err gamma", "Err NtD"],
            rows_md=t4_md,
            rows_tex=t4_tex,
            rows_png=t4_png,
            col_spec_tex="ccccccc",
            export_png=export_png_tables,
        ))

    # TABLA 05: Estudio paramétrico de Refinamiento de Malla
    if mesh_results:
        t5_md, t5_tex, t5_png = [], [], []
        for r in mesh_results:
            err_g = f"{r['rel_err_gamma_pct']:.3f}%" if r["rel_err_gamma_pct"] is not None else "N/A"
            err_g_tex = f"{r['rel_err_gamma_pct']:.3f}\\%" if r["rel_err_gamma_pct"] is not None else "N/A"
            mesh_str = f"{r['mesh_density']}x{r['mesh_density']}"
            t5_md.append([
                mesh_str, str(r["num_cells"]), str(r["num_vertices"]), str(r["iterations"]),
                f"`{_fmt_vec(r['estimated_gamma'], 4)}`", err_g, f"{r['reciprocity_max_diff']:.2e}", f"{r['elapsed_s']:.2f}",
            ])
            t5_tex.append([
                mesh_str, str(r["num_cells"]), str(r["num_vertices"]), str(r["iterations"]),
                _fmt_vec(r["estimated_gamma"], 4), err_g_tex, f"{r['reciprocity_max_diff']:.2e}", f"{r['elapsed_s']:.2f}",
            ])
            t5_png.append([
                mesh_str, str(r["num_cells"]), str(r["num_vertices"]), str(r["iterations"]),
                _fmt_vec(r["estimated_gamma"], 3), err_g, f"{r['reciprocity_max_diff']:.1e}", f"{r['elapsed_s']:.2f}s",
            ])

        generated.extend(_write_md_tex_png_table(
            tables_dir=tables_dir,
            stem="tabla_05_estudio_refinamiento_malla",
            title_md="Tabla 5: Estudio de Convergencia y Costo Computacional por Refinamiento de Malla",
            caption_tex="Estudio de refinamiento de malla sobre el caso base: precisión de reconstrucción, reciprocidad y costo computacional.",
            label_tex="tab:var_05_malla",
            headers_md=["Malla", "Celdas", "Vértices", "Iter.", "$\\hat{\\gamma}$", "Error $\\gamma$ [%]", "$\\max|F-F^T|$", "Tiempo [s]"],
            headers_tex=["Malla", "Celdas", "Vértices", "Iter.", "$\\hat{\\gamma}$", "Error $\\gamma$ [\\%]", "$\\max|F-F^T|$", "Tiempo [s]"],
            headers_png=["Malla", "Celdas", "Vértices", "Iter", "gamma_hat", "Err gamma", "Asim. NtD", "Tiempo"],
            rows_md=t5_md,
            rows_tex=t5_tex,
            rows_png=t5_png,
            col_spec_tex="cccccccc",
            export_png=export_png_tables,
        ))

    # Datasets CSV
    def _export_csv(data_list: List[Dict[str, Any]], csv_path: Path) -> Path:
        fieldnames = [
            "name", "factor", "backend", "mesh_density", "num_cells", "num_vertices",
            "alpha", "noise_std", "iterations", "initial_residual", "final_residual",
            "residual_reduction", "target_g1", "target_g2", "target_g3", "target_g4",
            "est_g1", "est_g2", "est_g3", "est_g4", "gamma_norm",
            "rel_err_gamma_pct", "ntd_rel_err_pct", "ntd_max_abs_err",
            "reciprocity_ok", "reciprocity_max_diff", "reciprocity_fro",
            "jac_init_cond", "jac_final_cond", "elapsed_s", "time_per_iter_s",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in data_list:
                tgt = r["target_gamma"] or [None, None, None, None]
                est = r["estimated_gamma"]
                writer.writerow({
                    "name": r["name"], "factor": r["factor"], "backend": r["backend"],
                    "mesh_density": r["mesh_density"], "num_cells": r["num_cells"], "num_vertices": r["num_vertices"],
                    "alpha": r["alpha"], "noise_std": r["noise_std"], "iterations": r["iterations"],
                    "initial_residual": r["initial_residual"], "final_residual": r["final_residual"],
                    "residual_reduction": r["residual_reduction"],
                    "target_g1": tgt[0], "target_g2": tgt[1], "target_g3": tgt[2], "target_g4": tgt[3],
                    "est_g1": est[0], "est_g2": est[1], "est_g3": est[2], "est_g4": est[3],
                    "gamma_norm": r["gamma_norm"], "rel_err_gamma_pct": r["rel_err_gamma_pct"],
                    "ntd_rel_err_pct": r["ntd_rel_err_pct"], "ntd_max_abs_err": r["ntd_max_abs_err"],
                    "reciprocity_ok": r["reciprocity_ok"], "reciprocity_max_diff": r["reciprocity_max_diff"],
                    "reciprocity_fro": r["reciprocity_fro"], "jac_init_cond": r["jac_init_cond"],
                    "jac_final_cond": r["jac_final_cond"], "elapsed_s": r["elapsed_s"],
                    "time_per_iter_s": r["time_per_iter_s"],
                })
        return csv_path

    generated.append(_export_csv(results, tables_dir / "resultados_variantes_controladas.csv"))
    if lcurve_results:
        generated.append(_export_csv(lcurve_results, tables_dir / "resultados_lcurve.csv"))
    if mesh_results:
        generated.append(_export_csv(mesh_results, tables_dir / "resultados_refinamiento_malla.csv"))

    return generated


def _draw_quadrant_blocks_3d(ax: Any, gamma_vals: Sequence[float], title: str, zmax: float, cmap_name: str = "viridis") -> None:
    """Dibuja los 4 cuadrantes Omega_1..Omega_4 como prismas 3D sobre [-1, 1]^2."""
    # Posiciones de esquina inferior izquierda (x0, y0) para Omega_1, Omega_2, Omega_3, Omega_4
    xs = np.array([-0.92, 0.08, -0.92, 0.08], dtype=float)
    ys = np.array([-0.92, -0.92, 0.08, 0.08], dtype=float)
    zs = np.zeros(4, dtype=float)
    dx = np.full(4, 0.84, dtype=float)
    dy = np.full(4, 0.84, dtype=float)
    dz = np.asarray(gamma_vals, dtype=float)

    norm = Normalize(vmin=0.0, vmax=max(zmax, 1e-6))
    cmap = _get_cmap(cmap_name)
    colors = [cmap(norm(val)) for val in dz]

    ax.bar3d(xs, ys, zs, dx, dy, dz, color=colors, edgecolor="#1a202c", linewidth=0.6, alpha=0.92, shade=True)

    q_tags = [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"]
    for i in range(4):
        cx = xs[i] + dx[i] / 2.0
        cy = ys[i] + dy[i] / 2.0
        ax.text(cx, cy, dz[i] + 0.06 * zmax, f"{q_tags[i]}\n{dz[i]:.2f}", ha="center", va="bottom",
                fontsize=7.8, fontweight="bold", color="#1a202c")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, zmax * 1.18)
    ax.set_xticks([-0.5, 0.5])
    ax.set_yticks([-0.5, 0.5])
    ax.set_xlabel("x", fontsize=8, labelpad=2)
    ax.set_ylabel("y", fontsize=8, labelpad=2)
    ax.set_zlabel(r"$\hat{\gamma}$", fontsize=8.5, labelpad=2)
    ax.set_title(title, fontsize=9.2, pad=6)
    ax.view_init(elev=28, azim=-52)


def _draw_matrix_bars_3d(ax: Any, mat: np.ndarray, title: str, zmax: float, cmap_name: str = "plasma") -> None:
    """Representa una matriz 3x3 Baby NtD como barras 3D de magnitud |F_{j,k}|."""
    vals = np.abs(mat)
    xpos, ypos = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    x_flat = xpos.ravel() - 0.32
    y_flat = ypos.ravel() - 0.32
    z_flat = np.zeros(9, dtype=float)
    dx = np.full(9, 0.64, dtype=float)
    dy = np.full(9, 0.64, dtype=float)
    dz = vals.ravel()

    norm = Normalize(vmin=0.0, vmax=max(zmax, 1e-6))
    cmap = _get_cmap(cmap_name)
    colors = [cmap(norm(v)) for v in dz]

    ax.bar3d(x_flat, y_flat, z_flat, dx, dy, dz, color=colors, edgecolor="#2d3748", linewidth=0.5, alpha=0.92, shade=True)
    for ii in range(3):
        for jj in range(3):
            ax.text(ii, jj, vals[ii, jj] + 0.05 * zmax, f"{mat[ii, jj]:.2f}", ha="center", va="bottom",
                    fontsize=7.2, fontweight="bold", color="#1a202c")

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels([r"$E_1$", r"$E_2$", r"$E_3$"], fontsize=8)
    ax.set_yticklabels([r"$E_1$", r"$E_2$", r"$E_3$"], fontsize=8)
    ax.set_zlim(0.0, zmax * 1.18)
    ax.set_xlabel("Inyección $j$", fontsize=8, labelpad=2)
    ax.set_ylabel("Medición $k$", fontsize=8, labelpad=2)
    ax.set_zlabel(r"$|\mathbb{F}_{jk}|$", fontsize=8, labelpad=2)
    ax.set_title(title, fontsize=9.2, pad=6)
    ax.view_init(elev=27, azim=-48)


def plot_variant_results(
    results: List[Dict[str, Any]],
    figures_dir: Path,
    lcurve_results: Optional[List[Dict[str, Any]]] = None,
    mesh_results: Optional[List[Dict[str, Any]]] = None,
) -> List[Path]:
    """Genera las 5 figuras esenciales de análisis (incluyendo representaciones 3D)."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    names = [r["name"] for r in results]
    x_idx = np.arange(len(names))
    synth = [r for r in results if r["rel_err_gamma_pct"] is not None]
    q_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # ==========================================================================
    # FIGURA 01 (3D): Reconstrucción 3D de Conductividad por Cuadrantes (Casos Clave)
    # ==========================================================================
    pref_3d = [
        "V0_base_referencia",
        "V2_contraste_multicuadrante",
        "V3_anomalia_aislante_q4",
        "V7_referencia_seo_woo",
    ]
    sel_3d = [r for r in results if r["name"] in pref_3d]
    if len(sel_3d) < 4:
        for r in results:
            if r not in sel_3d and len(sel_3d) < 4:
                sel_3d.append(r)

    zmax_g = max(max(r["estimated_gamma"]) for r in sel_3d)
    fig = plt.figure(figsize=(13.5, 9.2), dpi=180)
    for idx_p, r in enumerate(sel_3d, start=1):
        ax = fig.add_subplot(2, 2, idx_p, projection="3d")
        subtitle = (
            f"{r['name']}\n(Err. $\\gamma$ = {r['rel_err_gamma_pct']:.2f}%)"
            if r["rel_err_gamma_pct"] is not None
            else f"{r['name']}\n(Datos Seo & Woo | Err. NtD = {r['ntd_rel_err_pct']:.2f}%)"
        )
        _draw_quadrant_blocks_3d(ax, r["estimated_gamma"], subtitle, zmax=zmax_g)

    fig.suptitle(r"Figura 1: Representación 3D de la Conductividad Reconstruida $\hat{\gamma}$ por Cuadrante", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.90, wspace=0.14, hspace=0.22)
    p1 = figures_dir / "var_01_reconstruccion_3d_variantes.png"
    fig.savefig(p1, bbox_inches="tight", dpi=180)
    plt.close(fig)
    generated.append(p1)

    # ==========================================================================
    # FIGURA 02 (3D): Superficies 3D de Potencial Eléctrico y Perturbación (Canal 1)
    # ==========================================================================
    pot_pref = ["V0_base_referencia", "V3_anomalia_aislante_q4", "V7_referencia_seo_woo"]
    pot_sel = [r for r in results if r["name"] in pot_pref and r.get("potential_snapshot") is not None]
    if not pot_sel:
        pot_sel = [r for r in results if r.get("potential_snapshot") is not None][:3]

    if pot_sel:
        n_rows = len(pot_sel)
        fig = plt.figure(figsize=(13.5, 4.4 * n_rows), dpi=180)
        for row_i, r in enumerate(pot_sel):
            snap = r["potential_snapshot"]
            uf = snap["uf"]
            du = snap["du"]

            # Columna 1: Superficie 3D del potencial reconstruido u_1
            ax1 = fig.add_subplot(n_rows, 2, 2 * row_i + 1, projection="3d")
            uf_min = float(np.min(uf))
            uf_max = float(np.max(uf))
            z_span_u = max(uf_max - uf_min, 1e-6)
            offset_u = uf_min - 0.18 * z_span_u

            if snap["mode"] == "tri":
                triang = tri.Triangulation(snap["x"], snap["y"], snap["cells"])
                s1 = ax1.plot_trisurf(triang, uf, cmap="viridis", edgecolor="none", antialiased=True, alpha=0.95)
                ax1.tricontour(triang, uf, zdir="z", offset=offset_u, levels=12, cmap="viridis", linewidths=0.8)
            else:
                s1 = ax1.plot_surface(snap["X"], snap["Y"], uf, cmap="viridis", edgecolor="none", antialiased=True, alpha=0.95)
                ax1.contour(snap["X"], snap["Y"], uf, zdir="z", offset=offset_u, levels=12, cmap="viridis", linewidths=0.8)

            fig.colorbar(s1, ax=ax1, shrink=0.58, aspect=14, pad=0.08)
            ax1.set_zlim(offset_u, uf_max + 0.05 * z_span_u)
            ax1.set_xlabel("x", fontsize=8)
            ax1.set_ylabel("y", fontsize=8)
            ax1.set_zlabel(r"$u_1(x,y)$", fontsize=8)
            ax1.set_title(f"{r['name']} — Potencial Reconstruido 3D $u_1^{{\\hat{{\\gamma}}}}$", fontsize=9.2)
            ax1.view_init(elev=30, azim=225)

            # Columna 2: Superficie 3D de la perturbación delta u_1 = u_final - u_init
            ax2 = fig.add_subplot(n_rows, 2, 2 * row_i + 2, projection="3d")
            du_abs = max(float(np.max(np.abs(du))), 1e-6)
            offset_du = -1.25 * du_abs

            if snap["mode"] == "tri":
                triang = tri.Triangulation(snap["x"], snap["y"], snap["cells"])
                s2 = ax2.plot_trisurf(
                    triang, du, cmap="RdBu_r", vmin=-du_abs, vmax=du_abs,
                    edgecolor="none", antialiased=True, alpha=0.95
                )
                ax2.tricontour(triang, du, zdir="z", offset=offset_du, levels=12, cmap="RdBu_r", linewidths=0.8)
            else:
                s2 = ax2.plot_surface(
                    snap["X"], snap["Y"], du, cmap="RdBu_r", vmin=-du_abs, vmax=du_abs,
                    edgecolor="none", antialiased=True, alpha=0.95
                )
                ax2.contour(snap["X"], snap["Y"], du, zdir="z", offset=offset_du, levels=12, cmap="RdBu_r", linewidths=0.8)

            fig.colorbar(s2, ax=ax2, shrink=0.58, aspect=14, pad=0.08)
            ax2.set_zlim(offset_du, 1.15 * du_abs)
            ax2.set_xlabel("x", fontsize=8)
            ax2.set_ylabel("y", fontsize=8)
            ax2.set_zlabel(r"$\delta u_1$", fontsize=8)
            ax2.set_title(f"{r['name']} — Perturbación 3D $\\delta u_1 = u_1^{{\\hat{{\\gamma}}}} - u_1^{{\\gamma_0}}$", fontsize=9.2)
            ax2.view_init(elev=30, azim=225)

        fig.suptitle(r"Figura 2: Superficies 3D de Potencial Eléctrico y Perturbación por Contraste (Canal 1)", fontsize=13, fontweight="bold", y=0.995)
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.93, wspace=0.14, hspace=0.22)
        p2 = figures_dir / "var_02_superficie_3d_potenciales.png"
        fig.savefig(p2, bbox_inches="tight", dpi=180)
        plt.close(fig)
        generated.append(p2)

    # ==========================================================================
    # FIGURA 03 (2D): Convergencia de Gauss-Newton, Errores Globales y Trayectorias
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=180)

    # Panel (a): Convergencia logarítmica del residuo en todas las variantes
    for r in results:
        res_list = [max(v, 1e-12) for v in r["residuals"]]
        axes[0, 0].semilogy(np.arange(1, len(res_list) + 1), res_list, marker="o", linewidth=1.6, markersize=4.2, label=r["name"])
    axes[0, 0].set_xlabel("Iteración de Gauss-Newton $k$")
    axes[0, 0].set_ylabel(r"Norma del residuo $\|b_k\|_2$")
    axes[0, 0].set_title("(a) Convergencia del Residuo de Frontera")
    axes[0, 0].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0, 0].legend(fontsize=7.3, ncol=2)

    # Panel (b): Error relativo en gamma y en NtD por variante
    err_g_all = [r["rel_err_gamma_pct"] if r["rel_err_gamma_pct"] is not None else 0.0 for r in results]
    err_ntd_all = [r["ntd_rel_err_pct"] for r in results]
    w_b = 0.38
    axes[0, 1].bar(x_idx - w_b / 2, err_g_all, width=w_b, color="#2b6cb0", label=r"Error $\gamma$ [%] (sintéticas)")
    axes[0, 1].bar(x_idx + w_b / 2, err_ntd_all, width=w_b, color="#2f855a", label="Error Baby NtD [%]")
    axes[0, 1].set_xticks(x_idx)
    axes[0, 1].set_xticklabels(names, rotation=28, ha="right", fontsize=7.8)
    axes[0, 1].set_ylabel("Error relativo [%]")
    axes[0, 1].set_title("(b) Exactitud de Reconstrucción y Ajuste NtD")
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0, 1].legend(fontsize=8)

    # Panel (c): Trayectoria iterativa de gamma_m en variante multicuadrante (o primera sintética)
    r_c = next((r for r in results if r["name"] == "V2_contraste_multicuadrante"), synth[0] if synth else results[0])
    gh_c = np.asarray(r_c["gamma_history"], dtype=float)
    for m in range(4):
        axes[1, 0].plot(np.arange(len(gh_c)), gh_c[:, m], "o-", color=q_colors[m], linewidth=1.8, label=rf"$\hat{{\gamma}}_{m+1}$")
        if r_c["target_gamma"] is not None:
            axes[1, 0].axhline(r_c["target_gamma"][m], color=q_colors[m], linestyle="--", linewidth=1.0, alpha=0.55)
    axes[1, 0].set_xlabel("Iteración $k$")
    axes[1, 0].set_ylabel(r"Conductividad $\gamma_m$")
    axes[1, 0].set_title(f"(c) Evolución Iterativa: {r_c['name']} (Punteada = Objetivo)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(fontsize=8, ncol=2)

    # Panel (d): Trayectoria iterativa de gamma_m en variante con datos reales (o ruido)
    r_d = next((r for r in results if r["name"] == "V7_referencia_seo_woo"), results[-1])
    gh_d = np.asarray(r_d["gamma_history"], dtype=float)
    for m in range(4):
        axes[1, 1].plot(np.arange(len(gh_d)), gh_d[:, m], "s-", color=q_colors[m], linewidth=1.8, label=rf"$\hat{{\gamma}}_{m+1}$")
        if r_d["target_gamma"] is not None:
            axes[1, 1].axhline(r_d["target_gamma"][m], color=q_colors[m], linestyle="--", linewidth=1.0, alpha=0.55)
    axes[1, 1].set_xlabel("Iteración $k$")
    axes[1, 1].set_ylabel(r"Conductividad $\gamma_m$")
    axes[1, 1].set_title(f"(d) Evolución Iterativa: {r_d['name']}")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(fontsize=8, ncol=2)

    fig.suptitle("Figura 3: Convergencia de Gauss-Newton, Errores Globales y Estabilidad por Subdominio", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p3 = figures_dir / "var_03_convergencia_y_errores.png"
    fig.savefig(p3, bbox_inches="tight", dpi=180)
    plt.close(fig)
    generated.append(p3)

    # ==========================================================================
    # FIGURA 04 (3D + 2D): Matrices Baby NtD en 3D, Reciprocidad y Condicionamiento
    # ==========================================================================
    r_ref = next((r for r in results if r["name"] == "V7_referencia_seo_woo"), results[0])
    F_obs_ref = np.asarray(r_ref["F_obs"], dtype=float)
    F_sim_ref = np.asarray(r_ref["F_sim"], dtype=float)
    zmax_ntd = max(float(np.max(np.abs(F_obs_ref))), float(np.max(np.abs(F_sim_ref))))

    fig = plt.figure(figsize=(13.5, 9.0), dpi=180)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_matrix_bars_3d(ax1, F_obs_ref, f"(a) Matriz Observada 3D $|\\mathbb{{F}}_{{\\mathrm{{obs}}}}|$ ({r_ref['name']})", zmax=zmax_ntd, cmap_name="coolwarm")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _draw_matrix_bars_3d(ax2, F_sim_ref, f"(b) Matriz Reconstruida 3D $|\\mathbb{{F}}_{{\\mathrm{{sim}}}}|$ ({r_ref['name']})", zmax=zmax_ntd, cmap_name="coolwarm")

    ax3 = fig.add_subplot(2, 2, 3)
    asym_vals = [max(r["reciprocity_max_diff"], 1e-16) for r in results]
    ax3.semilogy(x_idx, asym_vals, marker="s", linewidth=1.8, color="#c05621", label=r"$\max|\mathbb{F} - \mathbb{F}^T|$")
    ax3.axhline(1e-2, color="red", linestyle="--", linewidth=1.0, label="Tolerancia (1e-2)")
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(names, rotation=28, ha="right", fontsize=7.8)
    ax3.set_ylabel(r"Asimetría $\max|\mathbb{F} - \mathbb{F}^T|$")
    ax3.set_title("(c) Verificación del Principio de Reciprocidad")
    ax3.grid(True, which="both", linestyle=":", alpha=0.6)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 2, 4)
    cond_init = [r["jac_init_cond"] for r in results]
    cond_final = [r["jac_final_cond"] for r in results]
    ax4.bar(x_idx - w_b / 2, cond_init, width=w_b, color="#718096", label=r"$\kappa(A_0)$ Inicial")
    ax4.bar(x_idx + w_b / 2, cond_final, width=w_b, color="#dd6b20", label=r"$\kappa(A_K)$ Final")
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(names, rotation=28, ha="right", fontsize=7.8)
    ax4.set_ylabel(r"Número de condición $\kappa(A)$")
    ax4.set_title("(d) Condicionamiento del Jacobiano $A \\in \\mathbb{R}^{6\\times 4}$")
    ax4.grid(axis="y", linestyle=":", alpha=0.6)
    ax4.legend(fontsize=8)

    fig.suptitle("Figura 4: Representación 3D del Operador Baby NtD, Reciprocidad y Condicionamiento", fontsize=13, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.11, top=0.91, wspace=0.22, hspace=0.34)
    p4 = figures_dir / "var_04_ntd_3d_y_reciprocidad.png"
    fig.savefig(p4, bbox_inches="tight", dpi=180)
    plt.close(fig)
    generated.append(p4)

    # ==========================================================================
    # FIGURA 05 (2D): Estudio Paramétrico Consolidado (Regularización L-Curve + Malla)
    # ==========================================================================
    if lcurve_results or mesh_results:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), dpi=180)

        if lcurve_results:
            alphas_lc = [r["alpha"] for r in lcurve_results]
            res_lc = [r["final_residual"] for r in lcurve_results]
            gnorm_lc = [r["gamma_norm"] for r in lcurve_results]
            gerr_lc = [r["rel_err_gamma_pct"] for r in lcurve_results]

            axes[0, 0].loglog(res_lc, gnorm_lc, "o-", color="#2b6cb0", linewidth=2)
            for a, rx, ry in zip(alphas_lc, res_lc, gnorm_lc):
                axes[0, 0].annotate(f"{a:.0e}", (rx, ry), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
            axes[0, 0].set_xlabel(r"Norma del residuo $\|b(\alpha)\|_2$")
            axes[0, 0].set_ylabel(r"Norma solución $\|\hat{\gamma}(\alpha)\|_2$")
            axes[0, 0].set_title("(a) Curva L de Regularización (Tikhonov bajo ruido)")
            axes[0, 0].grid(True, which="both", linestyle=":", alpha=0.6)

            axes[0, 1].semilogx(alphas_lc, gerr_lc, "s-", color="#c53030", linewidth=2)
            best_i = int(np.argmin(gerr_lc))
            axes[0, 1].scatter(
                [alphas_lc[best_i]], [gerr_lc[best_i]], color="#2f855a", s=85, zorder=5,
                label=f"Óptimo: $\\alpha$={alphas_lc[best_i]:.0e} ({gerr_lc[best_i]:.2f}%)"
            )
            axes[0, 1].set_xlabel(r"Parámetro de regularización $\alpha$")
            axes[0, 1].set_ylabel(r"Error relativo en $\gamma$ [%]")
            axes[0, 1].set_title("(b) Compromiso Sesgo-Varianza frente a $\\alpha$")
            axes[0, 1].grid(True, which="both", linestyle=":", alpha=0.6)
            axes[0, 1].legend(fontsize=8.5)
        else:
            axes[0, 0].axis("off")
            axes[0, 1].axis("off")

        if mesh_results:
            dens_mr = [r["mesh_density"] for r in mesh_results]
            d_labels = [f"{d}x{d}" for d in dens_mr]
            d_idx = np.arange(len(dens_mr))
            gammas_mr = np.array([r["estimated_gamma"] for r in mesh_results], dtype=float)
            times_mr = [r["elapsed_s"] for r in mesh_results]

            for m in range(4):
                axes[1, 0].plot(d_idx, gammas_mr[:, m], "o-", color=q_colors[m], linewidth=1.8, label=rf"$\hat{{\gamma}}_{m+1}$")
            axes[1, 0].set_xticks(d_idx)
            axes[1, 0].set_xticklabels(d_labels)
            axes[1, 0].set_xlabel("Densidad de malla FEM")
            axes[1, 0].set_ylabel(r"Conductividad estimada $\hat{\gamma}_m$")
            axes[1, 0].set_title("(c) Independencia de Malla en Conductividades")
            axes[1, 0].grid(True, linestyle=":", alpha=0.6)
            axes[1, 0].legend(fontsize=8, ncol=2)

            bars_t = axes[1, 1].bar(d_idx, times_mr, color="#6b46c1", edgecolor="#44337a")
            for b, tv in zip(bars_t, times_mr):
                axes[1, 1].text(b.get_x() + b.get_width() / 2, tv + max(times_mr) * 0.02, f"{tv:.2f}s",
                                ha="center", va="bottom", fontsize=8)
            axes[1, 1].set_xticks(d_idx)
            axes[1, 1].set_xticklabels(d_labels)
            axes[1, 1].set_xlabel("Densidad de malla FEM")
            axes[1, 1].set_ylabel("Tiempo de ejecución [s]")
            axes[1, 1].set_title("(d) Costo Computacional vs. Refinamiento de Malla")
            axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)
        else:
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")

        fig.suptitle("Figura 5: Análisis de Sensibilidad Paramétrica (Regularización Tikhonov y Refinamiento de Malla)", fontsize=13, fontweight="bold", y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        p5 = figures_dir / "var_05_regularizacion_y_malla.png"
        fig.savefig(p5, bbox_inches="tight", dpi=180)
        plt.close(fig)
        generated.append(p5)

    return generated
