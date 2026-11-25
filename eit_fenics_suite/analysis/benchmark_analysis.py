"""
exercises/ex6_benchmark_analysis.py
Módulo de análisis computacional, verificación y estudios paramétricos
para la suite EIT FEniCS Legacy.

Genera automáticamente:
  - 3 tablas en results/ (.md, .csv y .tex):
      1. Resumen de verificación numérica y errores (Ex1–Ex5)
      2. Matriz de pruebas unitarias automatizadas (pytest)
      3. Desempeño computacional y convergencia por resolución de malla
  - 3 gráficos en plots/ (.png, 300 dpi):
      1. analysis_mesh_convergence.png
      2. analysis_gn_regularization_noise.png
      3. analysis_tsvd_truncation.png

Requiere FEniCS Legacy. Sin fallback analítico.
"""

import csv
import os
import re
import subprocess
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fenics import Function, FunctionSpace, Expression, interpolate  # noqa: E402
try:
    from fenics import set_log_level, LogLevel
    set_log_level(LogLevel.ERROR)
except Exception:
    pass

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eit_lib import (  # noqa: E402
    EITGeometry,
    EITForwardSolver,
    EITSensitivity,
    TwoChannelEIT,
    extract_measurement_vector,
    reconstruct_quadrants,
    solve_least_squares,
    solve_svd_linearized,
)

PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Matriz experimental del Problema 1.1 (Ammari, pág. 16)
F_MEAS_TABLE = np.array([
    [-2.0285, -1.3025, -1.0962],
    [-1.3068, -2.3413, -1.3633],
    [-1.1053, -1.3724, -2.5987],
])


# =============================================================================
# UTILIDADES DE EXPORTACIÓN DE TABLAS (CSV, MARKDOWN, LATEX)
# =============================================================================

def save_table(name, headers, rows, caption="", label=""):
    """Guarda una tabla en formatos .csv, .md y .tex dentro de results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. CSV
    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # 2. Markdown
    md_path = os.path.join(RESULTS_DIR, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        if caption:
            f.write(f"### {caption}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")

    # 3. LaTeX
    tex_path = os.path.join(RESULTS_DIR, f"{name}.tex")
    col_spec = "l" + "c" * (len(headers) - 1)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n")
        f.write(" & ".join(headers) + " \\\\\n\\hline\n")
        for row in rows:
            escaped = [str(c).replace("_", "\\_").replace("%", "\\%") for c in row]
            f.write(" & ".join(escaped) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"[Tabla guardada] {os.path.relpath(md_path, ROOT_DIR)} (.csv / .md / .tex)")


def add_relative_noise(vec, rel_level, rng):
    """Agrega ruido gaussiano normalizado tal que ||e||_2 / ||vec||_2 = rel_level."""
    if rel_level <= 0.0:
        return vec.copy()
    noise = rng.standard_normal(len(vec))
    noise = noise / np.linalg.norm(noise)
    return vec + rel_level * np.linalg.norm(vec) * noise


# =============================================================================
# TABLA 1: RESUMEN DE VERIFICACIÓN NUMÉRICA Y ERRORES (EX1 - EX5)
# =============================================================================

def generate_table1_verification_summary():
    print("\n>>> Generando Tabla 1: Resumen de verificación numérica (Ex1–Ex5)...")
    rows = []

    # --- Ex 1: Problema 1.1 ---
    geom1 = EITGeometry(nx=32, ny=32)
    solver1 = EITForwardSolver(geom1, current_intensity=1.0)
    sens1 = EITSensitivity(geom1, current_intensity=1.0)
    V_dg1 = FunctionSpace(geom1.mesh, "DG", 0)
    sigma0 = Function(V_dg1)
    sigma0.vector()[:] = 1.0
    F_sim1, pots1 = solver1.compute_ntd_matrix(sigma0)
    A1 = sens1.build_jacobian(pots1, num_params=4)
    b_meas1 = extract_measurement_vector(F_MEAS_TABLE, sens1.pairs, symmetrize=True)
    b_sim1 = extract_measurement_vector(F_sim1, sens1.pairs, symmetrize=False)
    delta_b1 = b_meas1 - b_sim1
    delta_sigma1 = solve_least_squares(A1, delta_b1, alpha=1e-3)
    res_rel1 = np.linalg.norm(A1 @ delta_sigma1 - delta_b1) / np.linalg.norm(delta_b1)
    rows.append([
        "Ex 1: Problema 1.1",
        "Malla 32x32, alpha=1e-3",
        "Residuo linealizado relativo",
        f"{res_rel1:.4e}",
        f"sigma_recon = {np.round(1.0 + delta_sigma1, 3).tolist()}",
    ])

    # --- Ex 2: Reciprocidad ---
    geom2 = EITGeometry(nx=36, ny=36)
    solver2 = EITForwardSolver(geom2, current_intensity=1.0)
    V_cg2 = FunctionSpace(geom2.mesh, "Lagrange", 1)
    sigma_expr = Expression(
        "1.0 + 0.9 * exp(-4.0*(pow(x[0]-0.2, 2) + pow(x[1]-0.1, 2)))",
        degree=2,
    )
    sigma2 = interpolate(sigma_expr, V_cg2)
    F2, _ = solver2.compute_ntd_matrix(sigma2)
    max_asym2 = float(np.max(np.abs(F2 - F2.T)))
    frob_asym2 = float(np.linalg.norm(F2 - F2.T, ord="fro"))
    rows.append([
        "Ex 2: Reciprocidad",
        "Malla 36x36, sigma(x,y) gaussiana",
        "max|F - F^T| / ||F - F^T||_F",
        f"{max_asym2:.2e} / {frob_asym2:.2e}",
        "Simetria FEM verificada (N_indep = 6)",
    ])

    # --- Ex 3: Gauss-Newton ---
    sigma_true3 = np.array([2.5, 1.0, 1.0, 1.0])
    sigma_exact3 = reconstruct_quadrants(geom1.mesh, geom1.subdomains, sigma_true3)
    F_true3, _ = solver1.compute_ntd_matrix(sigma_exact3)
    b_true3 = extract_measurement_vector(F_true3, sens1.pairs, symmetrize=False)
    rng3 = np.random.default_rng(42)
    b_meas3 = b_true3 + 0.005 * rng3.standard_normal(len(b_true3))

    sigma_k = np.ones(4)
    it_used = 0
    for it in range(6):
        it_used = it + 1
        sf = reconstruct_quadrants(geom1.mesh, geom1.subdomains, sigma_k)
        F_k, pots_k = solver1.compute_ntd_matrix(sf)
        b_k = extract_measurement_vector(F_k, sens1.pairs, symmetrize=False)
        res = b_meas3 - b_k
        A_k = sens1.build_jacobian(pots_k, num_params=4)
        sigma_k = np.clip(sigma_k + solve_least_squares(A_k, res, alpha=1e-4), 0.2, 5.0)
        if np.linalg.norm(res) < 1e-4:
            break
    rel_err3 = np.linalg.norm(sigma_k - sigma_true3) / np.linalg.norm(sigma_true3)
    rows.append([
        "Ex 3: Gauss-Newton",
        "Malla 32x32, alpha=1e-4, ruido=0.005",
        "Error relativo ||sigma_k - sigma*||/||sigma*||",
        f"{rel_err3 * 100:.2f}%",
        f"Convergencia en {it_used} iteraciones",
    ])

    # --- Ex 4: TSVD ---
    delta_true4 = np.array([0.7, -0.3, 0.1, 0.4])
    rng4 = np.random.default_rng(10)
    delta_b4 = A1 @ delta_true4 + 0.002 * rng4.standard_normal(6)
    delta_rec4, S4, _, _ = solve_svd_linearized(A1, delta_b4, n_singular_values=4)
    cond_A4 = S4[0] / S4[-1]
    rel_err4 = np.linalg.norm(delta_rec4 - delta_true4) / np.linalg.norm(delta_true4)
    rows.append([
        "Ex 4: TSVD",
        "Malla 32x32, r=4, ruido=0.002",
        "cond(A) / Error relativo en delta_sigma",
        f"{cond_A4:.2f} / {rel_err4 * 100:.2f}%",
        f"s_1={S4[0]:.3e}, s_4={S4[-1]:.3e}",
    ])

    # --- Ex 5: Two-Channel EIT ---
    sigma_true5 = 0.25
    eps_true5 = 5000.0 * 8.854e-12
    omega5 = 2.0 * np.pi * 50e3
    two_ch = TwoChannelEIT(nx=32, ny=32, electrode_width=0.4, current_intensity=1.0)
    V_meas5, _ = two_ch.forward_solve(sigma_true5, eps_true5, omega5)
    sigma_rec5, eps_rec5, _ = two_ch.inverse_solve(V_meas5, omega5)
    err_sig5 = abs(sigma_rec5 - sigma_true5)
    err_eps5 = abs(eps_rec5 - eps_true5)
    phase5 = np.angle(V_meas5, deg=True)
    rows.append([
        "Ex 5: Two-Channel",
        "Malla 32x32, f=50 kHz",
        "Error abs. sigma / Error abs. epsilon",
        f"{err_sig5:.2e} / {err_eps5:.2e}",
        f"C_Omega={two_ch.energy_u0:.4f}, phi={phase5:.2f} deg",
    ])

    headers = ["Experimento", "Configuración", "Métrica clave", "Valor obtenido", "Observación"]
    save_table(
        "tabla1_resumen_verificacion",
        headers,
        rows,
        caption="Resumen de verificación numérica y errores de reconstrucción (Ejercicios 1 a 5).",
        label="tab:resumen_verificacion",
    )


# =============================================================================
# TABLA 2: MATRIZ DE PRUEBAS UNITARIAS (PYTEST)
# =============================================================================

def generate_table2_pytest_matrix():
    print("\n>>> Generando Tabla 2: Matriz de pruebas unitarias (pytest)...")
    tolerances_map = {
        "test_ntd_matrix_symmetric_for_homogeneous_sigma": "atol = 1e-8",
        "test_electrode_areas_consistency": "rtol = 1e-5 (np.allclose)",
        "test_extract_symmetrize_flag": "Igualdad exacta",
        "test_least_squares_recovers_noiseless": "atol = 1e-10",
        "test_svd_returns_correct_shapes": "Dimensiones exactas",
        "test_reciprocity_holds_for_inhomogeneous_sigma": "atol = 1e-6",
        "test_roundtrip_recovers_sigma_epsilon": "rtol = 1e-10",
    }

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=no", "tests"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    elapsed_total = time.perf_counter() - t0

    rows = []
    pattern = re.compile(r"^(tests/\S+)::(\S+)\s+(PASSED|FAILED|SKIPPED)")
    for line in proc.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            module_file, test_name, status = match.groups()
            tol = tolerances_map.get(test_name, "Verificación estándar")
            rows.append([module_file, test_name, tol, status])

    if not rows:
        rows.append(["tests/", "Sin resultados parseados", "-", f"Exit code {proc.returncode}"])

    rows.append([
        "Resumen suite",
        f"Total: {len(rows)} pruebas",
        f"Tiempo total: {elapsed_total:.2f} s",
        "OK" if proc.returncode == 0 else f"CODE {proc.returncode}",
    ])

    headers = ["Archivo de prueba", "Caso de prueba", "Criterio / Tolerancia", "Estado"]
    save_table(
        "tabla2_pruebas_unitarias",
        headers,
        rows,
        caption="Resultados de la verificación automatizada de la biblioteca mediante pytest.",
        label="tab:pruebas_unitarias",
    )


# =============================================================================
# TABLA 3 Y GRÁFICO 1: CONVERGENCIA DE MALLA Y COSTO COMPUTACIONAL
# =============================================================================

def run_mesh_convergence_study():
    print("\n>>> Ejecutando estudio de convergencia de malla y costo computacional...")
    resolutions = [ 16, 32, 64, 128]
    rows = []

    dofs_list = []
    times_fwd = []
    times_jac = []
    asym_list = []
    c_omega_list = []

    sigma_expr = Expression(
        "1.0 + 0.9 * exp(-4.0*(pow(x[0]-0.2, 2) + pow(x[1]-0.1, 2)))",
        degree=2,
    )

    for n in resolutions:
        t0 = time.perf_counter()
        geom = EITGeometry(nx=n, ny=n)
        solver = EITForwardSolver(geom, current_intensity=1.0)
        sens = EITSensitivity(geom, current_intensity=1.0)
        t_setup = time.perf_counter() - t0

        V_cg = FunctionSpace(geom.mesh, "Lagrange", 1)
        sigma_het = interpolate(sigma_expr, V_cg)

        t1 = time.perf_counter()
        F, pots = solver.compute_ntd_matrix(sigma_het)
        t_fwd = time.perf_counter() - t1

        t2 = time.perf_counter()
        _ = sens.build_jacobian(pots, num_params=4)
        t_jac = time.perf_counter() - t2

        two_ch = TwoChannelEIT(nx=n, ny=n, electrode_width=0.4, current_intensity=1.0)
        c_omega = float(two_ch.energy_u0)

        max_asym = float(np.max(np.abs(F - F.T)))
        n_cells = geom.mesh.num_cells()
        n_dofs = solver.V.dim()
        t_total = t_setup + t_fwd + t_jac

        dofs_list.append(n_dofs)
        times_fwd.append(t_fwd)
        times_jac.append(t_jac)
        asym_list.append(max(max_asym, 1e-16))
        c_omega_list.append(c_omega)

        rows.append([
            f"{n} x {n}",
            f"{n_cells}",
            f"{n_dofs}",
            f"{max_asym:.2e}",
            f"{c_omega:.5f}",
            f"{t_fwd:.4f}",
            f"{t_jac:.4f}",
            f"{t_total:.4f}",
        ])

    headers = [
        "Malla (nx x ny)",
        "Celdas",
        "DOFs CG(1)",
        "max|F - F^T|",
        "C_Omega",
        "t_forward [s]",
        "t_jacobiano [s]",
        "t_total [s]",
    ]
    save_table(
        "tabla3_desempeno_malla",
        headers,
        rows,
        caption="Desempeño computacional y convergencia geométrica en función de la resolución de malla.",
        label="tab:desempeno_malla",
    )

    # --- Gráfico 1: Convergencia de malla y costo computacional ---
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel izquierdo: Convergencia de C_Omega y error de reciprocidad
    ax_l = axes[0]
    ax_r = ax_l.twinx()
    l1, = ax_l.plot(resolutions, c_omega_list, "o-", color="#1f77b4", lw=2,
                    label=r"Factor geométrico $C_\Omega$")
    l2, = ax_r.semilogy(resolutions, asym_list, "s--", color="#d62728", lw=2,
                        label=r"Asimetría $\max|F - F^T|$")
    ax_l.set_xticks(resolutions)
    ax_l.set_xlabel("Resolución de malla ($n_x = n_y$)")
    ax_l.set_ylabel(r"$C_\Omega = \int_\Omega |\nabla u_0|^2\,dx$", color="#1f77b4")
    ax_r.set_ylabel(r"Error de reciprocidad $\max|F - F^T|$", color="#d62728")
    ax_l.set_title("Estabilidad Numérica y Convergencia de Malla")
    ax_l.grid(True, linestyle="--", alpha=0.5)
    ax_l.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="center right")

    # Panel derecho: Costo computacional vs DOFs
    ax2 = axes[1]
    ax2.plot(dofs_list, times_fwd, "o-", color="#2ca02c", lw=2, label="Problema directo (3 inyecciones)")
    ax2.plot(dofs_list, times_jac, "s--", color="#ff7f0e", lw=2, label="Ensamblaje Jacobiano $A$")
    ax2.plot(dofs_list, np.array(times_fwd) + np.array(times_jac), "^:", color="#9467bd", lw=2,
             label="Total (Directo + Jacobiano)")
    for d, n, tf in zip(dofs_list, resolutions, np.array(times_fwd) + np.array(times_jac)):
        ax2.annotate(f"{n}x{n}", (d, tf), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8)
    ax2.set_xlabel("Grados de libertad (DOFs en $CG_1$)")
    ax2.set_ylabel("Tiempo de ejecución [s]")
    ax2.set_title("Costo Computacional por Resolución de Malla")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "analysis_mesh_convergence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Gráfico guardado] {os.path.relpath(plot_path, ROOT_DIR)}")


# =============================================================================
# GRÁFICO 2: ESTABILIDAD DE GAUSS-NEWTON FRENTE A RUIDO Y REGULARIZACIÓN
# =============================================================================

def run_gn_regularization_noise_study():
    print("\n>>> Ejecutando análisis de regularización y ruido para Gauss-Newton...")
    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    sigma_true = np.array([2.5, 1.0, 1.0, 1.0])
    sigma_exact = reconstruct_quadrants(geom.mesh, geom.subdomains, sigma_true)
    F_true, _ = solver.compute_ntd_matrix(sigma_exact)
    b_true = extract_measurement_vector(F_true, sens.pairs, symmetrize=False)

    alphas = np.logspace(-6, -1, 11)
    noise_levels = [0.001, 0.01, 0.05]  # 0.1%, 1%, 5%
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    rng = np.random.default_rng(123)

    # Pre-generar un vector de medición ruidoso por nivel para comparación consistente
    noisy_datasets = {
        eta: add_relative_noise(b_true, eta, rng) for eta in noise_levels
    }

    err_curves = {eta: [] for eta in noise_levels}
    res_histories = {}

    for eta in noise_levels:
        b_meas = noisy_datasets[eta]
        for alpha in alphas:
            sigma_k = np.ones(4)
            hist = []
            for _ in range(6):
                sf = reconstruct_quadrants(geom.mesh, geom.subdomains, sigma_k)
                F_k, pots_k = solver.compute_ntd_matrix(sf)
                b_k = extract_measurement_vector(F_k, sens.pairs, symmetrize=False)
                res = b_meas - b_k
                hist.append(np.linalg.norm(res))
                A_k = sens.build_jacobian(pots_k, num_params=4)
                sigma_k = np.clip(sigma_k + solve_least_squares(A_k, res, alpha=alpha), 0.2, 5.0)
                if hist[-1] < 1e-4:
                    break
            rel_err = np.linalg.norm(sigma_k - sigma_true) / np.linalg.norm(sigma_true)
            err_curves[eta].append(rel_err * 100.0)
            if np.isclose(alpha, 1e-4):
                res_histories[eta] = hist

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel izquierdo: Error relativo vs alpha
    for eta, col in zip(noise_levels, colors):
        axes[0].semilogx(alphas, err_curves[eta], "o-", color=col, lw=2,
                         label=f"Ruido relativo {eta * 100:.1f}%")
    axes[0].axvline(1e-4, color="gray", linestyle=":", lw=1.5, label=r"$\alpha = 10^{-4}$ (Ex 3)")
    axes[0].set_xlabel(r"Parámetro de regularización de Tikhonov $\alpha$")
    axes[0].set_ylabel(r"Error relativo $\|\sigma_{\mathrm{rec}} - \sigma_{\mathrm{true}}\|_2 / \|\sigma_{\mathrm{true}}\|_2$ [%]")
    axes[0].set_title("Sensibilidad de Gauss-Newton a la Regularización")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

    # Panel derecho: Convergencia del residuo para alpha=1e-4
    for eta, col in zip(noise_levels, colors):
        hist = res_histories[eta]
        axes[1].semilogy(range(1, len(hist) + 1), hist, "s-", color=col, lw=2,
                         label=f"Ruido {eta * 100:.1f}%")
    axes[1].set_xlabel("Iteración Gauss-Newton")
    axes[1].set_ylabel(r"Norma del residuo $\|b_{\mathrm{meas}} - b_k\|_2$")
    axes[1].set_title(r"Convergencia del Residuo ($\alpha = 10^{-4}$)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "analysis_gn_regularization_noise.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Gráfico guardado] {os.path.relpath(plot_path, ROOT_DIR)}")


# =============================================================================
# GRÁFICO 3: COMPROMISO SESGO-VARIANZA EN TSVD (TRUNCAMIENTO r = 1..4)
# =============================================================================

def run_tsvd_truncation_study():
    print("\n>>> Ejecutando análisis de truncamiento y sesgo-varianza en TSVD...")
    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    V_dg = FunctionSpace(geom.mesh, "DG", 0)
    sigma0 = Function(V_dg)
    sigma0.vector()[:] = 1.0
    _, pots0 = solver.compute_ntd_matrix(sigma0)
    A = sens.build_jacobian(pots0, num_params=4)

    delta_true = np.array([0.7, -0.3, 0.1, 0.4])
    delta_b_exact = A @ delta_true

    ranks = [1, 2, 3, 4]
    noise_levels = [0.005, 0.02, 0.08]  # 0.5%, 2%, 8%
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    n_trials = 50
    rng = np.random.default_rng(2026)

    mean_errors = {eta: [] for eta in noise_levels}
    std_errors = {eta: [] for eta in noise_levels}
    tikhonov_ref = {}

    for eta in noise_levels:
        trial_noisy = [add_relative_noise(delta_b_exact, eta, rng) for _ in range(n_trials)]
        for r in ranks:
            errs = []
            for db in trial_noisy:
                d_rec, _, _, _ = solve_svd_linearized(A, db, n_singular_values=r)
                errs.append(np.linalg.norm(d_rec - delta_true) / np.linalg.norm(delta_true) * 100.0)
            mean_errors[eta].append(np.mean(errs))
            std_errors[eta].append(np.std(errs))

        # Referencia con Tikhonov (alpha=1e-3) bajo el mismo nivel de ruido
        tik_errs = []
        for db in trial_noisy:
            d_tik = solve_least_squares(A, db, alpha=1e-3)
            tik_errs.append(np.linalg.norm(d_tik - delta_true) / np.linalg.norm(delta_true) * 100.0)
        tikhonov_ref[eta] = np.mean(tik_errs)

    _, S, _, _ = solve_svd_linearized(A, delta_b_exact, n_singular_values=4)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel izquierdo: Error medio vs rango de truncamiento r (con barras de desviación)
    for eta, col in zip(noise_levels, colors):
        axes[0].errorbar(
            ranks, mean_errors[eta], yerr=std_errors[eta],
            fmt="o-", color=col, lw=2, capsize=4,
            label=f"TSVD (ruido {eta * 100:.1f}%)"
        )
        axes[0].axhline(tikhonov_ref[eta], color=col, linestyle=":", lw=1.3, alpha=0.8)
    axes[0].set_xticks(ranks)
    axes[0].set_xlabel("Rango de truncamiento $r$ (valores singulares retenidos)")
    axes[0].set_ylabel(r"Error relativo medio en $\delta\sigma$ [%]")
    axes[0].set_title("Compromiso Sesgo-Varianza en TSVD (50 realizaciones)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel derecho: Factores de filtro espectral w_i (TSVD r=3 vs Tikhonov)
    idx = np.arange(1, len(S) + 1)
    w_tsvd_r3 = np.array([1.0 if i <= 3 else 0.0 for i in idx])
    w_tik_1e3 = (S ** 2) / (S ** 2 + 1e-3)
    w_tik_1e2 = (S ** 2) / (S ** 2 + 1e-2)

    axes[1].plot(idx, w_tsvd_r3, "s--", color="#d62728", lw=2, label="TSVD ($r = 3$)")
    axes[1].plot(idx, w_tik_1e3, "o-", color="#1f77b4", lw=2, label=r"Tikhonov ($\alpha = 10^{-3}$)")
    axes[1].plot(idx, w_tik_1e2, "^-", color="#2ca02c", lw=2, label=r"Tikhonov ($\alpha = 10^{-2}$)")
    axes[1].set_xticks(idx)
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].set_xlabel("Índice del modo singular $i$")
    axes[1].set_ylabel(r"Factor de filtro espectral $w_i$")
    axes[1].set_title("Comparación de Factores de Filtro Espectral")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "analysis_tsvd_truncation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Gráfico guardado] {os.path.relpath(plot_path, ROOT_DIR)}")


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    print("=" * 70)
    print(" ANÁLISIS COMPUTACIONAL Y GENERACIÓN DE TABLAS/FIGURAS (CAPÍTULO 4)")
    print("=" * 70)

    generate_table1_verification_summary()
    generate_table2_pytest_matrix()
    run_mesh_convergence_study()
    run_gn_regularization_noise_study()
    run_tsvd_truncation_study()

    print("\n" + "=" * 70)
    print(" FINALIZADO CON ÉXITO.")
    print("  - Tablas (.csv, .md, .tex) en: results/")
    print("  - Figuras (.png, 300 dpi)  en: plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
