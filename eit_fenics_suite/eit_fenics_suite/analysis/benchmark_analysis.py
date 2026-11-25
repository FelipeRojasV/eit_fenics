"""
Análisis computacional y generación de tablas/figuras (Capítulo 4).

Lee los resultados crudos de results/raw/*.npz generados por los ejercicios.
Si falta un crudo, ejecuta el ejercicio correspondiente (Opción A: importa
y llama run_exercise_K() en el mismo proceso).

Granularidad: --only table1 | table2 | table3 | mesh | gn | tsvd | all

Requiere FEniCS Legacy (vía eit_lib). Sin fallback.
"""

import argparse
import csv
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eit_lib import EITGeometry, EITForwardSolver, EITSensitivity, TwoChannelEIT
from eit_lib.results_io import load_raw, raw_exists, save_raw
from eit_lib import (
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

PAIRS = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]


# ---------------------------------------------------------------------------
# E/S de tablas
# ---------------------------------------------------------------------------

def save_table(name, headers, rows, caption="", label=""):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    md_path = os.path.join(RESULTS_DIR, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        if caption:
            f.write(f"### {caption}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")

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
            esc = [str(c).replace("_", "\\_").replace("%", "\\%") for c in row]
            f.write(" & ".join(esc) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"[Tabla] {os.path.relpath(md_path, ROOT_DIR)} (.csv/.md/.tex)")


def add_relative_noise(vec, rel_level, rng):
    if rel_level <= 0.0:
        return vec.copy()
    noise = rng.standard_normal(len(vec))
    noise = noise / np.linalg.norm(noise)
    return vec + rel_level * np.linalg.norm(vec) * noise


def complex_from_pair(arr):
    return complex(arr[0], arr[1])


# ---------------------------------------------------------------------------
# Carga tolerante de crudos (Opción A: importa y llama run_exercise_K)
# ---------------------------------------------------------------------------

def ensure_raw(name, runner_module, runner_func):
    """
    Si results/raw/<name>.npz no existe, importa el módulo del ejercicio
    y llama runner_func() en el mismo proceso. Luego carga el crudo.
    """
    if not raw_exists(name):
        print(f"[Crudo] {name}.npz ausente. Ejecutando {runner_module.__name__}...")
        runner_func()
    data = load_raw(name)
    if data is None:
        raise RuntimeError(f"No se pudo generar/leer results/raw/{name}.npz")
    return data


def load_ex1():
    from exercises.ex1_forward_inverse_1_1 import run_exercise_1
    return ensure_raw("ex1", sys.modules[run_exercise_1.__module__], run_exercise_1)


def load_ex2():
    from exercises.ex2_reciprocity import run_exercise_2
    return ensure_raw("ex2", sys.modules[run_exercise_2.__module__], run_exercise_2)


def load_ex3():
    from exercises.ex3_least_squares import run_exercise_3
    return ensure_raw("ex3", sys.modules[run_exercise_3.__module__], run_exercise_3)


def load_ex4():
    from exercises.ex4_linearized_svd import run_exercise_4
    return ensure_raw("ex4", sys.modules[run_exercise_4.__module__], run_exercise_4)


def load_ex5():
    from exercises.ex5_two_channel_eit import run_exercise_5
    return ensure_raw("ex5", sys.modules[run_exercise_5.__module__], run_exercise_5)


# ---------------------------------------------------------------------------
# Tabla 1: resumen de verificación (Ex1–Ex5)
# ---------------------------------------------------------------------------

def generate_table1_verification_summary():
    print("\n>>> Tabla 1: verificación Ex1–Ex5")
    rows = []

    d1 = load_ex1()
    res1 = np.linalg.norm(d1["A"] @ d1["delta_sigma"] - d1["delta_b"]) / np.linalg.norm(d1["delta_b"])
    rows.append([
        "Ex 1: Problema 1.1", "Malla 32x32, alpha=1e-3",
        "Residuo linealizado relativo", f"{res1:.4e}",
        f"sigma_recon = {np.round(d1['sigma_recon'], 3).tolist()}",
    ])

    d2 = load_ex2()
    max_asym = float(np.max(np.abs(d2["F"] - d2["F"].T)))
    frob_asym = float(np.linalg.norm(d2["F"] - d2["F"].T, ord="fro"))
    rows.append([
        "Ex 2: Reciprocidad", "Malla 36x36, sigma gaussiana",
        "max|F-F^T| / ||F-F^T||_F", f"{max_asym:.2e} / {frob_asym:.2e}",
        f"Simetria verificada (N_indep={int(d2['n_indep'])})",
    ])

    d3 = load_ex3()
    rel_err3 = np.linalg.norm(d3["sigma_recon"] - d3["sigma_true"]) / np.linalg.norm(d3["sigma_true"])
    rows.append([
        "Ex 3: Gauss-Newton", "Malla 32x32, alpha=1e-4, ruido=0.005",
        "Error relativo sigma", f"{rel_err3 * 100:.2f}%",
        f"Convergencia en {len(d3['history_res'])} iteraciones",
    ])

    d4 = load_ex4()
    rel_err4 = np.linalg.norm(d4["delta_recon"] - d4["delta_true"]) / np.linalg.norm(d4["delta_true"])
    rows.append([
        "Ex 4: TSVD", "Malla 32x32, r=4, ruido=0.002",
        "cond(A) / Error relativo", f"{float(d4['cond_A']):.2f} / {rel_err4 * 100:.2f}%",
        f"s_1={d4['S'][0]:.3e}, s_4={d4['S'][-1]:.3e}",
    ])

    d5 = load_ex5()
    err_sig5 = abs(float(d5["sigma_recon"]) - float(d5["sigma_true"]))
    err_eps5 = abs(float(d5["eps_recon"]) - float(d5["epsilon_true"]))
    rows.append([
        "Ex 5: Two-Channel", f"Malla 32x32, f={float(d5['freq_hz']) / 1e3:.0f} kHz",
        "Error abs. sigma / epsilon", f"{err_sig5:.2e} / {err_eps5:.2e}",
        f"C_Omega={float(d5['C_Omega']):.4f}, phi={float(d5['phase_deg']):.2f} deg",
    ])

    save_table(
        "tabla1_resumen_verificacion",
        ["Experimento", "Configuración", "Métrica clave", "Valor obtenido", "Observación"],
        rows,
        caption="Resumen de verificación numérica y errores de reconstrucción (Ejercicios 1 a 5).",
        label="tab:resumen_verificacion",
    )


# ---------------------------------------------------------------------------
# Tabla 2: matriz de tests (sin subprocess; API de pytest)
# ---------------------------------------------------------------------------

class _ResultCollector:
    """Plugin mínimo de pytest que captura resultados por test."""

    def __init__(self):
        self.results = []  # (nodeid, outcome)
        self.start = None
        self.elapsed = 0.0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.results.append((report.nodeid, report.outcome))
        elif report.when == "setup" and report.outcome == "skipped":
            self.results.append((report.nodeid, "skipped"))

    def pytest_sessionstart(self, session):
        self.start = time.perf_counter()

    def pytest_sessionfinish(self, session, exitstatus):
        if self.start is not None:
            self.elapsed = time.perf_counter() - self.start


def generate_table2_pytest_matrix():
    print("\n>>> Tabla 2: matriz de tests (pytest API)")
    import pytest

    tolerances = {
        "test_ntd_matrix_symmetric_for_homogeneous_sigma": "atol = 1e-8",
        "test_electrode_areas_consistency": "rtol = 1e-5 (np.allclose)",
        "test_extract_symmetrize_flag": "Igualdad exacta",
        "test_least_squares_recovers_noiseless": "atol = 1e-10",
        "test_svd_returns_correct_shapes": "Dimensiones exactas",
        "test_reciprocity_holds_for_inhomogeneous_sigma": "atol = 1e-6",
        "test_roundtrip_recovers_sigma_epsilon": "rtol = 1e-10",
    }

    collector = _ResultCollector()
    tests_dir = os.path.join(ROOT_DIR, "tests")
    exit_code = pytest.main(
        ["-q", "--tb=no", "-p", "no:cacheprovider", tests_dir],
        plugins=[collector],
    )

    rows = []
    for nodeid, outcome in collector.results:
        module_file, _, test_name = nodeid.partition("::")
        tol = tolerances.get(test_name, "Verificación estándar")
        rows.append([module_file, test_name, tol, outcome.upper()])

    if not rows:
        rows.append(["tests/", "Sin resultados", "-", f"exit {exit_code}"])

    rows.append([
        "Resumen suite", f"Total: {len(rows)} pruebas",
        f"Tiempo total: {collector.elapsed:.2f} s",
        "OK" if exit_code == 0 else f"CODE {int(exit_code)}",
    ])

    save_table(
        "tabla2_pruebas_unitarias",
        ["Archivo de prueba", "Caso de prueba", "Criterio / Tolerancia", "Estado"],
        rows,
        caption="Resultados de la verificación automatizada de la biblioteca mediante pytest.",
        label="tab:pruebas_unitarias",
    )


# ---------------------------------------------------------------------------
# Tabla 3 + Gráfico 1: convergencia de malla y costo computacional
# ---------------------------------------------------------------------------

def run_mesh_convergence_study():
    from fenics import FunctionSpace, Expression, interpolate

    print("\n>>> Tabla 3 + Gráfico 1: convergencia de malla")
    resolutions = [8, 16, 32, 64]
    rows, dofs_list, t_fwd_list, t_jac_list, asym_list, c_list = [], [], [], [], [], []

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
        dofs_list.append(solver.V.dim())
        t_fwd_list.append(t_fwd)
        t_jac_list.append(t_jac)
        asym_list.append(max(max_asym, 1e-16))
        c_list.append(c_omega)

        rows.append([
            f"{n} x {n}", f"{geom.mesh.num_cells()}", f"{solver.V.dim()}",
            f"{max_asym:.2e}", f"{c_omega:.5f}",
            f"{t_fwd:.4f}", f"{t_jac:.4f}", f"{t_setup + t_fwd + t_jac:.4f}",
        ])

    save_table(
        "tabla3_desempeno_malla",
        ["Malla", "Celdas", "DOFs CG(1)", "max|F-F^T|", "C_Omega",
         "t_forward [s]", "t_jacobiano [s]", "t_total [s]"],
        rows,
        caption="Desempeño computacional y convergencia geométrica en función de la malla.",
        label="tab:desempeno_malla",
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax_l = axes[0]
    ax_r = ax_l.twinx()
    l1, = ax_l.plot(resolutions, c_list, "o-", color="#1f77b4", lw=2, label=r"$C_\Omega$")
    l2, = ax_r.semilogy(resolutions, asym_list, "s--", color="#d62728", lw=2,
                        label=r"$\max|F-F^T|$")
    ax_l.set_xticks(resolutions)
    ax_l.set_xlabel("Resolución ($n_x=n_y$)")
    ax_l.set_ylabel(r"$C_\Omega$", color="#1f77b4")
    ax_r.set_ylabel(r"$\max|F-F^T|$", color="#d62728")
    ax_l.set_title("Estabilidad numérica y convergencia")
    ax_l.grid(True, linestyle="--", alpha=0.5)
    ax_l.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="center right")

    ax2 = axes[1]
    ax2.plot(dofs_list, t_fwd_list, "o-", color="#2ca02c", lw=2, label="Directo (3 iny.)")
    ax2.plot(dofs_list, t_jac_list, "s--", color="#ff7f0e", lw=2, label="Jacobiano A")
    ax2.plot(dofs_list, np.array(t_fwd_list) + np.array(t_jac_list), "^:",
             color="#9467bd", lw=2, label="Total")
    for d, n in zip(dofs_list, resolutions):
        ax2.annotate(f"{n}x{n}", (d, t_fwd_list[resolutions.index(n)] + t_jac_list[resolutions.index(n)]),
                     textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax2.set_xlabel("DOFs CG(1)")
    ax2.set_ylabel("Tiempo [s]")
    ax2.set_title("Costo computacional por resolución")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "analysis_mesh_convergence.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Gráfico] {os.path.relpath(out, ROOT_DIR)}")


# ---------------------------------------------------------------------------
# Gráfico 2: GN — regularización y ruido
# ---------------------------------------------------------------------------

def run_gn_regularization_noise_study():
    print("\n>>> Gráfico 2: GN regularización/ruido")
    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    sigma_true = np.array([2.5, 1.0, 1.0, 1.0])
    sigma_exact = reconstruct_quadrants(geom.mesh, geom.subdomains, sigma_true)
    F_true, _ = solver.compute_ntd_matrix(sigma_exact)
    b_true = extract_measurement_vector(F_true, PAIRS, symmetrize=False)

    alphas = np.logspace(-6, -1, 11)
    noise_levels = [0.001, 0.01, 0.05]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    rng = np.random.default_rng(123)

    noisy = {eta: add_relative_noise(b_true, eta, rng) for eta in noise_levels}
    err_curves = {eta: [] for eta in noise_levels}
    res_hist = {}

    for eta in noise_levels:
        b_meas = noisy[eta]
        for alpha in alphas:
            sigma_k = np.ones(4)
            hist = []
            for _ in range(6):
                sf = reconstruct_quadrants(geom.mesh, geom.subdomains, sigma_k)
                F_k, pots_k = solver.compute_ntd_matrix(sf)
                b_k = extract_measurement_vector(F_k, PAIRS, symmetrize=False)
                res = b_meas - b_k
                hist.append(float(np.linalg.norm(res)))
                A_k = sens.build_jacobian(pots_k, num_params=4)
                sigma_k = np.clip(sigma_k + solve_least_squares(A_k, res, alpha=alpha), 0.2, 5.0)
                if hist[-1] < 1e-4:
                    break
            err_curves[eta].append(float(np.linalg.norm(sigma_k - sigma_true) / np.linalg.norm(sigma_true) * 100))
            if np.isclose(alpha, 1e-4):
                res_hist[eta] = hist

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for eta, col in zip(noise_levels, colors):
        axes[0].semilogx(alphas, err_curves[eta], "o-", color=col, lw=2,
                         label=f"Ruido {eta * 100:.1f}%")
    axes[0].axvline(1e-4, color="gray", linestyle=":", lw=1.5, label=r"$\alpha=10^{-4}$")
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel("Error relativo [%]")
    axes[0].set_title("Sensibilidad a la regularización")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

    for eta, col in zip(noise_levels, colors):
        h = res_hist.get(eta, [])
        if h:
            axes[1].semilogy(range(1, len(h) + 1), h, "s-", color=col, lw=2,
                             label=f"Ruido {eta * 100:.1f}%")
    axes[1].set_xlabel("Iteración GN")
    axes[1].set_ylabel(r"$\|b_{meas}-b_k\|$")
    axes[1].set_title(r"Convergencia del residuo ($\alpha=10^{-4}$)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "analysis_gn_regularization_noise.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Gráfico] {os.path.relpath(out, ROOT_DIR)}")


# ---------------------------------------------------------------------------
# Gráfico 3: TSVD — truncamiento
# ---------------------------------------------------------------------------

def run_tsvd_truncation_study():
    print("\n>>> Gráfico 3: TSVD truncamiento")
    geom = EITGeometry(nx=32, ny=32)
    solver = EITForwardSolver(geom, current_intensity=1.0)
    sens = EITSensitivity(geom, current_intensity=1.0)

    from fenics import Function, FunctionSpace
    V_dg = FunctionSpace(geom.mesh, "DG", 0)
    sigma0 = Function(V_dg)
    sigma0.vector()[:] = 1.0
    _, pots0 = solver.compute_ntd_matrix(sigma0)
    A = sens.build_jacobian(pots0, num_params=4)

    delta_true = np.array([0.7, -0.3, 0.1, 0.4])
    delta_b_exact = A @ delta_true

    ranks = [1, 2, 3, 4]
    noise_levels = [0.005, 0.02, 0.08]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    n_trials = 50
    rng = np.random.default_rng(2026)

    mean_err = {eta: [] for eta in noise_levels}
    std_err = {eta: [] for eta in noise_levels}
    tik_ref = {}

    for eta in noise_levels:
        trials = [add_relative_noise(delta_b_exact, eta, rng) for _ in range(n_trials)]
        for r in ranks:
            errs = []
            for db in trials:
                d_rec, _, _, _ = solve_svd_linearized(A, db, n_singular_values=r)
                errs.append(float(np.linalg.norm(d_rec - delta_true) / np.linalg.norm(delta_true) * 100))
            mean_err[eta].append(float(np.mean(errs)))
            std_err[eta].append(float(np.std(errs)))
        tik = []
        for db in trials:
            d_tik = solve_least_squares(A, db, alpha=1e-3)
            tik.append(float(np.linalg.norm(d_tik - delta_true) / np.linalg.norm(delta_true) * 100))
        tik_ref[eta] = float(np.mean(tik))

    _, S, _, _ = solve_svd_linearized(A, delta_b_exact, n_singular_values=4)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for eta, col in zip(noise_levels, colors):
        axes[0].errorbar(ranks, mean_err[eta], yerr=std_err[eta], fmt="o-",
                         color=col, lw=2, capsize=4, label=f"TSVD (ruido {eta * 100:.1f}%)")
        axes[0].axhline(tik_ref[eta], color=col, linestyle=":", lw=1.3, alpha=0.8)
    axes[0].set_xticks(ranks)
    axes[0].set_xlabel("Rango de truncamiento $r$")
    axes[0].set_ylabel("Error relativo medio [%]")
    axes[0].set_title("Sesgo-varianza en TSVD (50 realizaciones)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    idx = np.arange(1, len(S) + 1)
    w_tsvd = np.array([1.0 if i <= 3 else 0.0 for i in idx])
    w_tik_3 = (S ** 2) / (S ** 2 + 1e-3)
    w_tik_2 = (S ** 2) / (S ** 2 + 1e-2)
    axes[1].plot(idx, w_tsvd, "s--", color="#d62728", lw=2, label="TSVD ($r=3$)")
    axes[1].plot(idx, w_tik_3, "o-", color="#1f77b4", lw=2, label=r"Tikhonov ($\alpha=10^{-3}$)")
    axes[1].plot(idx, w_tik_2, "^-", color="#2ca02c", lw=2, label=r"Tikhonov ($\alpha=10^{-2}$)")
    axes[1].set_xticks(idx)
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].set_xlabel("Modo singular $i$")
    axes[1].set_ylabel("Factor de filtro $w_i$")
    axes[1].set_title("Factores de filtro espectral")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "analysis_tsvd_truncation.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Gráfico] {os.path.relpath(out, ROOT_DIR)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SECTIONS = {
    "table1": generate_table1_verification_summary,
    "table2": generate_table2_pytest_matrix,
    "table3": run_mesh_convergence_study,
    "mesh": run_mesh_convergence_study,
    "gn": run_gn_regularization_noise_study,
    "tsvd": run_tsvd_truncation_study,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark y análisis EIT")
    parser.add_argument("--only", default="all",
                        help="table1|table2|table3|mesh|gn|tsvd|all")
    args = parser.parse_args()

    if args.only == "all":
        targets = ["table1", "table2", "table3", "gn", "tsvd"]
    elif args.only in SECTIONS:
        targets = [args.only]
    else:
        parser.error(f"--only inválido: {args.only}")

    for t in targets:
        SECTIONS[t]()

    print("\n[OK] Análisis completado. Revisa results/ y plots/.")


if __name__ == "__main__":
    main()