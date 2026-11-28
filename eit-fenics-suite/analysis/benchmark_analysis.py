"""
Análisis computacional y generación de tablas/figuras (Ex1–Ex4).

Lee results/raw/*.npz. Si falta un crudo, ejecuta el ejercicio correspondiente.

Granularidad: --only table1 | table2 | table3 | gn | tsvd | all
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

from eit_lib import (
    EITGeometry, EITForwardSolver, EITSensitivity,
    extract_measurement_vector, reconstruct_quadrants,
    solve_least_squares, solve_svd_linearized,
)
from eit_lib.results_io import load_raw, raw_exists

try:
    from fenics import set_log_level
    set_log_level(40)  # 40 = ERROR
except Exception:
    pass

PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

PAIRS = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]


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

    print(f"[Tabla] {os.path.relpath(md_path, ROOT_DIR)}")


def add_relative_noise(vec, rel_level, rng):
    if rel_level <= 0.0:
        return vec.copy()
    noise = rng.standard_normal(len(vec))
    noise /= np.linalg.norm(noise)
    return vec + rel_level * np.linalg.norm(vec) * noise


def ensure_raw(name, runner_module, runner_func):
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


def generate_table1_verification_summary():
    print("\n>>> Tabla 1: verificación Ex1–Ex4")
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
        f"{len(d3['history_res'])} iteraciones ejecutadas",
    ])

    d4 = load_ex4()
    rel_err4 = np.linalg.norm(d4["delta_recon"] - d4["delta_true"]) / np.linalg.norm(d4["delta_true"])
    rows.append([
        "Ex 4: TSVD", "Malla 32x32, r=4, ruido=0.002",
        "cond(A) / Error relativo", f"{float(d4['cond_A']):.2f} / {rel_err4 * 100:.2f}%",
        f"s_1={d4['S'][0]:.3e}, s_4={d4['S'][-1]:.3e}",
    ])

    save_table(
        "tabla1_resumen_verificacion",
        ["Experimento", "Configuración", "Métrica clave", "Valor obtenido", "Observación"],
        rows,
        caption="Resumen de verificación numérica y errores de reconstrucción (Ejercicios 1 a 4).",
        label="tab:resumen_verificacion",
    )


class _ResultCollector:
    def __init__(self):
        self.results = []
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
    print("\n>>> Tabla 2: matriz de tests")
    import pytest

    tolerances = {
        "test_ntd_matrix_symmetric_for_homogeneous_sigma": "atol = 1e-8",
        "test_electrode_areas_consistency": "rtol = 1e-5 (np.allclose)",
        "test_extract_symmetrize_flag": "Igualdad exacta",
        "test_least_squares_recovers_noiseless": "atol = 1e-10",
        "test_svd_returns_correct_shapes": "Dimensiones exactas",
        "test_reciprocity_holds_for_inhomogeneous_sigma": "atol = 1e-6",
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
        caption="Resultados de la verificación automatizada mediante pytest.",
        label="tab:pruebas_unitarias",
    )


def run_mesh_convergence_study():
    from fenics import FunctionSpace, Expression, interpolate

    print("\n>>> Tabla 3 + Gráfico 1: convergencia de malla")
    resolutions = [16, 32, 64, 128]

    rows, dofs_list, t_fwd_list, t_jac_list, asym_list = [], [], [], [], []

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

        max_asym = float(np.max(np.abs(F - F.T)))
        dofs_list.append(solver.V.dim())
        t_fwd_list.append(t_fwd)
        t_jac_list.append(t_jac)
        asym_list.append(max(max_asym, 1e-16))

        rows.append([
            f"{n} x {n}", f"{geom.mesh.num_cells()}", f"{solver.V.dim()}",
            f"{max_asym:.2e}",
            f"{t_fwd:.4f}", f"{t_jac:.4f}", f"{t_setup + t_fwd + t_jac:.4f}",
        ])

    save_table(
        "tabla3_desempeno_malla",
        ["Malla", "Celdas", "DOFs CG(1)", "max|F-F^T|",
         "t_forward [s]", "t_jacobiano [s]", "t_total [s]"],
        rows,
        caption="Desempeño computacional y convergencia geométrica en función de la malla.",
        label="tab:desempeno_malla",
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax_l = axes[0]
    ax_l.semilogy(resolutions, asym_list, "s--", color="#d62728", lw=2,
                  label=r"$\max|F-F^T|$")
    ax_l.set_xticks(resolutions)
    ax_l.set_xlabel("Resolución ($n_x=n_y$)")
    ax_l.set_ylabel(r"$\max|F-F^T|$")
    ax_l.set_title("Error de reciprocidad vs resolución")
    ax_l.grid(True, linestyle="--", alpha=0.5)
    ax_l.legend()

    ax2 = axes[1]
    ax2.plot(dofs_list, t_fwd_list, "o-", color="#2ca02c", lw=2, label="Forward")
    ax2.plot(dofs_list, t_jac_list, "s--", color="#ff7f0e", lw=2, label="Jacobiano")
    ax2.plot(dofs_list, np.array(t_fwd_list) + np.array(t_jac_list), "^:",
             color="#9467bd", lw=2, label="Total")
    for d, n, tf, tj in zip(dofs_list, resolutions, t_fwd_list, t_jac_list):
        ax2.annotate(f"{n}x{n}", (d, tf + tj), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8)
    ax2.set_xlabel("DOFs CG(1)")
    ax2.set_ylabel("Tiempo [s]")
    ax2.set_title("Costo computacional")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "analysis_mesh_convergence.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Gráfico] {os.path.relpath(out, ROOT_DIR)}")


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
            err_curves[eta].append(
                float(np.linalg.norm(sigma_k - sigma_true) / np.linalg.norm(sigma_true) * 100)
            )
            if np.isclose(alpha, 1e-4):
                res_hist[eta] = hist

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for eta, col in zip(noise_levels, colors):
        axes[0].semilogx(alphas, err_curves[eta], "o-", color=col, lw=2,
                         label=f"Ruido {eta * 100:.1f}%")
    axes[0].axvline(1e-4, color="gray", linestyle=":", lw=1.5)
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel("Error relativo [%]")
    axes[0].set_title("Sensibilidad a regularización")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

    for eta, col in zip(noise_levels, colors):
        h = res_hist.get(eta, [])
        if h:
            axes[1].semilogy(range(1, len(h) + 1), h, "s-", color=col, lw=2,
                             label=f"Ruido {eta * 100:.1f}%")
    axes[1].set_xlabel("Iteración GN")
    axes[1].set_ylabel("Residuo")
    axes[1].set_title(r"Convergencia ($\alpha=10^{-4}$)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "analysis_gn_regularization_noise.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Gráfico] {os.path.relpath(out, ROOT_DIR)}")


def run_tsvd_truncation_study():
    print("\n>>> Gráfico 3: TSVD truncamiento")
    from fenics import Function, FunctionSpace

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
    axes[0].set_xlabel("Rango $r$")
    axes[0].set_ylabel("Error relativo medio [%]")
    axes[0].set_title("Error de reconstrucción TSVD (50 realizaciones)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    idx = np.arange(1, len(S) + 1)
    w_tsvd = np.array([1.0 if i <= 3 else 0.0 for i in idx])
    w_tik_3 = (S ** 2) / (S ** 2 + 1e-3)
    w_tik_2 = (S ** 2) / (S ** 2 + 1e-2)
    axes[1].plot(idx, w_tsvd, "s--", color="#d62728", lw=2, label="TSVD r=3")
    axes[1].plot(idx, w_tik_3, "o-", color="#1f77b4", lw=2, label=r"Tikhonov $\alpha=10^{-3}$")
    axes[1].plot(idx, w_tik_2, "^-", color="#2ca02c", lw=2, label=r"Tikhonov $\alpha=10^{-2}$")
    axes[1].set_xticks(idx)
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].set_xlabel("Modo $i$")
    axes[1].set_ylabel("Factor de filtro $w_i$")
    axes[1].set_title("Factores de filtro espectral")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "analysis_tsvd_truncation.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Gráfico] {os.path.relpath(out, ROOT_DIR)}")


SECTIONS = {
    "table1": generate_table1_verification_summary,
    "table2": generate_table2_pytest_matrix,
    "table3": run_mesh_convergence_study,
    "gn": run_gn_regularization_noise_study,
    "tsvd": run_tsvd_truncation_study,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark y análisis EIT (Ex1–Ex4)")
    parser.add_argument("--only", default="all",
                        help="table1|table2|table3|gn|tsvd|all")
    args = parser.parse_args()

    if args.only == "all":
        targets = ["table1", "table2", "table3", "gn", "tsvd"]
    elif args.only in SECTIONS:
        targets = [args.only]
    else:
        parser.error(f"--only inválido: {args.only}")

    for t in targets:
        SECTIONS[t]()

    print("\n[OK] Análisis completado.")


if __name__ == "__main__":
    main()
