"""Ejecutor de variantes controladas y estudios paramétricos reutilizando módulos existentes."""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.geometry.domain import Rectangular4ChannelDomain
from src.materials.conductivity import SubdomainConductivity
from src.forward.solver import ForwardSolver
from src.measurements.ntd_map import NtDMeasurementManager
from src.reconstruction.least_squares import GaussNewtonReconstructor
from src.utils.helpers import verify_reciprocity_matrix

from .variants import ControlledVariant


def _is_fenics_domain(domain: Any) -> bool:
    """Detecta si el dominio corresponde a una malla FEniCS/DOLFIN."""
    return hasattr(domain, "mesh") and hasattr(domain.mesh, "coordinates")


def _extract_mesh_stats(domain: Any) -> Tuple[int, int]:
    """Devuelve (num_celdas, num_vertices) de forma agnóstica al backend."""
    if _is_fenics_domain(domain):
        return int(domain.mesh.num_cells()), int(domain.mesh.num_vertices())
    nx = int(getattr(domain, "Nx", domain.mesh_density + 1))
    ny = int(getattr(domain, "Ny", domain.mesh_density + 1))
    return int((nx - 1) * (ny - 1)), int(nx * ny)


def _extract_potential_snapshot(domain: Any, u_init_0: Any, u_final_0: Any) -> Optional[Dict[str, Any]]:
    """Extrae arreglos livianos de potencial (Canal 1) para superficies 3D sin retener objetos C++."""
    if u_init_0 is None or u_final_0 is None:
        return None
    if _is_fenics_domain(domain) and hasattr(u_init_0, "compute_vertex_values"):
        coords = np.array(domain.mesh.coordinates(), dtype=float)
        cells = np.array(domain.mesh.cells(), dtype=int)
        u0 = np.array(u_init_0.compute_vertex_values(domain.mesh), dtype=float)
        uf = np.array(u_final_0.compute_vertex_values(domain.mesh), dtype=float)
        return {
            "mode": "tri",
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cells": cells,
            "u0": u0,
            "uf": uf,
            "du": uf - u0,
        }
    if hasattr(domain, "X") and hasattr(domain, "Y"):
        u0 = np.asarray(u_init_0, dtype=float)
        uf = np.asarray(u_final_0, dtype=float)
        return {
            "mode": "grid",
            "X": np.array(domain.X, dtype=float),
            "Y": np.array(domain.Y, dtype=float),
            "u0": u0,
            "uf": uf,
            "du": uf - u0,
        }
    return None


def _compute_jacobian_spectrum(reconstructor: GaussNewtonReconstructor, u_solutions: Any) -> Dict[str, Any]:
    """Calcula valores singulares y número de condición del Jacobiano A in R^{6x4}."""
    if u_solutions is None:
        return {"singular_values": [np.nan] * 4, "cond_number": float("nan")}
    A = np.asarray(reconstructor.assemble_jacobian(u_solutions), dtype=float)
    svals = np.linalg.svd(A, compute_uv=False)
    cond = float(svals[0] / max(svals[-1], 1e-15)) if len(svals) > 0 else float("nan")
    return {
        "singular_values": [float(s) for s in svals],
        "cond_number": cond,
    }


def run_single_variant(
    variant: ControlledVariant,
    verbose: bool = False,
    capture_potentials: bool = True,
) -> Dict[str, Any]:
    """Ejecuta una variante controlada empleando la arquitectura modular existente."""
    t0 = time.perf_counter()

    domain = Rectangular4ChannelDomain(
        mesh_density=variant.mesh_density,
        electrode_width=variant.electrode_width,
    )
    is_fenics = _is_fenics_domain(domain)
    num_cells, num_vertices = _extract_mesh_stats(domain)

    forward_solver = ForwardSolver(domain)
    meas_mgr = NtDMeasurementManager(domain, forward_solver)

    if variant.use_reference_data:
        f_obs = meas_mgr.get_problem_1_1_reference_data().copy()
        F_obs = meas_mgr.get_problem_1_1_reference_matrix().copy()
        target_gamma = None
    else:
        if variant.target_gamma is None or len(variant.target_gamma) != 4:
            raise ValueError(
                f"La variante '{variant.name}' requiere target_gamma de 4 valores o use_reference_data=True."
            )
        target_gamma = np.asarray(variant.target_gamma, dtype=float)
        cond_true = SubdomainConductivity(domain, initial_val=1.0)
        cond_true.update_gamma(target_gamma)

        f_clean, _ = meas_mgr.get_simulated_vector(cond_true)
        F_clean = meas_mgr.get_simulated_matrix(cond_true)

        if variant.noise_std > 0.0:
            rng = np.random.default_rng(variant.seed)
            noise_vec = rng.normal(0.0, variant.noise_std, size=f_clean.shape)
            f_obs = f_clean + noise_vec
            F_obs = F_clean.copy()
            for idx_pair, (j, k) in enumerate(meas_mgr.pairs):
                F_obs[j, k] = f_obs[idx_pair]
                F_obs[k, j] = f_obs[idx_pair]
        else:
            f_obs = f_clean
            F_obs = F_clean

    cond_est = SubdomainConductivity(domain, initial_val=1.0)
    reconstructor = GaussNewtonReconstructor(domain, cond_est, meas_mgr)
    gamma_est, history = reconstructor.solve(
        f_obs,
        max_iter=variant.max_iter,
        alpha=variant.alpha,
        tol=variant.tol,
        verbose=verbose,
        return_history=True,
    )

    f_sim, u_final_sols = meas_mgr.get_simulated_vector(cond_est)
    F_sim = meas_mgr.get_simulated_matrix(cond_est)
    is_sym, max_asym = verify_reciprocity_matrix(F_sim)
    asym_fro = float(np.linalg.norm(F_sim - F_sim.T, ord="fro"))
    ntd_rel_err_pct = float(
        100.0 * np.linalg.norm(f_sim - f_obs) / max(np.linalg.norm(f_obs), 1e-12)
    )
    ntd_max_abs_err = float(np.max(np.abs(f_sim - f_obs)))

    jac_init = _compute_jacobian_spectrum(reconstructor, history.get("u_initial"))
    jac_final = _compute_jacobian_spectrum(reconstructor, history.get("u_final") or u_final_sols)

    if target_gamma is not None:
        signed_err = (gamma_est - target_gamma).tolist()
        abs_err = np.abs(gamma_est - target_gamma).tolist()
        rel_err_per_q = (100.0 * np.abs(gamma_est - target_gamma) / np.maximum(np.abs(target_gamma), 1e-12)).tolist()
        rel_err_gamma_pct = float(
            100.0 * np.linalg.norm(gamma_est - target_gamma) / max(np.linalg.norm(target_gamma), 1e-12)
        )
        target_list = target_gamma.tolist()
    else:
        signed_err = None
        abs_err = None
        rel_err_per_q = None
        rel_err_gamma_pct = None
        target_list = None

    residuals = [float(r) for r in history.get("residuals", [])]
    gamma_hist = [np.asarray(g, dtype=float).tolist() for g in history.get("gamma_history", [])]
    init_res = residuals[0] if residuals else 0.0
    final_res = float(np.linalg.norm(f_sim - f_obs))
    reduction_factor = float(init_res / max(final_res, 1e-12)) if init_res > 0 else 1.0

    pot_snapshot = None
    if capture_potentials:
        u_init_list = history.get("u_initial")
        u_final_list = history.get("u_final") or u_final_sols
        if u_init_list and u_final_list:
            pot_snapshot = _extract_potential_snapshot(domain, u_init_list[0], u_final_list[0])

    elapsed_s = time.perf_counter() - t0
    n_iters = len(residuals)
    time_per_iter = float(elapsed_s / max(n_iters, 1))

    return {
        "name": variant.name,
        "description": variant.description,
        "factor": variant.metadata.get("factor", "general"),
        "backend": "FEniCS" if is_fenics else "NumPy-FDM",
        "mesh_density": variant.mesh_density,
        "num_cells": num_cells,
        "num_vertices": num_vertices,
        "alpha": variant.alpha,
        "noise_std": variant.noise_std,
        "use_reference_data": variant.use_reference_data,
        "iterations": n_iters,
        "initial_residual": init_res,
        "final_residual": final_res,
        "residual_reduction": reduction_factor,
        "residuals": residuals,
        "gamma_history": gamma_hist,
        "target_gamma": target_list,
        "estimated_gamma": [float(g) for g in gamma_est],
        "gamma_norm": float(np.linalg.norm(gamma_est)),
        "signed_err_per_quadrant": signed_err,
        "abs_err_per_quadrant": abs_err,
        "rel_err_per_quadrant_pct": rel_err_per_q,
        "rel_err_gamma_pct": rel_err_gamma_pct,
        "f_obs": [float(v) for v in f_obs],
        "f_sim": [float(v) for v in f_sim],
        "ntd_rel_err_pct": ntd_rel_err_pct,
        "ntd_max_abs_err": ntd_max_abs_err,
        "reciprocity_ok": bool(is_sym),
        "reciprocity_max_diff": float(max_asym),
        "reciprocity_fro": asym_fro,
        "jac_init_svals": jac_init["singular_values"],
        "jac_init_cond": jac_init["cond_number"],
        "jac_final_svals": jac_final["singular_values"],
        "jac_final_cond": jac_final["cond_number"],
        "elapsed_s": float(elapsed_s),
        "time_per_iter_s": time_per_iter,
        "F_obs": F_obs,
        "F_sim": F_sim,
        "potential_snapshot": pot_snapshot,
    }


def run_all_variants(variants: List[ControlledVariant], verbose: bool = True) -> List[Dict[str, Any]]:
    """Ejecuta todas las variantes controladas de manera secuencial."""
    results: List[Dict[str, Any]] = []
    for idx, var in enumerate(variants, start=1):
        if verbose:
            print(f"[{idx}/{len(variants)}] Ejecutando variante: {var.name} ...")
        res = run_single_variant(var, verbose=False, capture_potentials=True)
        results.append(res)
        if verbose:
            err_txt = (
                f"{res['rel_err_gamma_pct']:.2f}%"
                if res["rel_err_gamma_pct"] is not None
                else "N/A (datos reales)"
            )
            print(
                f"    -> Backend: {res['backend']} | Iter: {res['iterations']} | "
                f"Err gamma: {err_txt} | Err NtD: {res['ntd_rel_err_pct']:.3f}% | "
                f"cond(J): {res['jac_final_cond']:.2f} | gamma_hat: {[round(x, 3) for x in res['estimated_gamma']]}"
            )
    return results


def run_lcurve_study(
    alphas: Sequence[float] = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    target_gamma: Tuple[float, float, float, float] = (1.0, 2.0, 1.0, 1.0),
    mesh_density: int = 32,
    noise_std: float = 1e-2,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Barrido paramétrico de regularización Tikhonov (Curva L) bajo ruido controlado."""
    study: List[Dict[str, Any]] = []
    if verbose:
        print(f"Iniciando barrido Curva-L ({len(alphas)} valores de alpha, ruido={noise_std:.1e})...")
    for a in alphas:
        v = ControlledVariant(
            name=f"LC_alpha_{a:.0e}",
            description=f"Barrido Curva-L con alpha={a:.1e}",
            target_gamma=target_gamma,
            mesh_density=mesh_density,
            alpha=float(a),
            noise_std=noise_std,
            max_iter=15,
            seed=seed,
            metadata={"factor": "curva_L"},
        )
        res = run_single_variant(v, verbose=False, capture_potentials=False)
        study.append(res)
    return study


def run_mesh_refinement_study(
    densities: Sequence[int] = (16, 24, 32, 48),
    target_gamma: Tuple[float, float, float, float] = (1.0, 2.0, 1.0, 1.0),
    alpha: float = 1e-4,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Barrido paramétrico de refinamiento de malla sobre el caso base."""
    study: List[Dict[str, Any]] = []
    if verbose:
        print(f"Iniciando estudio de refinamiento de malla ({list(densities)})...")
    for d in densities:
        v = ControlledVariant(
            name=f"MR_malla_{d}x{d}",
            description=f"Estudio de refinamiento con malla {d}x{d}",
            target_gamma=target_gamma,
            mesh_density=int(d),
            alpha=alpha,
            noise_std=0.0,
            max_iter=15,
            metadata={"factor": "refinamiento_malla"},
        )
        res = run_single_variant(v, verbose=False, capture_potentials=False)
        study.append(res)
    return study
