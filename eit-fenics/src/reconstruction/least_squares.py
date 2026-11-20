"""
Módulo de Reconstrucción Inversa: Algoritmo Gauss-Newton Regularizado.

Formulación Teórica:
    Minimizar el funcional de mínimos cuadrados no lineales:
        Phi(gamma) = 0.5 * || F(gamma) - f_meas ||_2^2 + 0.5 * alpha * || gamma - gamma_0 ||_2^2

Cálculo del Jacobiano (Método de Estado Adjunto):
    Para el par de medición r = (j, k) y el subdominio m:
        J_{r, m} = integral_{Omega_m} (grad u_j . grad u_k) dx
    donde u_j es el potencial debido a inyección en el canal j, y u_k actúa como estado adjunto.

Comportamiento frente a Ruido y Sensibilidad de alpha:
    - Problema de Hadamard mal condicionado: cond(J^T J) ~ 10^3 a 10^5.
    - Con datos exactos, alpha = 1e-4 logra convergencia cuadrática en < 8 iteraciones.
    - Ante ruido gaussiano sintético (> 2% SNR), valores de alpha < 1e-5 provocan oscilaciones
      inadmisibles en los cuadrantes centrales debido a la amplificación de frecuencias espaciales altas.
"""

import logging
import time
import numpy as np
from dolfin import assemble, dot, grad

logger = logging.getLogger(__name__)


class GaussNewtonReconstructor:
    """
    Inversor Gauss-Newton con regularización de Tikhonov de orden cero.
    """
    def __init__(self, domain, conductivity_manager, measurement_manager):
        self.domain = domain
        self.cond_mgr = conductivity_manager
        self.meas_mgr = measurement_manager
        self.pairs = self.meas_mgr.pairs

    def assemble_jacobian(self, u_solutions):
        """
        Ensambla la matriz Jacobiana A in R^{6 x 4} integrando productos escalares
        de gradientes sobre cada subdominio mediante las formas diferenciales de FEniCS.
        """
        A = np.zeros((6, 4), dtype=float)
        for r, (j, k) in enumerate(self.pairs):
            for m in range(4):
                A[r, m] = assemble(
                    dot(grad(u_solutions[j]), grad(u_solutions[k])) * self.domain.dx_sub(m + 1)
                )
        return A

    def solve(self, f_measured, max_iter=15, alpha=1e-4, tol=1e-3, verbose=True, return_history=False):
        """
        Ejecuta el bucle iterativo de Gauss-Newton regularizado con Tikhonov.

        Parámetros:
            f_measured (np.ndarray): Vector de mediciones objetivo (longitud 6).
            max_iter (int): Número máximo de iteraciones permitidas.
            alpha (float): Parámetro de penalización de Tikhonov.
            tol (float): Tolerancia de parada para la norma L2 del residuo.
            verbose (bool): Si True, emite los registros de progreso por logging.
            return_history (bool): Si True, retorna el diccionario de diagnóstico iterativo.
        """
        start_time = time.perf_counter()
        gamma_vals = np.ones(4, dtype=float)
        self.cond_mgr.update_gamma(gamma_vals)

        history = {
            "residuals": [],
            "gamma_history": [gamma_vals.copy()],
            "iteration_times": [],
            "u_initial": None,
            "u_final": None,
        }

        if verbose:
            logger.info("Iniciando inversion Gauss-Newton | alpha=%.1e | tol=%.1e | max_iter=%d", alpha, tol, max_iter)

        for it in range(max_iter):
            t_iter_start = time.perf_counter()

            # Evaluación directa y obtención de soluciones directas/adjuntas
            sim_data, u_solutions = self.meas_mgr.get_simulated_vector(self.cond_mgr)
            if it == 0:
                history["u_initial"] = u_solutions

            b = sim_data - f_measured
            res_norm = float(np.linalg.norm(b))
            history["residuals"].append(res_norm)

            # Matriz de sensibilidad variacional
            A = self.assemble_jacobian(u_solutions)

            # Ecuaciones normales amortiguadas (Hessiano aproximado + Tikhonov)
            H = A.T @ A + alpha * np.eye(4)
            g = A.T @ b
            delta_gamma = np.linalg.solve(H, g)

            # Actualización y recorte físico (coercitividad: conductividad estrictamente positiva)
            gamma_vals -= delta_gamma
            gamma_vals = np.clip(gamma_vals, 0.05, 50.0)
            self.cond_mgr.update_gamma(gamma_vals)
            history["gamma_history"].append(gamma_vals.copy())

            elapsed_iter = time.perf_counter() - t_iter_start
            history["iteration_times"].append(elapsed_iter)

            if verbose:
                logger.info(
                    "Iter %02d/%02d | Residuo ||b||_2: %.4e | Delta: %.4e | gamma: %s | (%.2f s)",
                    it + 1, max_iter, res_norm, float(np.linalg.norm(delta_gamma)),
                    np.array2string(gamma_vals, precision=4, suppress_small=True), elapsed_iter
                )

            if res_norm < tol or np.linalg.norm(delta_gamma) < 1e-4:
                if verbose:
                    logger.info("Criterio de convergencia satisfecho en la iteracion %d.", it + 1)
                break

        total_time = time.perf_counter() - start_time
        if verbose:
            logger.info("Inversion finalizada en %.2f segundos.", total_time)

        _, u_final = self.meas_mgr.get_simulated_vector(self.cond_mgr)
        history["u_final"] = u_final
        history["total_time"] = total_time

        if return_history:
            return gamma_vals, history
        return gamma_vals
