r"""
Módulo del Solucionador Directo (Forward Problem) en FEniCS.

Ecuación de Gobierno (Problema Elíptico de Poisson/Laplace generalizado):
    div(sigma * grad u) = 0   en Omega
    u = 0                     en E_0 (Electrodo de referencia / Tierra)
    sigma * du/dn = -I / |E_j| en E_j (Electrodo activo j in {1, 2, 3})
    sigma * du/dn = 0         en dOmega \ (E_0 U E_j) (Frontera aislante)

Nota Metodológica:
    La fijación de potencial nulo en el electrodo E_0 (DirichletBC homogéneo)
    elimina el subespacio nulo del operador de Laplace (gauge fixing) y garantiza
    la existencia y unicidad de solución en H^1(Omega) según el lema de Lax-Milgram.
"""

from dolfin import (
    TrialFunction, TestFunction, Function, DirichletBC, Constant,
    dot, grad, dx, solve
)


class ForwardSolver:
    def __init__(self, domain):
        self.domain = domain

    def solve_channel(self, conductivity, j_active, I=1.0):
        """
        Resuelve la EDP elíptica para el patrón de excitación del canal j_active.

        Parámetros:
            conductivity (SubdomainConductivity): Instancia con el campo sigma en DG0.
            j_active (int): Índice del electrodo activo (1, 2 o 3).
            I (float): Intensidad total de corriente inyectada [A].

        Retorna:
            dolfin.Function: Campo continuo de potencial escalar u_j in H^1(Omega).
        """
        V = self.domain.V
        u = TrialFunction(V)
        v = TestFunction(V)

        # Forma bilineal a(u, v) = integral_Omega (sigma * grad(u) . grad(v)) dx
        sigma_fenics = conductivity.get_fenics_function()
        a = sigma_fenics * dot(grad(u), grad(v)) * dx

        # Forma lineal L(v) = integral_{E_j} (g_inj * v) ds
        area_j = self.domain.area_elec[j_active - 1]
        g_inj = Constant(-I / area_j)
        L = g_inj * v * self.domain.ds_elec(j_active)

        # Condición de calibre / tierra en electrodo 0 (etiqueta 10 en boundaries)
        bc_ground = DirichletBC(V, Constant(0.0), self.domain.boundaries, 10)

        u_sol = Function(V)
        solve(a == L, u_sol, bc_ground)
        return u_sol
