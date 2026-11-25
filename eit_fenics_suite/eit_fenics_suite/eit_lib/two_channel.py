"""
eit_lib/two_channel.py
Implementación de Example 1.5 (Ammari, pág. 11): Two-Channel EIT con
conductividad compleja gamma = sigma + i * omega * epsilon.
"""

from ._fenics_check import require_fenics
require_fenics()

from fenics import (
    RectangleMesh, Point, MeshFunction, SubDomain, Measure, assemble,
    Constant, FunctionSpace, DirichletBC, TrialFunction, TestFunction,
    dot, grad, dx, solve, Function, near,
)


class TwoChannelEIT:
    """
    Sistema EIT de 2 canales:
    - Dominio Omega = [-1, 1] x [-1, 1]
    - Electrodo E0: Fondo (y = -1, |x| <= w), u|_{E0} = 0 (Tierra)
    - Electrodo E1: Techo (y = 1, |x| <= w), inyección de corriente I0 cos(omega t)
    - Propiedades eléctricas uniformes: sigma y epsilon constantes
    - Reconstrucción de gamma = sigma + i*omega*epsilon a partir del voltaje complejo V_meas
    """

    def __init__(self, nx=32, ny=32, electrode_width=0.4, current_intensity=1.0):
        self.I0 = current_intensity
        self.w = electrode_width / 2.0

        self.mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 1.0), nx, ny)
        self.boundaries = MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        self.boundaries.set_all(0)

        w_val = self.w

        class Electrode0(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], -1.0) and abs(x[0]) <= w_val

        class Electrode1(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 1.0) and abs(x[0]) <= w_val

        self.e0 = Electrode0()
        self.e1 = Electrode1()
        self.e0.mark(self.boundaries, 10)  # E0 (tierra)
        self.e1.mark(self.boundaries, 11)  # E1 (inyección y medición)

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        self.area_E0 = assemble(Constant(1.0) * self.ds(10))
        self.area_E1 = assemble(Constant(1.0) * self.ds(11))

        self.V = FunctionSpace(self.mesh, "Lagrange", 1)
        self.bc0 = DirichletBC(self.V, Constant(0.0), self.boundaries, 10)

        # Potencial geométrico de referencia u0 con gamma = 1
        self.u0 = self._solve_reference_potential()
        self.energy_u0 = assemble(dot(grad(self.u0), grad(self.u0)) * dx)
        self.v0_E1 = assemble(self.u0 * self.ds(11)) / self.area_E1

    def _solve_reference_potential(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        g1 = Constant(self.I0 / self.area_E1)

        a = dot(grad(u), grad(v)) * dx
        L = g1 * v * self.ds(11)

        u0_sol = Function(self.V)
        solve(a == L, u0_sol, self.bc0)
        return u0_sol

    def forward_solve(self, sigma, epsilon, omega):
        """
        Calcula el voltaje complejo medido en E1:
        gamma = sigma + i * omega * epsilon
        V_meas = v0_E1 / gamma
        """
        gamma = complex(sigma, omega * epsilon)
        V_meas = self.v0_E1 / gamma
        return V_meas, gamma

    def inverse_solve(self, V_meas, omega):
        """
        Reconstruye sigma y epsilon a partir del voltaje complejo V_meas:
        gamma = C_Omega / V_meas, con C_Omega = int_Omega |grad u0|^2 dx.
        """
        C_Omega = self.energy_u0
        gamma_recon = C_Omega / V_meas
        sigma_recon = gamma_recon.real
        epsilon_recon = gamma_recon.imag / omega if omega > 0 else 0.0
        return sigma_recon, epsilon_recon, gamma_recon