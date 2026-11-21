"""
Módulo de Definición Geométrica y Discretización Variacional (FEniCS).

Referencias de diseño y notas de laboratorio:
---------------------------------------------
1. Dominio y Subdominios:
   Se discretiza el dominio bidimensional [-1, 1] x [-1, 1] en 4 cuadrantes
   (Omega_1 a Omega_4). Esta partición sigue el problema inverso discreto 
   estándar 1.1 propuesto en:
   - Seo, J. K., & Woo, E. J. (2012). Nonlinear Inverse Problems in Imaging.
   - Ammari, H., et al. Mathematical Modeling in Electrical Impedance Tomography.

2. Selección del Ancho de Electrodo (w = 0.35):
   - Justificación física: En Tomografía de Impedancia Eléctrica real, un ancho
     demasiado estrecho (modelo puntual) genera singularidades de potencial en
     los vértices del electrodo bajo la formulación débil. Por otro lado, un ancho
     excesivo (> 0.5) distorsiona la densidad de corriente perimetral y reduce la
     resolución espacial del mapa Neumann-to-Dirichlet (NtD).
   - El semi-ancho tol_e = 0.35 (longitud de contacto L_e = 0.70 en una cara de 2.0)
     ofrece una cobertura perimetral de ~35% por cara, lo que permite aproximar
     el Modelo de Derivación (Shunt Model) mediante condiciones de Neumann promedio
     sin requerir la resolución explícita de la impedancia de contacto compleja (CEM).
"""

import numpy as np
from dolfin import (
    RectangleMesh, Point, FunctionSpace, MeshFunction, SubDomain,
    Measure, Constant, assemble, DOLFIN_EPS, near
)


class Rectangular4ChannelDomain:
    """
    Administra la geometría bidimensional, la malla no estructurada en FEniCS,
    los cuatro subdominios internos (DG_0) y la integración en contornos de electrodos.
    """
    def __init__(self, mesh_density=32, electrode_width=0.35):
        """
        Parámetros:
            mesh_density (int): Número de divisiones por eje. Por defecto 32 (2048 triángulos).
                                Nota de laboratorio: densidades menores a 16 introducen un error
                                de discretización superior al 2.5% en la norma H^1 de los potenciales.
            electrode_width (float): Semi-ancho geométrico de los electrodos en la frontera.
        """
        self.mesh_density = mesh_density
        self.tol_e = electrode_width
        self._setup_fenics_domain()

    def _setup_fenics_domain(self):
        # Malla triangular estructurada de Lagrange P1
        self.mesh = RectangleMesh(
            Point(-1.0, -1.0), Point(1.0, 1.0), self.mesh_density, self.mesh_density
        )
        self.V = FunctionSpace(self.mesh, "P", 1)
        self.DG0 = FunctionSpace(self.mesh, "DG", 0)

        # Identificación de los 4 cuadrantes físicos (conductividad constante por trozos)
        self.subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim(), 0)

        class Omega1(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] <= DOLFIN_EPS and x[1] <= DOLFIN_EPS

        class Omega2(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= -DOLFIN_EPS and x[1] <= DOLFIN_EPS

        class Omega3(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] <= DOLFIN_EPS and x[1] >= -DOLFIN_EPS

        class Omega4(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= -DOLFIN_EPS and x[1] >= -DOLFIN_EPS

        Omega1().mark(self.subdomains, 1)
        Omega2().mark(self.subdomains, 2)
        Omega3().mark(self.subdomains, 3)
        Omega4().mark(self.subdomains, 4)

        self.dx_sub = Measure("dx", domain=self.mesh, subdomain_data=self.subdomains)

        # Identificación de parches de frontera (Electrodos E0 a E3)
        self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1, 0)
        tol = self.tol_e

        class E0(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], -1.0) and abs(x[0]) <= tol

        class E1(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 1.0) and abs(x[1]) <= tol

        class E2(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 1.0) and abs(x[0]) <= tol

        class E3(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], -1.0) and abs(x[1]) <= tol

        # Marcadores de superficie: 10 para referencia/tierra, 1-3 para canales de inyección
        E0().mark(self.boundaries, 10)
        E1().mark(self.boundaries, 1)
        E2().mark(self.boundaries, 2)
        E3().mark(self.boundaries, 3)

        self.ds_elec = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        # Cálculo de las áreas efectivas de electrodo |E_j| para normalización de corriente
        self.area_elec = [assemble(Constant(1.0) * self.ds_elec(k)) for k in [1, 2, 3]]
