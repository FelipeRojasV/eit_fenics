"""
eit_lib/mesh_setup.py
Definición de geometrías, fronteras de electrodos y partición de subdominios
para sistemas de 4 electrodos en FEniCS Legacy.
"""

from ._fenics_check import require_fenics
require_fenics()

from fenics import (
    RectangleMesh, Point, MeshFunction, SubDomain, Measure, assemble, Constant,
    near,
)


class EITGeometry:
    """
    Geometría rectangular [-1, 1] x [-1, 1] con 4 electrodos superficiales:
    - E0: Fondo (y = -1, |x| <= w) -> Referencia / Tierra
    - E1: Derecha (x = 1, |y| <= w)
    - E2: Techo (y = 1, |x| <= w)
    - E3: Izquierda (x = -1, |y| <= w)
    """

    def __init__(self, nx=32, ny=32, electrode_width=0.4):
        self.nx = nx
        self.ny = ny
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
                return on_boundary and near(x[0], 1.0) and abs(x[1]) <= w_val

        class Electrode2(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 1.0) and abs(x[0]) <= w_val

        class Electrode3(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], -1.0) and abs(x[1]) <= w_val

        self.e0 = Electrode0()
        self.e1 = Electrode1()
        self.e2 = Electrode2()
        self.e3 = Electrode3()

        self.e0.mark(self.boundaries, 10)  # E0
        self.e1.mark(self.boundaries, 11)  # E1
        self.e2.mark(self.boundaries, 12)  # E2
        self.e3.mark(self.boundaries, 13)  # E3

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        self.electrode_ids = [11, 12, 13]

        self.area_E0 = assemble(Constant(1.0) * self.ds(10))
        self.area_Ek = [
            assemble(Constant(1.0) * self.ds(tag)) for tag in self.electrode_ids
        ]

        # Partición en 4 cuadrantes (L = 4 parámetros para inversión)
        self.subdomains = MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim()
        )
        self.subdomains.set_all(0)

        class Quad1(SubDomain):
            def inside(self, x, ob):
                return x[0] >= 0.0 and x[1] >= 0.0

        class Quad2(SubDomain):
            def inside(self, x, ob):
                return x[0] < 0.0 and x[1] >= 0.0

        class Quad3(SubDomain):
            def inside(self, x, ob):
                return x[0] < 0.0 and x[1] < 0.0

        class Quad4(SubDomain):
            def inside(self, x, ob):
                return x[0] >= 0.0 and x[1] < 0.0

        Quad1().mark(self.subdomains, 1)
        Quad2().mark(self.subdomains, 2)
        Quad3().mark(self.subdomains, 3)
        Quad4().mark(self.subdomains, 4)
        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.subdomains)

    def electrode_areas(self):
        """
        Áreas de los 4 electrodos en orden [E0, E1, E2, E3].

        E0 es el electrodo de tierra y no inyecta corriente en el esquema
        por defecto, pero su área se expone explícitamente para permitir
        esquemas alternativos de inyección/medición por E0 y para cerrar
        la API de la geometría. Esto resuelve la inconsistencia anterior
        en la que `area_E0` se calculaba pero nunca se usaba.
        """
        return [self.area_E0] + list(self.area_Ek)