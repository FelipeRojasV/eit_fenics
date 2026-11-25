"""
eit_lib: Librería modular de Tomografía de Impedancia Eléctrica (EIT)
basada en FEniCS Legacy (dolfin 2019.1.0).

Importar este paquete lanza ImportError si FEniCS Legacy no está
disponible, para garantizar que todos los resultados provengan de FEM real.
"""

from ._fenics_check import fenics_available, require_fenics

# Falla ruidosamente si FEniCS no está instalado.
require_fenics()

from .mesh_setup import EITGeometry
from .forward_solver import EITForwardSolver
from .sensitivity import EITSensitivity
from .inversion import (
    extract_measurement_vector,
    solve_least_squares,
    solve_svd_linearized,
)
from .representation import reconstruct_quadrants
from .two_channel import TwoChannelEIT

__all__ = [
    "EITGeometry",
    "EITForwardSolver",
    "EITSensitivity",
    "extract_measurement_vector",
    "solve_least_squares",
    "solve_svd_linearized",
    "reconstruct_quadrants",
    "TwoChannelEIT",
    "fenics_available",
    "require_fenics",
]