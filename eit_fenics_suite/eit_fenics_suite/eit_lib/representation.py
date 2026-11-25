"""
eit_lib/representation.py
Utilidades de representación malla <-> vector.
Convierte valores por cuadrante en una Function DG(0) de FEniCS.
Separado de inversion.py porque es una operación de representación,
no un algoritmo de inversión.
"""

from ._fenics_check import require_fenics
require_fenics()

from fenics import FunctionSpace, Function, cells
import numpy as np


def reconstruct_quadrants(mesh, subdomains, sigma_values):
    """
    Asigna valores constantes por cuadrante a una Function DG(0) en FEniCS.

    Parámetros
    ----------
    mesh : dolfin.Mesh
    subdomains : dolfin.MeshFunction
        Marcas por celda (1, 2, 3, 4) para los cuatro cuadrantes.
    sigma_values : array-like, length 4

    Returns
    -------
    dolfin.Function en DG(0)
    """
    V = FunctionSpace(mesh, "DG", 0)
    func = Function(V)
    arr = np.zeros(mesh.num_cells())
    for cell in cells(mesh):
        sub_id = subdomains[cell]
        arr[cell.index()] = sigma_values[sub_id - 1]
    func.vector()[:] = arr
    return func