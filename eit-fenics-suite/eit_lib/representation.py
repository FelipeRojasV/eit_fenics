"""
Utilidades de representación malla <-> vector.
"""

from ._fenics_check import require_fenics
require_fenics()

from fenics import FunctionSpace, Function, cells
import numpy as np


def reconstruct_quadrants(mesh, subdomains, sigma_values):
    V = FunctionSpace(mesh, "DG", 0)
    func = Function(V)
    arr = np.zeros(mesh.num_cells())
    for cell in cells(mesh):
        arr[cell.index()] = sigma_values[subdomains[cell] - 1]
    func.vector()[:] = arr
    return func
