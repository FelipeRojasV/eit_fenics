import numpy as np
from fenics import Function, FunctionSpace

from eit_lib import EITGeometry, EITForwardSolver


def test_ntd_matrix_symmetric_for_homogeneous_sigma():
    geom = EITGeometry(nx=16, ny=16)
    solver = EITForwardSolver(geom, current_intensity=1.0)

    V_dg = FunctionSpace(geom.mesh, "DG", 0)
    sigma = Function(V_dg)
    sigma.vector()[:] = 1.0

    F, _ = solver.compute_ntd_matrix(sigma)
    assert F.shape == (3, 3)
    assert np.allclose(F, F.T, atol=1e-8)


def test_electrode_areas_consistency():
    geom = EITGeometry(nx=16, ny=16)
    areas = geom.electrode_areas()
    assert len(areas) == 4
    assert all(a > 0 for a in areas)
    # Los cuatro electrodos tienen la misma longitud en esta geometría
    assert np.allclose(areas, areas[0])