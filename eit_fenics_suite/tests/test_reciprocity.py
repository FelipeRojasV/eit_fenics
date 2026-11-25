import numpy as np
from fenics import FunctionSpace, interpolate, Expression

from eit_lib import EITGeometry, EITForwardSolver


def test_reciprocity_holds_for_inhomogeneous_sigma():
    geom = EITGeometry(nx=16, ny=16)
    solver = EITForwardSolver(geom, current_intensity=1.0)

    V = FunctionSpace(geom.mesh, "Lagrange", 1)
    sigma_expr = Expression(
        "1.0 + 0.5 * exp(-4.0*(pow(x[0]-0.2, 2) + pow(x[1]-0.1, 2)))",
        degree=2,
    )
    sigma = interpolate(sigma_expr, V)
    F, _ = solver.compute_ntd_matrix(sigma)

    assert np.allclose(F, F.T, atol=1e-6)