"""
tests/conftest.py
Salta toda la suite de tests si FEniCS Legacy no está disponible.
"""

import pytest

pytest.importorskip("fenics", reason="FEniCS Legacy (dolfin) no está disponible")