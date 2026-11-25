"""
run_all.py
Ejecuta todos los ejercicios (exercises/*.py) y módulos de análisis
(analysis/*.py) en orden alfabético, sin necesidad de editar listas.
"""

import glob
import os
import subprocess
import sys

try:
    import fenics  # noqa: F401
except ImportError:
    sys.stderr.write(
        "\n[ERROR] FEniCS Legacy (dolfin) no disponible. "
        "Instala con conda (fenics=2019.1.0).\n"
    )
    sys.exit(1)


def discover(pattern):
    return sorted(glob.glob(os.path.join(os.path.dirname(__file__),