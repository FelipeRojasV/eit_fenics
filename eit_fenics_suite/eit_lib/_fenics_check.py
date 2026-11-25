"""
eit_lib/_fenics_check.py
Centraliza la verificación de la dependencia FEniCS Legacy (dolfin 2019.1.0).
Todos los módulos de eit_lib y los ejercicios importan desde aquí.
"""

_FENICS_AVAILABLE = None


def fenics_available():
    """Devuelve True si `fenics` (dolfin) puede importarse."""
    global _FENICS_AVAILABLE
    if _FENICS_AVAILABLE is not None:
        return _FENICS_AVAILABLE
    try:
        import fenics  # noqa: F401
        _FENICS_AVAILABLE = True
    except ImportError:
        _FENICS_AVAILABLE = False
    return _FENICS_AVAILABLE


def require_fenics():
    """
    Lanza ImportError si FEniCS Legacy no está disponible.
    Esta suite NO tiene modo fallback: los resultados deben generarse
    siempre con FEM real a través de FEniCS.
    """
    if not fenics_available():
        raise ImportError(
            "\n"
            "==============================================================\n"
            " EIT FEniCS Legacy Suite requiere FEniCS Legacy (dolfin).\n"
            " Esta suite no incluye modo fallback: los resultados deben\n"
            " generarse con FEM real a través de FEniCS.\n"
            "\n"
            " Instalación recomendada (conda):\n"
            "   conda create -n fenics -c conda-forge fenics=2019.1.0 \\\n"
            "         matplotlib scipy numpy pytest\n"
            "   conda activate fenics\n"
            "\n"
            " Ver README.md para más detalles.\n"
            "==============================================================\n"
        )