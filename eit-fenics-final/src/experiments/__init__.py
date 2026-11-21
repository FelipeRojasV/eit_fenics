"""Paquete de experimentación controlada para variantes del Problema 1.1 (EIT 4 canales)."""

from .variants import ControlledVariant, get_default_variants
from .runner import (
    run_single_variant,
    run_all_variants,
    run_lcurve_study,
    run_mesh_refinement_study,
)
from .reporting import export_variant_tables, plot_variant_results

__all__ = [
    "ControlledVariant",
    "get_default_variants",
    "run_single_variant",
    "run_all_variants",
    "run_lcurve_study",
    "run_mesh_refinement_study",
    "export_variant_tables",
    "plot_variant_results",
]
