"""Definición de variantes controladas con conductividad constante por cuadrantes."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ControlledVariant:
    """Parámetros de una variante controlada sobre el sistema EIT de 4 canales."""

    name: str
    description: str
    target_gamma: Optional[Tuple[float, float, float, float]] = None
    mesh_density: int = 32
    electrode_width: float = 0.35
    alpha: float = 1e-4
    noise_std: float = 0.0
    max_iter: int = 15
    tol: float = 1e-4
    seed: int = 42
    use_reference_data: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


def get_default_variants() -> List[ControlledVariant]:
    """Retorna variantes controladas modificando un factor experimental a la vez."""
    return [
        ControlledVariant(
            name="V0_base_referencia",
            description="Caso base sintético con contraste en el cuadrante Omega_2",
            target_gamma=(1.0, 2.0, 1.0, 1.0),
            mesh_density=32,
            alpha=1e-4,
            noise_std=0.0,
            max_iter=15,
            metadata={"factor": "linea_base"},
        ),
        ControlledVariant(
            name="V1_medio_homogeneo",
            description="Medio homogéneo unitario para verificar ausencia de contrastes espurios",
            target_gamma=(1.0, 1.0, 1.0, 1.0),
            mesh_density=32,
            alpha=1e-4,
            noise_std=0.0,
            max_iter=15,
            metadata={"factor": "distribucion_conductividad"},
        ),
        ControlledVariant(
            name="V2_contraste_multicuadrante",
            description="Conductividades constantes distintas en los cuatro cuadrantes",
            target_gamma=(1.6, 0.8, 1.3, 1.9),
            mesh_density=32,
            alpha=1e-4,
            noise_std=0.0,
            max_iter=15,
            metadata={"factor": "distribucion_conductividad"},
        ),
        ControlledVariant(
            name="V3_anomalia_aislante_q4",
            description="Anomalía poco conductora en Omega_4 análoga al patrón del Problema 1.1",
            target_gamma=(1.0, 1.0, 1.0, 0.3),
            mesh_density=32,
            alpha=1e-4,
            noise_std=0.0,
            max_iter=15,
            metadata={"factor": "distribucion_conductividad"},
        ),
        ControlledVariant(
            name="V4_reg_baja_ruido",
            description="Regularización baja (alpha=1e-6) bajo ruido gaussiano aditivo en frontera",
            target_gamma=(1.0, 2.0, 1.0, 1.0),
            mesh_density=32,
            alpha=1e-6,
            noise_std=1e-2,
            max_iter=15,
            seed=42,
            metadata={"factor": "regularizacion_y_ruido"},
        ),
        ControlledVariant(
            name="V5_reg_alta_ruido",
            description="Regularización alta (alpha=1e-2) bajo el mismo ruido gaussiano de V4",
            target_gamma=(1.0, 2.0, 1.0, 1.0),
            mesh_density=32,
            alpha=1e-2,
            noise_std=1e-2,
            max_iter=15,
            seed=42,
            metadata={"factor": "regularizacion_y_ruido"},
        ),
        ControlledVariant(
            name="V6_malla_fina",
            description="Refinamiento de malla a 48x48 manteniendo los parámetros de V0",
            target_gamma=(1.0, 2.0, 1.0, 1.0),
            mesh_density=48,
            alpha=1e-4,
            noise_std=0.0,
            max_iter=15,
            metadata={"factor": "refinamiento_malla"},
        ),
        ControlledVariant(
            name="V7_referencia_seo_woo",
            description="Reproducción directa con datos experimentales Baby NtD del Problema 1.1",
            target_gamma=None,
            mesh_density=32,
            alpha=1e-4,
            noise_std=0.0,
            max_iter=15,
            use_reference_data=True,
            metadata={"factor": "datos_referencia_libro"},
        ),
    ]
