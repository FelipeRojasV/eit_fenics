"""Script ejecutable para correr las variantes controladas del Problema 1.1 (Suite v3 con 3D).

No modifica ningún módulo existente del repositorio eit-fenics.
Guarda 5 figuras esenciales (incluyendo vistas 3D), 5 tablas (.md, .tex, .png), 3 CSV y log
en results/controlled_variants/.
"""

import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiments import (
    export_variant_tables,
    get_default_variants,
    plot_variant_results,
    run_all_variants,
    run_lcurve_study,
    run_mesh_refinement_study,
)


def setup_logger(out_root: Path) -> logging.Logger:
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "controlled_variants.log"
    logger = logging.getLogger("controlled_variants")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _clean_previous_outputs(figures_dir: Path, tables_dir: Path) -> None:
    """Limpia figuras/tablas previas dentro de results/controlled_variants/ para evitar archivos obsoletos."""
    if figures_dir.exists():
        for old_fig in figures_dir.glob("var_*.png"):
            old_fig.unlink()
    if tables_dir.exists():
        for pattern in ("tabla_*.md", "tabla_*.tex", "tabla_*.png", "resumen_variantes_controladas.*", "detalle_cuadrantes_variantes.*"):
            for old_tbl in tables_dir.glob(pattern):
                old_tbl.unlink()


def main() -> None:
    out_root = ROOT_DIR / "results" / "controlled_variants"
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    _clean_previous_outputs(figures_dir, tables_dir)
    logger = setup_logger(out_root)

    logger.info("=== INICIO DE EXPERIMENTACIÓN CON VARIANTES CONTROLADAS v3 (5 FIGURAS CON 3D) ===")
    variants = get_default_variants()
    logger.info("Total de variantes configuradas: %d", len(variants))

    results = run_all_variants(variants, verbose=True)

    for r in results:
        err_str = f"{r['rel_err_gamma_pct']:.2f}%" if r["rel_err_gamma_pct"] is not None else "N/A"
        logger.info(
            "Variante=%s | Backend=%s | Malla=%dx%d | alpha=%.1e | ruido=%.1e | "
            "iter=%d | err_gamma=%s | err_ntd=%.3f%% | cond(J)=%.2f | gamma_hat=%s",
            r["name"], r["backend"], r["mesh_density"], r["mesh_density"],
            r["alpha"], r["noise_std"], r["iterations"], err_str,
            r["ntd_rel_err_pct"], r["jac_final_cond"],
            [round(x, 4) for x in r["estimated_gamma"]],
        )

    lcurve_results = run_lcurve_study(
        alphas=(1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
        target_gamma=(1.0, 2.0, 1.0, 1.0),
        mesh_density=32,
        noise_std=1e-2,
        seed=42,
        verbose=True,
    )

    mesh_results = run_mesh_refinement_study(
        densities=(16, 24, 32, 48),
        target_gamma=(1.0, 2.0, 1.0, 1.0),
        alpha=1e-4,
        verbose=True,
    )

    table_files = export_variant_tables(
        results, tables_dir, lcurve_results=lcurve_results, mesh_results=mesh_results, export_png_tables=True
    )
    fig_files = plot_variant_results(
        results, figures_dir, lcurve_results=lcurve_results, mesh_results=mesh_results
    )

    logger.info("Artefactos de tablas generados (%d): %s", len(table_files), [p.name for p in table_files])
    logger.info("Figuras esenciales generadas (%d): %s", len(fig_files), [p.name for p in fig_files])
    logger.info("=== EXPERIMENTACIÓN FINALIZADA. SALIDAS EN: %s ===", out_root)


if __name__ == "__main__":
    main()
