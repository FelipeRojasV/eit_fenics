"""
Script Utilitario para Generar Tablas y Figuras Diagnósticas de EIT.

Genera automáticamente:
1. Tablas en Markdown (.md) para informes rápidos y Notion/GitHub.
2. Tablas en LaTeX (.tex) con formato booktabs para la memoria.
3. Tablas exportadas como Figuras (.png) de alta resolución.
4. Figura diagnóstica del log de convergencia (diagnostico_log_convergencia.png).
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.tables import EITTableGenerator


def main():
    print("======================================================================")
    print("  GENERACIÓN AUTOMÁTICA DE TABLAS Y FIGURAS DIAGNÓSTICAS DE EIT")
    print("======================================================================")

    tables_dir = os.path.join(ROOT_DIR, "results", "tables")
    figures_dir = os.path.join(ROOT_DIR, "results", "figures")
    log_file = os.path.join(ROOT_DIR, "results", "logs", "eit_experiment.log")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Destino de tablas (.md / .tex): {tables_dir}")
    print(f"Destino de figuras (.png):       {figures_dir}")
    print(f"Archivo de log a procesar:      {log_file}\n")

    # Exportar tablas a tablas_dir y figuras a figures_dir
    generated_files = EITTableGenerator.export_all(
        output_dir=tables_dir,
        figures_dir=figures_dir,
        results_data={
            "log_file": log_file if os.path.exists(log_file) else None
        },
        export_images=True
    )

    print(f"\n[OK] Se generaron {len(generated_files)} artefactos exitosamente:")
    for f in generated_files:
        rel_path = os.path.relpath(f, ROOT_DIR)
        print(f"  -> {rel_path}")

    print("\nTablas y figuras listas para incorporar en el documento final de tesis.")


if __name__ == "__main__":
    main()
