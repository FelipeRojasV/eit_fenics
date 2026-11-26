# EIT FEniCS Legacy Suite

Biblioteca modular en Python sobre FEniCS Legacy (dolfin 2019.1.0) para la
resolución y el análisis numérico de problemas directos e inversos en
Tomografía de Impedancia Eléctrica (EIT). El desarrollo sigue el marco
matemático de Ammari, Seo y Woo en *Mathematical Modeling in Biomedical
Imaging I*, y cubre los ejercicios 1 a 4 del capítulo: problema directo e
inverso en un sistema de cuatro electrodos, verificación de la relación de
reciprocidad, reconstrucción por mínimos cuadrados no lineales y
reconstrucción linealizada mediante descomposición en valores singulares.

El caso de conductividad compleja (Two-Channel EIT, Example 1.5 del texto)
no forma parte de esta versión. Se trata de un problema de naturaleza
espectroscópica que se aparta del objetivo central de reconstrucción de la
conductividad en el dominio.

La suite requiere FEniCS Legacy y no incluye modo fallback. Si FEniCS no
está disponible, cualquier importación de `eit_lib` lanza `ImportError`.

## Instalación

    conda create -n fenics -c conda-forge fenics=2019.1.0 matplotlib scipy numpy pytest
    conda activate fenics
    pip install -e .

## Estructura

    eit_fenics_suite/
    ├── eit_lib/                  Núcleo de la biblioteca
    ├── exercises/                Ejercicios 1 a 4
    ├── analysis/                 Módulo de tablas y figuras
    ├── tests/                    Pruebas con pytest
    ├── results/                  Tablas .csv, .md y .tex (+ raw/)
    ├── plots/                    Figuras .png a 300 dpi
    ├── run_all.py
    ├── pyproject.toml
    └── README.md

## Uso

    python run_all.py
    python analysis/benchmark_analysis.py --only table1
    python analysis/benchmark_analysis.py --only table3
    python analysis/benchmark_analysis.py --only gn

Secciones: `table1`, `table2`, `table3`, `gn`, `tsvd`, `all`.

## Salidas

Tablas en `results/` (`.csv`, `.md`, `.tex`):

- `tabla1_resumen_verificacion`: residuo, reciprocidad, error GN y cond(A) para Ex1–Ex4.
- `tabla2_pruebas_unitarias`: matriz de tests vía API de pytest.
- `tabla3_desempeno_malla`: DOFs, reciprocidad y tiempos para mallas 16, 32, 64 y 128.

Figuras en `plots/` (300 dpi):

- `analysis_mesh_convergence.png`
- `analysis_gn_regularization_noise.png`
- `analysis_tsvd_truncation.png`

Los ejercicios guardan sus crudos en `results/raw/exK.npz`; el análisis los lee.
Si faltan, ejecuta el ejercicio correspondiente antes de construir la tabla.

## Pruebas

    pytest -v

Se omiten automáticamente si FEniCS no está disponible.
