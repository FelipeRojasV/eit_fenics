# eit-fenics: Biblioteca Modular para Tomografía de Impedancia Eléctrica (EIT)

`eit-fenics` es una biblioteca modular en Python diseñada para simular y resolver el problema directo e inverso de **Tomografía de Impedancia Eléctrica (EIT)** utilizando el **Método de Elementos Finitos (FEM)** con **FEniCS Legacy (DOLFIN 2019.1.0)**.

## Estructura del Proyecto
- `src/geometry`: Definición de malla continua, subdominios (L=4) y electrodos de contorno en FEniCS.
- `src/materials`: Manejo del campo de conductividad constante por trozos en espacio discontinuo `DG_0`.
- `src/forward`: Solucionador variacional de la EDP elíptica $\nabla \cdot (\sigma \nabla u_j) = 0$.
- `src/measurements`: Extracción de potenciales promediados y construcción de la matriz reducida Neumann-to-Dirichlet (NtD).
- `src/reconstruction`: Inversor de Gauss-Newton regularizado con Tikhonov y cálculo del Jacobiano mediante estado adjunto.
- `src/visualization`: Suite completa de visualización científica y diagnóstico paramétrico.
- `src/utils`: Herramientas de soporte y verificación de reciprocidad física.

## Ejecución de Pruebas Unitarias
```bash
pytest tests/test_pipeline.py -v
```

## Ejecución del Pipeline Completo
```bash
python examples/solve_problem_1_1.py
```
Los resultados visuales se guardan automáticamente en `results/figures/` y el registro detallado en `results/logs/eit_experiment.log`.
