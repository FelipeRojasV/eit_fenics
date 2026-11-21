# Suite de Variantes Controladas con Representaciones 3D para `eit-fenics` (v3)

Extiende `eit-fenics` **sin modificar ningún módulo existente** (`geometry`, `materials`, `forward`, `measurements`, `reconstruction`, `visualization`, `utils` ni `examples/solve_problem_1_1.py`). Todas las corridas utilizan de forma nativa **FEniCS Legacy** en tu entorno `fenics_legacy`.

## 5 Figuras Esenciales (`results/controlled_variants/figures/`)

1. **`var_01_reconstruccion_3d_variantes.png` (3D):** Prismas 3D de conductividad reconstruida $\hat{\gamma}$ por cuadrante ($\Omega_1 \dots \Omega_4$) para cuatro escenarios clave (`V0_base_referencia`, `V2_contraste_multicuadrante`, `V3_anomalia_aislante_q4` y `V7_referencia_seo_woo`).
2. **`var_02_superficie_3d_potenciales.png` (3D):** Superficies 3D (`plot_trisurf` sobre la malla FEniCS con contornos proyectados en el plano base) del potencial eléctrico reconstruido $u_1^{\hat{\gamma}}(x,y)$ y su perturbación $\delta u_1 = u_1^{\hat{\gamma}} - u_1^{\gamma_0}$ (Canal 1).
3. **`var_03_convergencia_y_errores.png` (2D):** Panel 2x2 con convergencia logarítmica del residuo de las 8 variantes, errores relativos en $\gamma$ y Baby NtD, y trayectorias iterativas de $\hat{\gamma}_m$.
4. **`var_04_ntd_3d_y_reciprocidad.png` (3D + 2D):** Barras 3D de la matriz *Baby NtD* observada vs. reconstruida ($|\mathbb{F}_{\mathrm{obs}}|$, $|\mathbb{F}_{\mathrm{sim}}|$), verificación del principio de reciprocidad ($\max|\mathbb{F}-\mathbb{F}^T|$) y número de condición del Jacobiano $\kappa(A)$.
5. **`var_05_regularizacion_y_malla.png` (2D):** Estudio paramétrico consolidado: Curva L de Tikhonov, compromiso sesgo-varianza bajo ruido, estabilidad de $\hat{\gamma}_m$ al refinar la malla (`16x16` a `48x48`) y tiempo de ejecución.

## 5 Tablas Científicas + 3 CSV (`results/controlled_variants/tables/`)
Cada tabla se exporta en `.md`, `.tex` (LaTeX) y `.png`:
- `tabla_01_resumen_variantes`
- `tabla_02_detalle_cuadrantes`
- `tabla_03_condicionamiento_y_reciprocidad`
- `tabla_04_estudio_regularizacion_lcurve`
- `tabla_05_estudio_refinamiento_malla`
- `resultados_variantes_controladas.csv`, `resultados_lcurve.csv`, `resultados_refinamiento_malla.csv`

## Comandos de actualización y ejecución en Linux Mint (`fenics_legacy`)

```bash
cd /home/present_day/eit-fenics-10
unzip -o ~/Descargas/eit-fenics_variantes_controladas_v3_3d.zip -d /tmp/addon_eit_v3
cp -rf /tmp/addon_eit_v3/eit-fenics_variantes_controladas_v3_3d/. .
rm -rf /tmp/addon_eit_v3

pytest tests/test_pipeline.py tests/test_controlled_variants.py -v
python examples/run_controlled_variants.py
ls -lh results/controlled_variants/figures/ results/controlled_variants/tables/
```
