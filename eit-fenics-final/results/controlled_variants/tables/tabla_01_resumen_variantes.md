# Tabla 1: Resumen Cuantitativo de Variantes Controladas (EIT 4 Canales)

| Variante | Factor | Malla | $\alpha$ | $\sigma_{\text{ruido}}$ | Iter. | $\gamma^*$ | $\hat{\gamma}$ | Error $\gamma$ [%] | Error NtD [%] | $\max|F-F^T|$ | Tiempo [s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `V0_base_referencia` | linea_base | 32x32 | 1.0e-04 | 0.0e+00 | 5 | `[1.00, 2.00, 1.00, 1.00]` | `[1.000, 2.000, 1.000, 1.000]` | 0.00% | 0.000% | 2.22e-16 | 0.49 |
| `V1_medio_homogeneo` | distribucion_conductividad | 32x32 | 1.0e-04 | 0.0e+00 | 1 | `[1.00, 1.00, 1.00, 1.00]` | `[1.000, 1.000, 1.000, 1.000]` | 0.00% | 0.000% | 1.67e-16 | 0.27 |
| `V2_contraste_multicuadrante` | distribucion_conductividad | 32x32 | 1.0e-04 | 0.0e+00 | 5 | `[1.60, 0.80, 1.30, 1.90]` | `[1.600, 0.800, 1.300, 1.900]` | 0.00% | 0.000% | 2.78e-16 | 0.47 |
| `V3_anomalia_aislante_q4` | distribucion_conductividad | 32x32 | 1.0e-04 | 0.0e+00 | 8 | `[1.00, 1.00, 1.00, 0.30]` | `[1.000, 1.000, 1.000, 0.300]` | 0.00% | 0.000% | 3.33e-16 | 0.61 |
| `V4_reg_baja_ruido` | regularizacion_y_ruido | 32x32 | 1.0e-06 | 1.0e-02 | 5 | `[1.00, 2.00, 1.00, 1.00]` | `[0.951, 2.020, 1.052, 0.995]` | 2.80% | 0.891% | 2.78e-16 | 0.48 |
| `V5_reg_alta_ruido` | regularizacion_y_ruido | 32x32 | 1.0e-02 | 1.0e-02 | 10 | `[1.00, 2.00, 1.00, 1.00]` | `[0.951, 2.020, 1.052, 0.995]` | 2.80% | 0.891% | 2.78e-16 | 0.70 |
| `V6_malla_fina` | refinamiento_malla | 48x48 | 1.0e-04 | 0.0e+00 | 5 | `[1.00, 2.00, 1.00, 1.00]` | `[1.000, 2.000, 1.000, 1.000]` | 0.00% | 0.000% | 2.22e-16 | 1.25 |
| `V7_referencia_seo_woo` | datos_referencia_libro | 32x32 | 1.0e-04 | 0.0e+00 | 9 | `N/A` | `[0.379, 0.635, 0.625, 1.008]` | N/A | 2.684% | 4.44e-16 | 0.65 |
