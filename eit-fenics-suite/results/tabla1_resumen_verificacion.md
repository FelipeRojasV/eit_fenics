### Resumen de verificación numérica y errores de reconstrucción (Ejercicios 1 a 4).

| Experimento | Configuración | Métrica clave | Valor obtenido | Observación |
| --- | --- | --- | --- | --- |
| Ex 1: Problema 1.1 | Malla 32x32, alpha=1e-3 | Residuo linealizado relativo | 7.5944e-03 | sigma_recon = [3.084, 3.572, 4.261, 3.454] |
| Ex 2: Reciprocidad | Malla 36x36, sigma gaussiana | max|F-F^T| / ||F-F^T||_F | 8.88e-16 / 1.29e-15 | Simetria verificada (N_indep=6) |
| Ex 3: Gauss-Newton | Malla 32x32, alpha=1e-4, ruido=0.005 | Error relativo sigma | 1.43% | Convergencia en 6 iteraciones |
| Ex 4: TSVD | Malla 32x32, r=4, ruido=0.002 | cond(A) / Error relativo | 7.79 / 0.55% | s_1=1.578e+00, s_4=2.026e-01 |
