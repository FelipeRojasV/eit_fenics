### Desempeño computacional y convergencia geométrica en función de la resolución de malla.

| Malla (nx x ny) | Celdas | DOFs CG(1) | max|F - F^T| | C_Omega | t_forward [s] | t_jacobiano [s] | t_total [s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 x 16 | 512 | 289 | 3.33e-16 | 1.89455 | 0.0250 | 0.0211 | 0.0602 |
| 32 x 32 | 2048 | 1089 | 4.44e-16 | 1.77131 | 0.0273 | 0.0186 | 0.0667 |
| 64 x 64 | 8192 | 4225 | 7.77e-16 | 1.79800 | 0.1185 | 0.0472 | 0.2314 |
| 128 x 128 | 32768 | 16641 | 3.33e-16 | 1.80979 | 0.4454 | 0.1170 | 0.9118 |
