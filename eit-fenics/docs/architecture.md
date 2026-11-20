# Arquitectura del Software `eit-fenics`

La biblioteca implementa un flujo desacoplado y rigurosamente fundamentado en la formulación variacional de elementos finitos:

1. `geometry`: Discretización del dominio $[-1, 1] \times [-1, 1]$ con malla de Lagrange P1 y partición en 4 cuadrantes.
2. `materials`: Asignación del vector de parámetros $\gamma \in \mathbb{R}^4$ sobre elementos $DG_0$.
3. `forward`: Solución directa del problema elíptico bajo calibración de potencial nulo en $E_0$.
4. `measurements`: Mapeo frontera y lectura de voltajes con conservación de reciprocidad.
5. `reconstruction`: Algoritmo iterativo Gauss-Newton con cálculo variacional del Jacobiano y regularización de Tikhonov.
6. `visualization`: Renderizado de campos escalares, vectoriales de corriente $\mathbf{J} = -\sigma \nabla u$, mapas de sensibilidad y evolución de convergencia.
7. `utils`: Verificación analítica y matricial de simetría física.
