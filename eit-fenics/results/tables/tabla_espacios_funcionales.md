| Variable Física | Símbolo | Espacio Continuo | Espacio Discreto (FEniCS) | Base Polinomial | Dimensión / DoFs |
| --- | --- | --- | --- | --- | --- |
| Potencial Eléctrico | $u_j$ | $H^1(\Omega)$ | FunctionSpace(mesh, 'P', 1) | Lagrange lineal continuo | 1089 nodos |
| Conductividad Cuadrantes | $\sigma$ | $L^\infty(\Omega)$ | FunctionSpace(mesh, 'DG', 0) | Constante por trozos | 4 subdominios (L=4) |
| Densidad Corriente Normal | $g_j = -I/|\mathcal{E}_j|$ | $H^{-1/2}(\partial\Omega)$ | Measure('ds', subdomain_data) | Constante por electrodo | 3 canales activos |
| Flujo de Corriente | $\mathbf{J} = -\sigma \nabla u$ | $[L^2(\Omega)]^2$ | VectorFunctionSpace(mesh, 'P', 1) | Vectorial proyectado | 2178 componentes |
| Malla de Dominio | $\Omega = [-1,1]^2$ | $\mathbb{R}^2$ | RectangleMesh | Triángulos estructurados | 2048 celdas |
