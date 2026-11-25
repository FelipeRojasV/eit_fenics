# \# EIT FEniCS Legacy Suite

# 

# Biblioteca modular en Python basada en \*\*FEniCS Legacy\*\* (`dolfin` 2019.1.0)

# para la resolución y análisis numérico de problemas directos e inversos en

# \*\*Tomografía de Impedancia Eléctrica (EIT)\*\*, siguiendo el marco matemático

# de Habib Ammari, Jin Keun Seo y Eung Je Woo (\*Mathematical Modeling in

# Biomedical Imaging I\*).

# 

# > \*\*Esta suite requiere FEniCS Legacy.\*\* No incluye modo fallback analítico:

# > los resultados deben generarse siempre con FEM real. Si FEniCS no está

# > instalado, cualquier import de `eit\_lib` lanza `ImportError` con

# > instrucciones de instalación.

# 

# \---

# 

# \## Instalación

# 

# FEniCS Legacy (2019.1.0) \*\*no es instalable vía pip\*\*. Se recomienda conda:

# 

# ```bash

# conda create -n fenics -c conda-forge fenics=2019.1.0 matplotlib scipy numpy pytest

# conda activate fenics

# pip install -e .

