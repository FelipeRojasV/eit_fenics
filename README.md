# EIT FEniCS

Electrical Impedance Tomography (EIT) solver, sensitivity matrices, and reconstruction algorithms using FEniCS legacy.

## Requisitos y Entorno
Este proyecto utiliza FEniCS Legacy (versión 2019.1.0) ejecutándose en un entorno aislado de Conda.

### Configuración del entorno
```bash
conda create -n fenics-env python=3.9 -y
conda activate fenics-env
conda install -c conda-forge fenics=2019.1.0 matplotlib scipy -y
