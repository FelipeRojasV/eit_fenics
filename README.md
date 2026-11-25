# EIT FEniCS Legacy Suite

Biblioteca modular desarrollada en Python con **FEniCS Legacy** (`dolfin` 2019.1.0), orientada a la resolución y el análisis numérico de problemas directos e inversos de **Tomografía de Impedancia Eléctrica (EIT)**.

El desarrollo sigue el marco matemático presentado por Habib Ammari, Jin Keun Seo y Eung Je Woo en *Mathematical Modeling in Biomedical Imaging I*.

## Requisitos

Esta biblioteca está diseñada para funcionar exclusivamente con **FEniCS Legacy**. No cuenta con un modo alternativo basado en soluciones analíticas simplificadas, ya que todos los resultados deben obtenerse mediante el método de elementos finitos (FEM).

Por este motivo, si FEniCS no está instalado o no se encuentra disponible en el entorno de ejecución, la importación de `eit_lib` generará un `ImportError` con instrucciones para instalar las dependencias necesarias.

---

## Instalación

FEniCS Legacy (2019.1.0) no se encuentra disponible mediante `pip`. Por ello, se recomienda instalarlo utilizando Conda y el canal `conda-forge`.

```bash
conda create -n fenics -c conda-forge fenics=2019.1.0 matplotlib scipy numpy pytest
conda activate fenics
pip install -e .
