"""
Utilidades de E/S para resultados crudos en results/raw/*.npz.
Centraliza rutas y nombres de arrays para que ejercicios y análisis
hablen el mismo idioma.
"""

import os
import numpy as np

RAW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "raw")
)


def raw_path(name):
    """Ruta absoluta a results/raw/<name>.npz."""
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, f"{name}.npz")


def save_raw(name, arrays):
    """Guarda un dict de arrays numpy en results/raw/<name>.npz."""
    np.savez(raw_path(name), **arrays)
    return raw_path(name)


def load_raw(name):
    """Carga results/raw/<name>.npz como dict de numpy arrays. None si no existe."""
    path = raw_path(name)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def raw_exists(name):
    return os.path.exists(raw_path(name))