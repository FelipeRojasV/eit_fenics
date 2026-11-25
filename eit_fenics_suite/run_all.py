"""
run_all.py
Ejecutor integral de los 5 ejercicios de Tomografía de Impedancia Eléctrica (EIT).
Requiere FEniCS Legacy; aborta explícitamente si no está disponible.
"""

import subprocess
import sys
import os

# Aborta temprano si FEniCS no está instalado.
try:
    import fenics  # noqa: F401
except ImportError:
    sys.stderr.write(
        "\n[ERROR] FEniCS Legacy (dolfin) no está disponible.\n"
        "        Esta suite NO tiene modo fallback.\n"
        "        Instala con:\n"
        "          conda create -n fenics -c conda-forge fenics=2019.1.0 "
        "matplotlib scipy numpy pytest\n"
        "          conda activate fenics\n\n"
    )
    sys.exit(1)

scripts = [
    "exercises/ex1_forward_inverse_1_1.py",
    "exercises/ex2_reciprocity.py",
    "exercises/ex3_least_squares.py",
    "exercises/ex4_linearized_svd.py",
    "exercises/ex5_two_channel_eit.py",
    "analysis/benchmark_analysis.py",
]


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    os.makedirs("plots", exist_ok=True)

    print("\n" + "=" * 70)
    print(" INICIANDO EJECUCIÓN DE LA SUITE MODULAR EIT (FENICS LEGACY)")
    print("=" * 70 + "\n")

    success_count = 0
    for script in scripts:
        print(f"\n>>> Ejecutando {script} ...")
        script_path = os.path.join(root_dir, script)
        ret = subprocess.run([sys.executable, script_path], capture_output=False)
        if ret.returncode == 0:
            success_count += 1
        else:
            print(f"[ERROR] Falló la ejecución de {script}")

    print("\n" + "=" * 70)
    print(f" FINALIZADO: {success_count}/{len(scripts)} ejercicios ejecutados con éxito.")
    print(" Las imágenes generadas se encuentran en la carpeta 'plots/':")
    if os.path.exists("plots"):
        for f in os.listdir("plots"):
            if f.endswith(".png"):
                print(f"  - plots/{f}")
    if os.path.exists("results"):
        for f in os.listdir("results"):
         print(f"  - results/{f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
