"""
Ejecuta exercises/*.py y analysis/*.py en orden alfabético.
"""

import glob
import os
import subprocess
import sys

try:
    import fenics  # noqa: F401
except ImportError:
    sys.stderr.write(
        "\n[ERROR] FEniCS Legacy (dolfin) no disponible. "
        "Instala con conda (fenics=2019.1.0).\n"
    )
    sys.exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))


def discover(folder):
    return sorted(glob.glob(os.path.join(ROOT, folder, "*.py")))


def main():
    os.chdir(ROOT)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "raw"), exist_ok=True)

    scripts = discover("exercises") + discover("analysis")

    print("\n" + "=" * 70)
    print(" SUITE EIT — FENICS LEGACY")
    print("=" * 70)

    ok = 0
    for path in scripts:
        rel = os.path.relpath(path, ROOT)
        print(f"\n>>> {rel}")
        ret = subprocess.run([sys.executable, path], cwd=ROOT)
        if ret.returncode == 0:
            ok += 1
        else:
            print(f"[ERROR] Falló: {rel}")

    print("\n" + "=" * 70)
    print(f" FINALIZADO: {ok}/{len(scripts)} OK")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
