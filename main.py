"""
Punto de entrada principal del proyecto de Mejora de Imagenes.

Delega el control a la instancia Typer definida en src/cli.py.
Ejecutar con:
    python main.py --help
    python main.py all
    python main.py histogram --clip-limit 3.0
    python main.py transform --gamma 0.5
    python main.py colors
"""

from __future__ import annotations

import sys


def _check_python_version() -> None:
    """Verifica que la version de Python sea >= 3.10.

    Raises:
        SystemExit: Si la version de Python es inferior a 3.10.
    """
    if sys.version_info < (3, 10):
        print(
            f"Error: Se requiere Python 3.10 o superior. "
            f"Version actual: {sys.version_info.major}.{sys.version_info.minor}",
            file=sys.stderr,
        )
        sys.exit(1)


_check_python_version()

from src.cli import app  # noqa: E402 -- importacion post-validacion de version

if __name__ == "__main__":
    app()
