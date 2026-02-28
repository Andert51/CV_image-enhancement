"""
Paquete principal del proyecto de Mejora de Imagenes.

Modulos disponibles:
    histogram_ops   - Analisis de histogramas y ecualizacion.
    transformations - Transformaciones puntuales de intensidad.
    color_spaces    - Conversion y analisis de espacios de color.
    cli             - Interfaz de linea de comandos (CLI).

Version: 1.0.0
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "utils",
    "histogram_ops",
    "transformations",
    "color_spaces",
    "cli",
]
