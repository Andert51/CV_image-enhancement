"""
Interfaz de linea de comandos (CLI) para el proyecto de Mejora de Imagenes.

Construida con Typer, expone los siguientes subcomandos:
    histogram   - Ejecuta el modulo de analisis de histogramas y ecualizacion.
    transform   - Ejecuta el modulo de transformaciones puntuales.
    colors      - Ejecuta el modulo de conversion y analisis de espacios de color.
    all         - Ejecuta los tres modulos en secuencia completa.

Uso:
    python main.py histogram --input-dir data/input --output-dir data/output
    python main.py transform --gamma 0.5 --c-gamma 1.0
    python main.py colors
    python main.py all --gamma 1.5
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from src.color_spaces import run_color_analysis
from src.histogram_ops import run_histogram_analysis
from src.transformations import run_transformations
from src.utils import configure_logging

# -------------------------------------------------------------------------
# Instancia principal de la aplicacion Typer
# -------------------------------------------------------------------------

app: typer.Typer = typer.Typer(
    name="cv-image-enhancement",
    help=(
        "Pipeline de procesamiento y mejora de imagenes. "
        "Incluye analisis de histogramas, transformaciones puntuales "
        "y conversion entre espacios de color."
    ),
    add_completion=False,
    rich_markup_mode="markdown",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Valores por defecto centralizados
# -------------------------------------------------------------------------

DEFAULT_INPUT_DIR: Path = Path("data") / "DIP3E_Original_Images_CH03"
DEFAULT_OUTPUT_DIR: Path = Path("data") / "output"
DEFAULT_CLIP_LIMIT: float = 2.0
DEFAULT_TILE_ROWS: int = 8
DEFAULT_TILE_COLS: int = 8
DEFAULT_C_LOG: float = 255.0 / __import__("math").log1p(255.0)
DEFAULT_C_GAMMA: float = 1.0
DEFAULT_GAMMA: float = 1.5


# -------------------------------------------------------------------------
# Callback raiz: configuracion global de logging
# -------------------------------------------------------------------------


@app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Habilita el modo de registro detallado (DEBUG).",
    ),
) -> None:
    """Configura el nivel de registro antes de ejecutar cualquier subcomando."""
    level: int = logging.DEBUG if verbose else logging.INFO
    configure_logging(level)


# -------------------------------------------------------------------------
# Subcomando: histogram
# -------------------------------------------------------------------------


@app.command("histogram")
def cmd_histogram(
    input_dir: Path = typer.Option(
        DEFAULT_INPUT_DIR,
        "--input-dir",
        "-i",
        help="Directorio que contiene las imagenes de entrada.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directorio raiz para los archivos de salida.",
        resolve_path=True,
    ),
    clip_limit: float = typer.Option(
        DEFAULT_CLIP_LIMIT,
        "--clip-limit",
        help=(
            "Limite de recorte para CLAHE. Controla la amplificacion maxima "
            "del contraste local. Rango recomendado: [1.0, 4.0]."
        ),
    ),
    tile_rows: int = typer.Option(
        DEFAULT_TILE_ROWS,
        "--tile-rows",
        help="Numero de filas en la cuadricula de teselas para CLAHE.",
    ),
    tile_cols: int = typer.Option(
        DEFAULT_TILE_COLS,
        "--tile-cols",
        help="Numero de columnas en la cuadricula de teselas para CLAHE.",
    ),
) -> None:
    """Analisis de histogramas y ecualizacion (Global y CLAHE).

    Procesa todas las imagenes en el directorio de entrada y genera:
    - Imagenes ecualizadas (metodo global y CLAHE).
    - Figuras comparativas con histogramas y CDF superpuestas.
    """
    if clip_limit <= 0:
        typer.echo(
            "Error: --clip-limit debe ser un valor positivo.", err=True
        )
        raise typer.Exit(code=1)
    if tile_rows <= 0 or tile_cols <= 0:
        typer.echo(
            "Error: --tile-rows y --tile-cols deben ser enteros positivos.", err=True
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"[histogram] Directorio de entrada: {input_dir}\n"
        f"[histogram] Directorio de salida:  {output_dir}\n"
        f"[histogram] CLAHE clip_limit={clip_limit}, "
        f"tile_grid=({tile_rows},{tile_cols})"
    )

    try:
        run_histogram_analysis(
            input_dir=input_dir,
            output_dir=output_dir,
            clip_limit=clip_limit,
            tile_grid_size=(tile_cols, tile_rows),
        )
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Error inesperado en el modulo de histogramas.")
        raise typer.Exit(code=2) from exc

    typer.echo("[histogram] Pipeline completado exitosamente.")


# -------------------------------------------------------------------------
# Subcomando: transform
# -------------------------------------------------------------------------


@app.command("transform")
def cmd_transform(
    input_dir: Path = typer.Option(
        DEFAULT_INPUT_DIR,
        "--input-dir",
        "-i",
        help="Directorio que contiene las imagenes de entrada.",
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directorio raiz para los archivos de salida.",
        resolve_path=True,
    ),
    c_log: float = typer.Option(
        DEFAULT_C_LOG,
        "--c-log",
        help=(
            "Constante c para la transformacion logaritmica s = c*log(1+r). "
            "Si se especifica 0, se usa el escalado automatico a [0, 255]."
        ),
    ),
    c_gamma: float = typer.Option(
        DEFAULT_C_GAMMA,
        "--c-gamma",
        help="Constante c para la correccion gamma s = c * r^gamma.",
    ),
    gamma: float = typer.Option(
        DEFAULT_GAMMA,
        "--gamma",
        "-g",
        help=(
            "Exponente gamma. "
            "gamma < 1: expande sombras. "
            "gamma > 1: expande luces. "
            "gamma = 1: identidad."
        ),
    ),
) -> None:
    """Transformaciones puntuales de intensidad: lineal, logaritmica y gamma.

    Genera para cada imagen:
    - Imagenes resultantes de cada transformacion en formato PNG.
    - Cuadricula comparativa 2x5 con curvas T(r) y histogramas.
    """
    if gamma <= 0:
        typer.echo("Error: --gamma debe ser un valor positivo.", err=True)
        raise typer.Exit(code=1)
    if c_gamma <= 0:
        typer.echo("Error: --c-gamma debe ser un valor positivo.", err=True)
        raise typer.Exit(code=1)

    c_log_eff: float = c_log if c_log > 0 else DEFAULT_C_LOG

    typer.echo(
        f"[transform] Directorio de entrada: {input_dir}\n"
        f"[transform] Directorio de salida:  {output_dir}\n"
        f"[transform] c_log={c_log_eff:.4f}, c_gamma={c_gamma:.4f}, "
        f"gamma={gamma:.4f}"
    )

    try:
        run_transformations(
            input_dir=input_dir,
            output_dir=output_dir,
            c_log=c_log_eff,
            c_gamma=c_gamma,
            gamma=gamma,
        )
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Error inesperado en el modulo de transformaciones.")
        raise typer.Exit(code=2) from exc

    typer.echo("[transform] Pipeline completado exitosamente.")


# -------------------------------------------------------------------------
# Subcomando: colors
# -------------------------------------------------------------------------


@app.command("colors")
def cmd_colors(
    input_dir: Path = typer.Option(
        DEFAULT_INPUT_DIR,
        "--input-dir",
        "-i",
        help="Directorio que contiene las imagenes de entrada.",
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directorio raiz para los archivos de salida.",
        resolve_path=True,
    ),
) -> None:
    """Conversion y analisis de espacios de color: HSV, YCbCr y CIE L*a*b*.

    Para cada imagen genera:
    - Canales individuales exportados como PNG en escala de grises.
    - Figuras comparativas con histogramas por canal.
    """
    typer.echo(
        f"[colors] Directorio de entrada: {input_dir}\n"
        f"[colors] Directorio de salida:  {output_dir}"
    )

    try:
        run_color_analysis(
            input_dir=input_dir,
            output_dir=output_dir,
        )
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Error inesperado en el modulo de espacios de color.")
        raise typer.Exit(code=2) from exc

    typer.echo("[colors] Pipeline completado exitosamente.")


# -------------------------------------------------------------------------
# Subcomando: all (pipeline completo)
# -------------------------------------------------------------------------


@app.command("all")
def cmd_all(
    input_dir: Path = typer.Option(
        DEFAULT_INPUT_DIR,
        "--input-dir",
        "-i",
        help="Directorio que contiene las imagenes de entrada.",
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directorio raiz para los archivos de salida.",
        resolve_path=True,
    ),
    clip_limit: float = typer.Option(
        DEFAULT_CLIP_LIMIT,
        "--clip-limit",
        help="Limite de recorte para CLAHE.",
    ),
    tile_rows: int = typer.Option(DEFAULT_TILE_ROWS, "--tile-rows"),
    tile_cols: int = typer.Option(DEFAULT_TILE_COLS, "--tile-cols"),
    c_log: float = typer.Option(
        DEFAULT_C_LOG,
        "--c-log",
        help="Constante c para la transformacion logaritmica.",
    ),
    c_gamma: float = typer.Option(
        DEFAULT_C_GAMMA,
        "--c-gamma",
        help="Constante c para la correccion gamma.",
    ),
    gamma: float = typer.Option(
        DEFAULT_GAMMA,
        "--gamma",
        "-g",
        help="Exponente gamma.",
    ),
) -> None:
    """Ejecuta el pipeline completo: histogramas, transformaciones y espacios de color.

    Equivale a ejecutar los subcomandos histogram, transform y colors
    de forma secuencial sobre el mismo directorio de entrada.
    """
    typer.echo(
        "=================================================================\n"
        "Iniciando pipeline completo de Mejora de Imagenes\n"
        f"  Entrada:  {input_dir}\n"
        f"  Salida:   {output_dir}\n"
        "================================================================="
    )

    errors: list[str] = []

    # --- Modulo 1: Histogramas ---
    typer.echo("\n[1/3] Modulo: Analisis de Histogramas y Ecualizacion")
    try:
        run_histogram_analysis(
            input_dir=input_dir,
            output_dir=output_dir,
            clip_limit=clip_limit,
            tile_grid_size=(tile_cols, tile_rows),
        )
        typer.echo("      Completado.")
    except Exception as exc:
        msg = f"Error en modulo de histogramas: {exc}"
        logger.error(msg)
        errors.append(msg)

    # --- Modulo 2: Transformaciones ---
    typer.echo("\n[2/3] Modulo: Transformaciones Puntuales de Intensidad")
    c_log_eff: float = c_log if c_log > 0 else DEFAULT_C_LOG
    try:
        run_transformations(
            input_dir=input_dir,
            output_dir=output_dir,
            c_log=c_log_eff,
            c_gamma=c_gamma,
            gamma=gamma,
        )
        typer.echo("      Completado.")
    except Exception as exc:
        msg = f"Error en modulo de transformaciones: {exc}"
        logger.error(msg)
        errors.append(msg)

    # --- Modulo 3: Espacios de Color ---
    typer.echo("\n[3/3] Modulo: Conversion y Analisis de Espacios de Color")
    try:
        run_color_analysis(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        typer.echo("      Completado.")
    except Exception as exc:
        msg = f"Error en modulo de espacios de color: {exc}"
        logger.error(msg)
        errors.append(msg)

    typer.echo(
        "\n=================================================================\n"
        f"Pipeline finalizado. Errores encontrados: {len(errors)}\n"
        "================================================================="
    )

    if errors:
        typer.echo("\nResumen de errores:", err=True)
        for err in errors:
            typer.echo(f"  - {err}", err=True)
        raise typer.Exit(code=1)
