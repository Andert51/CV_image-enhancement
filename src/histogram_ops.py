"""
Modulo de analisis de histogramas y ecualizacion de imagenes.

Implementa las siguientes operaciones:
    - Calculo del histograma de intensidades y su funcion de densidad
      de probabilidad (PDF) acumulada (CDF).
    - Ecualizacion de histograma global mediante transformacion de la CDF.
    - Ecualizacion de histograma adaptativa limitada por contraste (CLAHE).
    - Generacion de figuras comparativas de calidad publicable.

Fundamentacion Matematica:
    La ecualizacion de histograma produce una transformacion T tal que
    la imagen de salida s = T(r) tenga una distribucion de intensidades
    aproximadamente uniforme. La funcion de transformacion es:

        T(r_k) = (L - 1) * sum_{j=0}^{k} p_r(r_j)

    donde p_r(r_j) = n_j / N es la probabilidad estimada de la intensidad r_j,
    n_j es el numero de pixeles con dicha intensidad, N es el total de pixeles
    y L = 256 para imagenes de 8 bits.

Referencias:
    Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing
    (4th ed.). Pearson. Cap. 3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.utils import (
    ensure_output_dir,
    list_images,
    load_image_gray,
    save_image,
    validate_grayscale,
)

matplotlib.use("Agg")  # Backend no interactivo para generacion de archivos.

logger = logging.getLogger(__name__)

_FIGURE_DPI: int = 150
_FIGURE_FONT_SIZE: int = 9
_HIST_COLOR_ORIGINAL: str = "#2C3E50"
_HIST_COLOR_GLOBAL: str = "#2980B9"
_HIST_COLOR_CLAHE: str = "#E74C3C"


# ---------------------------------------------------------------------------
# Funciones Primitivas
# ---------------------------------------------------------------------------


def compute_histogram(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula el histograma normalizado (PDF) de una imagen en escala de grises.

    Args:
        img: Imagen monocanal uint8 de forma (H, W).

    Returns:
        Tupla (bins, pdf) donde:
            - bins: Arreglo de enteros [0, 255] representando los niveles
                    de intensidad.
            - pdf:  Arreglo float64 con la probabilidad de cada nivel
                    (suma normalizada a 1.0).

    Raises:
        TypeError: Si `img` no es np.ndarray.
        ValueError: Si la imagen no es monocanal uint8.
    """
    validate_grayscale(img)
    total_pixels: int = img.size
    hist_raw: np.ndarray = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    pdf: np.ndarray = hist_raw / total_pixels
    bins: np.ndarray = np.arange(256, dtype=np.int32)
    return bins, pdf


def compute_cdf(pdf: np.ndarray) -> np.ndarray:
    """Calcula la Funcion de Distribucion Acumulada (CDF) a partir de la PDF.

    Args:
        pdf: Arreglo float64 de longitud 256 con la densidad de probabilidad.

    Returns:
        Arreglo float64 de longitud 256 con la CDF normalizada en [0, 1].

    Raises:
        ValueError: Si `pdf` no tiene exactamente 256 elementos.
    """
    if pdf.shape != (256,):
        raise ValueError(
            f"La PDF debe tener exactamente 256 elementos, se recibio: {pdf.shape}"
        )
    cdf: np.ndarray = np.cumsum(pdf)
    return cdf


def equalize_histogram_global(img: np.ndarray) -> np.ndarray:
    """Aplica ecualizacion de histograma global a una imagen en escala de grises.

    La transformacion es:
        s = round((L - 1) * CDF(r))

    donde L = 256 y la CDF es calculada sobre la imagen completa. Esto
    redistribuye las intensidades hacia una distribucion aproximadamente uniforme.

    Args:
        img: Imagen monocanal uint8 de forma (H, W).

    Returns:
        Imagen ecualizada de mismo shape y dtype uint8.

    Raises:
        TypeError: Si `img` no es np.ndarray.
        ValueError: Si la imagen no es monocanal uint8.
    """
    validate_grayscale(img)
    _, pdf = compute_histogram(img)
    cdf: np.ndarray = compute_cdf(pdf)

    # Tabla de busqueda (LUT): mapea cada intensidad r a s = (L-1) * CDF(r)
    lut: np.ndarray = np.round(255.0 * cdf).astype(np.uint8)
    equalized: np.ndarray = lut[img]

    logger.debug(
        "Ecualizacion global aplicada. Entropia original=%.4f bits.",
        _compute_entropy(pdf),
    )
    return equalized


def equalize_histogram_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Aplica Ecualizacion de Histograma Adaptativa Limitada por Contraste (CLAHE).

    CLAHE divide la imagen en teselas (tiles) y aplica ecualizacion local
    a cada una. El parametro `clip_limit` limita la amplificacion del contraste
    para evitar la sobreampificacion del ruido.

    Args:
        img: Imagen monocanal uint8 de forma (H, W).
        clip_limit: Umbral de recorte para la amplificacion del contraste.
            Valores tipicos en [1.0, 4.0]. Un valor de 40.0 equivale
            aproximadamente a la ecualizacion global.
        tile_grid_size: Dimensiones de la cuadricula de teselas (cols, rows).

    Returns:
        Imagen procesada por CLAHE con mismo shape y dtype uint8.

    Raises:
        TypeError: Si `img` no es np.ndarray.
        ValueError: Si la imagen no es monocanal uint8 o parametros invalidos.
    """
    validate_grayscale(img)
    if clip_limit <= 0:
        raise ValueError(
            f"clip_limit debe ser un valor positivo, se recibio: {clip_limit}"
        )
    if len(tile_grid_size) != 2 or any(s <= 0 for s in tile_grid_size):
        raise ValueError(
            f"tile_grid_size debe ser una tupla de dos enteros positivos, "
            f"se recibio: {tile_grid_size}"
        )

    clahe_obj = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    result: np.ndarray = clahe_obj.apply(img)
    logger.debug(
        "CLAHE aplicado con clip_limit=%.2f, tile_grid_size=%s.",
        clip_limit,
        tile_grid_size,
    )
    return result


# ---------------------------------------------------------------------------
# Funciones de Visualizacion
# ---------------------------------------------------------------------------


def _compute_entropy(pdf: np.ndarray) -> float:
    """Calcula la entropia de Shannon de la distribucion de intensidades.

    H = -sum(p * log2(p)) para p > 0

    Args:
        pdf: Arreglo de probabilidades de longitud 256.

    Returns:
        Valor de entropia en bits.
    """
    nonzero_mask: np.ndarray = pdf > 0
    return float(-np.sum(pdf[nonzero_mask] * np.log2(pdf[nonzero_mask])))


def _plot_histogram_on_ax(
    ax: Axes,
    img: np.ndarray,
    title: str,
    color: str,
    show_cdf: bool = True,
) -> None:
    """Dibuja el histograma de intensidades sobre un eje Matplotlib dado.

    Args:
        ax: Eje Matplotlib sobre el cual renderizar.
        img: Imagen monocanal uint8.
        title: Titulo del subplot.
        color: Color de las barras del histograma (formato CSS o hex).
        show_cdf: Si es True, superpone la CDF normalizada como linea.
    """
    _, pdf = compute_histogram(img)
    bins = np.arange(256)

    ax.bar(bins, pdf * 256, color=color, alpha=0.75, width=1.0, label="PDF (x256)")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Nivel de intensidad (r)", fontsize=_FIGURE_FONT_SIZE)
    ax.set_ylabel("Frecuencia relativa", fontsize=_FIGURE_FONT_SIZE)
    ax.set_title(title, fontsize=_FIGURE_FONT_SIZE + 1, fontweight="bold")

    if show_cdf:
        cdf = compute_cdf(pdf)
        ax_twin = ax.twinx()
        ax_twin.plot(
            bins,
            cdf,
            color="black",
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            label="CDF",
        )
        ax_twin.set_ylabel("CDF acumulada", fontsize=_FIGURE_FONT_SIZE)
        ax_twin.set_ylim(0, 1.05)
        ax_twin.tick_params(axis="y", labelsize=_FIGURE_FONT_SIZE - 1)

    entropy_val = _compute_entropy(pdf)
    ax.text(
        0.98,
        0.97,
        f"H={entropy_val:.3f} bits",
        transform=ax.transAxes,
        fontsize=_FIGURE_FONT_SIZE - 1,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )
    ax.tick_params(axis="both", labelsize=_FIGURE_FONT_SIZE - 1)


def generate_comparison_figure(
    original: np.ndarray,
    equalized_global: np.ndarray,
    equalized_clahe: np.ndarray,
    output_path: Path,
    image_name: str = "",
) -> None:
    """Genera y exporta la figura comparativa de ecualizacion de histograma.

    Produce una cuadricula de 2x3 con:
        Fila 1: Imagenes (original, ecualizada global, ecualizada CLAHE).
        Fila 2: Histogramas + CDF de cada imagen.

    Args:
        original: Imagen original en escala de grises uint8.
        equalized_global: Imagen ecualizada con metodo global.
        equalized_clahe: Imagen ecualizada con CLAHE.
        output_path: Ruta de destino del archivo PNG generado.
        image_name: Nombre de la imagen para el titulo de la figura.

    Raises:
        TypeError: Si argumentos no son del tipo esperado.
        IOError: Si la figura no puede guardarse en disco.
    """
    validate_grayscale(original)
    validate_grayscale(equalized_global)
    validate_grayscale(equalized_clahe)
    if not isinstance(output_path, Path):
        raise TypeError(
            f"output_path debe ser pathlib.Path, se recibio: {type(output_path).__name__}"
        )

    fig: Figure
    axes: np.ndarray

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 8),
        dpi=_FIGURE_DPI,
        constrained_layout=True,
    )

    fig.suptitle(
        f"Analisis de Histograma: {image_name}",
        fontsize=_FIGURE_FONT_SIZE + 3,
        fontweight="bold",
    )

    img_titles = ["Original", "Ecualizacion Global", "CLAHE"]
    images_list = [original, equalized_global, equalized_clahe]
    hist_colors = [_HIST_COLOR_ORIGINAL, _HIST_COLOR_GLOBAL, _HIST_COLOR_CLAHE]

    for col_idx, (img_arr, title, color) in enumerate(
        zip(images_list, img_titles, hist_colors)
    ):
        # Fila 0: imagenes
        ax_img: Axes = axes[0, col_idx]
        ax_img.imshow(img_arr, cmap="gray", vmin=0, vmax=255)
        ax_img.set_title(title, fontsize=_FIGURE_FONT_SIZE + 1, fontweight="bold")
        ax_img.axis("off")

        # Fila 1: histogramas
        ax_hist: Axes = axes[1, col_idx]
        _plot_histogram_on_ax(ax_hist, img_arr, f"Histograma: {title}", color)

    ensure_output_dir(output_path.parent)
    try:
        fig.savefig(str(output_path), bbox_inches="tight")
        logger.info("Figura comparativa guardada en: %s", output_path)
    except Exception as exc:
        raise IOError(
            f"No se pudo guardar la figura en '{output_path}': {exc}"
        ) from exc
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Funcion de Ejecucion por Lote
# ---------------------------------------------------------------------------


def run_histogram_analysis(
    input_dir: Path,
    output_dir: Path,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> None:
    """Ejecuta el pipeline completo de analisis de histograma sobre un directorio.

    Para cada imagen en `input_dir`:
        1. Carga la imagen en escala de grises.
        2. Aplica ecualizacion global y CLAHE.
        3. Guarda las imagenes procesadas como PNG.
        4. Genera la figura comparativa 2x3.

    Args:
        input_dir: Directorio que contiene las imagenes de entrada.
        output_dir: Directorio donde se depositaran todos los resultados.
        clip_limit: Parametro clip_limit para CLAHE.
        tile_grid_size: Parametro tile_grid_size para CLAHE.

    Raises:
        FileNotFoundError: Si `input_dir` no existe.
    """
    ensure_output_dir(output_dir)
    images: list[Path] = list_images(input_dir)

    if not images:
        logger.warning(
            "No se encontraron imagenes en el directorio: %s. "
            "Extensiones buscadas: %s",
            input_dir,
            ".tif, .tiff, .png, .jpg, .jpeg, .bmp",
        )
        return

    logger.info(
        "Iniciando analisis de histograma sobre %d imagen(es).", len(images)
    )

    for img_path in images:
        stem: str = img_path.stem
        logger.info("Procesando: %s", img_path.name)

        try:
            gray: np.ndarray = load_image_gray(img_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(
                "No se pudo cargar '%s': %s. Se omite esta imagen.", img_path.name, exc
            )
            continue

        try:
            eq_global: np.ndarray = equalize_histogram_global(gray)
            eq_clahe: np.ndarray = equalize_histogram_clahe(
                gray, clip_limit=clip_limit, tile_grid_size=tile_grid_size
            )
        except Exception as exc:
            logger.error(
                "Error al procesar '%s': %s. Se omite esta imagen.", img_path.name, exc
            )
            continue

        # Guardar imagenes procesadas
        save_image(
            eq_global, output_dir / "histogram" / f"{stem}_eq_global.png"
        )
        save_image(
            eq_clahe, output_dir / "histogram" / f"{stem}_eq_clahe.png"
        )

        # Generar figura comparativa
        fig_path: Path = output_dir / "histogram" / f"{stem}_comparison.png"
        generate_comparison_figure(
            original=gray,
            equalized_global=eq_global,
            equalized_clahe=eq_clahe,
            output_path=fig_path,
            image_name=stem,
        )

    logger.info("Analisis de histograma completado. Resultados en: %s", output_dir)
