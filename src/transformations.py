"""
Modulo de mejora de contraste mediante transformaciones puntuales de intensidad.

Implementa las transformaciones de nivel de gris definidas sobre la funcion
s = T(r), donde r es la intensidad de entrada y s es la intensidad de salida,
ambas en el rango [0, 255] para imagenes de 8 bits.

Transformaciones implementadas:
    1. Estiramiento lineal de contraste (normalizacion Min-Max).
    2. Transformacion logaritmica: s = c * log(1 + r).
    3. Transformacion de potencia / correccion gamma: s = c * r^gamma.

Fundamentacion Matematica:
    1. Estiramiento Lineal:
        s = (r - r_min) / (r_max - r_min) * (L - 1)
        Garantiza que el rango dinamico de salida ocupe todo [0, L-1].

    2. Transformacion Logaritmica:
        s = c * log(1 + r),  c = (L - 1) / log(1 + r_max)
        Expande los niveles oscuros y comprime los claros. Util para
        visualizar imagenes con alta concentracion de pixeles en valores
        bajos (e.g., espectros de magnitud DFT).

    3. Correccion Gamma (Ley de Potencia):
        s = c * r^gamma,  r normalizado en [0, 1]
        gamma < 1: realza imagen oscura (expansion de sombras).
        gamma > 1: oscurece imagen brillante (expansion de luces).
        gamma = 1: transformacion identidad (con c = 1).
        Es la base de la correccion gamma utilizada en pantallas y
        estandares de color (sRGB, BT.709).

Referencias:
    Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing
    (4th ed.). Pearson. Seccion 3.2.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_FIGURE_DPI: int = 150
_FONT_SIZE: int = 9


# ---------------------------------------------------------------------------
# Transformaciones Puntuales
# ---------------------------------------------------------------------------


def linear_stretch(img: np.ndarray) -> np.ndarray:
    """Estira el contraste de una imagen aplicando normalizacion Min-Max.

    Mapea el rango dinamico [r_min, r_max] a [0, 255]. Si la imagen ya
    ocupa el rango completo (r_min = 0, r_max = 255), la salida es
    identica a la entrada.

    Args:
        img: Imagen monocanal uint8 de forma (H, W).

    Returns:
        Imagen resultante con rango dinamico completo [0, 255], dtype uint8.

    Raises:
        TypeError: Si `img` no es np.ndarray.
        ValueError: Si la imagen no es monocanal uint8 o es de varianza cero.
    """
    validate_grayscale(img)
    r_min: float = float(img.min())
    r_max: float = float(img.max())

    if r_max == r_min:
        logger.warning(
            "La imagen tiene varianza cero (r_min = r_max = %.1f). "
            "Se retorna la imagen original sin modificacion.",
            r_min,
        )
        return img.copy()

    # Operacion vectorizada: evita bucles Python para maximo rendimiento.
    stretched: np.ndarray = (
        (img.astype(np.float64) - r_min) / (r_max - r_min) * 255.0
    ).clip(0, 255).astype(np.uint8)

    logger.debug(
        "Estiramiento lineal aplicado: [%.1f, %.1f] -> [0, 255].", r_min, r_max
    )
    return stretched


def log_transform(img: np.ndarray, c: float | None = None) -> np.ndarray:
    """Aplica la transformacion logaritmica: s = c * log(1 + r).

    La constante `c` se escala automaticamente para que la salida ocupe
    exactamente el rango [0, 255] si no se especifica.

    Args:
        img: Imagen monocanal uint8 de forma (H, W).
        c: Constante de escalado. Si es None, se calcula como
           c = 255 / log(1 + 255), lo que garantiza s_max = 255.

    Returns:
        Imagen transformada con rango [0, 255], dtype uint8.

    Raises:
        TypeError: Si `img` no es np.ndarray.
        ValueError: Si la imagen no es monocanal uint8 o c <= 0.
    """
    validate_grayscale(img)

    if c is None:
        c = 255.0 / np.log1p(255.0)
    elif c <= 0:
        raise ValueError(
            f"La constante c debe ser positiva, se recibio: {c}"
        )

    # log1p(x) = log(1 + x): numericamente estable para x cercano a 0.
    transformed: np.ndarray = (
        c * np.log1p(img.astype(np.float64))
    ).clip(0, 255).astype(np.uint8)

    logger.debug("Transformacion logaritmica aplicada con c=%.4f.", c)
    return transformed


def gamma_correction(
    img: np.ndarray,
    gamma: float,
    c: float = 1.0,
) -> np.ndarray:
    """Aplica correccion gamma (transformacion de potencia): s = c * r^gamma.

    La imagen se normaliza a [0, 1] antes de la operacion y se reescala
    a [0, 255] al final.

    Args:
        img: Imagen monocanal uint8 de forma (H, W).
        gamma: Exponente de la transformacion de potencia.
            - gamma < 1: expande niveles oscuros, comprime brillantes.
            - gamma > 1: expande niveles brillantes, comprime oscuros.
            - gamma = 1: transformacion identidad (con c = 1).
        c: Constante de amplitud. Tipicamente 1.0.

    Returns:
        Imagen corregida con rango [0, 255], dtype uint8.

    Raises:
        TypeError: Si `img` no es np.ndarray.
        ValueError: Si la imagen no es monocanal uint8, c <= 0 o gamma <= 0.
    """
    validate_grayscale(img)
    if c <= 0:
        raise ValueError(f"La constante c debe ser positiva, se recibio: {c}")
    if gamma <= 0:
        raise ValueError(f"gamma debe ser positivo, se recibio: {gamma}")

    # Normalizacion a [0, 1], aplicacion de la ley de potencia, reescalado.
    normalized: np.ndarray = img.astype(np.float64) / 255.0
    corrected: np.ndarray = c * np.power(normalized, gamma)
    result: np.ndarray = (corrected * 255.0).clip(0, 255).astype(np.uint8)

    logger.debug("Correccion gamma aplicada: gamma=%.4f, c=%.4f.", gamma, c)
    return result


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------


def _plot_transform_curve(ax: Axes, gamma: float, c_log: float) -> None:
    """Dibuja las curvas de transformacion T(r) para las tres funciones.

    Args:
        ax: Eje Matplotlib sobre el cual renderizar.
        gamma: Valor del exponente gamma para la curva de potencia.
        c_log: Constante c para la curva logaritmica.
    """
    r: np.ndarray = np.linspace(0, 255, 512)
    r_norm: np.ndarray = r / 255.0

    # Curva identidad
    ax.plot(r, r, linestyle=":", color="#7F8C8D", linewidth=1.0, label="Identidad")

    # Curva logaritmica
    s_log = c_log * np.log1p(r)
    ax.plot(
        r,
        s_log.clip(0, 255),
        linestyle="-",
        color="#27AE60",
        linewidth=1.5,
        label=f"Log (c={c_log:.2f})",
    )

    # Curvas gamma
    for g, color in [(0.4, "#E74C3C"), (gamma, "#2980B9"), (2.5, "#8E44AD")]:
        label_str = f"Gamma={g:.1f}"
        if g == gamma:
            label_str += " (param)"
        ax.plot(
            r,
            (np.power(r_norm, g) * 255.0).clip(0, 255),
            linestyle="-",
            color=color,
            linewidth=1.5 if g == gamma else 1.0,
            alpha=1.0 if g == gamma else 0.6,
            label=label_str,
        )

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("Intensidad de entrada r", fontsize=_FONT_SIZE)
    ax.set_ylabel("Intensidad de salida s", fontsize=_FONT_SIZE)
    ax.set_title("Curvas de transformacion T(r)", fontsize=_FONT_SIZE + 1)
    ax.legend(fontsize=_FONT_SIZE - 1, loc="upper left")
    ax.tick_params(axis="both", labelsize=_FONT_SIZE - 1)
    ax.grid(True, alpha=0.3)


def generate_transformation_grid(
    img: np.ndarray,
    c_log: float,
    c_gamma: float,
    gamma: float,
    output_path: Path,
    image_name: str = "",
) -> None:
    """Genera la figura comparativa de todas las transformaciones puntuales.

    Produce una cuadricula de 2 filas x 5 columnas + panel de curvas:
        Fila 1: Imagenes (original, estiramiento, log, gamma, curvas T(r)).
        Fila 2: Histogramas de cada imagen transformada.

    Args:
        img: Imagen original en escala de grises uint8.
        c_log: Constante c para la transformacion logaritmica.
        c_gamma: Constante c para la correccion gamma.
        gamma: Exponente gamma para la correccion gamma.
        output_path: Ruta de destino del archivo PNG.
        image_name: Nombre de la imagen para el titulo.

    Raises:
        TypeError: Si los argumentos no son del tipo esperado.
        ValueError: Si los parametros son invalidos.
        IOError: Si la figura no puede guardarse.
    """
    validate_grayscale(img)
    if not isinstance(output_path, Path):
        raise TypeError(
            f"output_path debe ser pathlib.Path, se recibio: {type(output_path).__name__}"
        )

    # Calcular transformaciones
    img_linear: np.ndarray = linear_stretch(img)
    img_log: np.ndarray = log_transform(img, c=c_log)
    img_gamma: np.ndarray = gamma_correction(img, gamma=gamma, c=c_gamma)

    labels: list[str] = [
        "Original",
        "Estiramiento Lineal",
        f"Logaritmica (c={c_log:.2f})",
        f"Gamma ({gamma:.2f})",
    ]
    images_list: list[np.ndarray] = [img, img_linear, img_log, img_gamma]
    hist_colors: list[str] = ["#2C3E50", "#27AE60", "#F39C12", "#2980B9"]

    fig: Figure
    axes: np.ndarray

    # Layout: 2 filas, 5 columnas (4 imagenes/histogramas + 1 panel de curvas)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(20, 8),
        dpi=_FIGURE_DPI,
        constrained_layout=True,
    )

    fig.suptitle(
        f"Comparativa de Transformaciones Puntuales: {image_name}",
        fontsize=_FONT_SIZE + 3,
        fontweight="bold",
    )

    for col_idx, (img_arr, label, color) in enumerate(
        zip(images_list, labels, hist_colors)
    ):
        # Fila 0: imagenes
        ax_img: Axes = axes[0, col_idx]
        ax_img.imshow(img_arr, cmap="gray", vmin=0, vmax=255)
        ax_img.set_title(label, fontsize=_FONT_SIZE, fontweight="bold")
        ax_img.axis("off")

        # Fila 1: histogramas
        ax_hist: Axes = axes[1, col_idx]
        hist_raw: np.ndarray = np.bincount(img_arr.ravel(), minlength=256)
        pdf: np.ndarray = hist_raw / img_arr.size
        bins: np.ndarray = np.arange(256)
        ax_hist.bar(bins, pdf * 256, color=color, alpha=0.75, width=1.0)
        ax_hist.set_xlim(0, 255)
        ax_hist.set_xlabel("Nivel r", fontsize=_FONT_SIZE - 1)
        ax_hist.set_ylabel("Freq. relativa", fontsize=_FONT_SIZE - 1)
        ax_hist.set_title(f"Histograma: {label}", fontsize=_FONT_SIZE - 1)
        ax_hist.tick_params(axis="both", labelsize=_FONT_SIZE - 2)

    # Panel de curvas en columna 4
    _plot_transform_curve(axes[0, 4], gamma=gamma, c_log=c_log)
    axes[1, 4].axis("off")

    ensure_output_dir(output_path.parent)
    try:
        fig.savefig(str(output_path), bbox_inches="tight")
        logger.info("Figura de transformaciones guardada en: %s", output_path)
    except Exception as exc:
        raise IOError(
            f"No se pudo guardar la figura en '{output_path}': {exc}"
        ) from exc
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Funcion de Ejecucion por Lote
# ---------------------------------------------------------------------------


def run_transformations(
    input_dir: Path,
    output_dir: Path,
    c_log: float = 1.0,
    c_gamma: float = 1.0,
    gamma: float = 1.5,
) -> None:
    """Ejecuta el pipeline de transformaciones puntuales sobre un directorio.

    Para cada imagen en `input_dir`:
        1. Carga la imagen en escala de grises.
        2. Aplica las tres transformaciones (lineal, log, gamma).
        3. Guarda cada imagen resultante como PNG.
        4. Genera la figura comparativa 2x5.

    Args:
        input_dir: Directorio de imagenes de entrada.
        output_dir: Directorio raiz de salida.
        c_log: Constante de la transformacion logaritmica.
            Si es <= 0 o no especificado correctamente, se usa el
            escalado automatico.
        c_gamma: Constante de la correccion gamma.
        gamma: Exponente gamma.

    Raises:
        FileNotFoundError: Si `input_dir` no existe.
    """
    ensure_output_dir(output_dir)
    images: list[Path] = list_images(input_dir)

    if not images:
        logger.warning(
            "No se encontraron imagenes en: %s", input_dir
        )
        return

    logger.info(
        "Iniciando transformaciones puntuales sobre %d imagen(es). "
        "Parametros: c_log=%.4f, c_gamma=%.4f, gamma=%.4f.",
        len(images),
        c_log,
        c_gamma,
        gamma,
    )

    c_log_eff: float = c_log if c_log > 0 else (255.0 / np.log1p(255.0))

    for img_path in images:
        stem: str = img_path.stem
        logger.info("Procesando: %s", img_path.name)

        try:
            gray: np.ndarray = load_image_gray(img_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(
                "No se pudo cargar '%s': %s. Se omite.", img_path.name, exc
            )
            continue

        try:
            img_linear: np.ndarray = linear_stretch(gray)
            img_log: np.ndarray = log_transform(gray, c=c_log_eff)
            img_gamma_out: np.ndarray = gamma_correction(gray, gamma=gamma, c=c_gamma)
        except Exception as exc:
            logger.error(
                "Error al transformar '%s': %s. Se omite.", img_path.name, exc
            )
            continue

        trans_dir: Path = output_dir / "transformations"
        save_image(img_linear, trans_dir / f"{stem}_linear.png")
        save_image(img_log, trans_dir / f"{stem}_log.png")
        save_image(img_gamma_out, trans_dir / f"{stem}_gamma_{gamma:.2f}.png")

        fig_path: Path = trans_dir / f"{stem}_transform_grid.png"
        generate_transformation_grid(
            img=gray,
            c_log=c_log_eff,
            c_gamma=c_gamma,
            gamma=gamma,
            output_path=fig_path,
            image_name=stem,
        )

    logger.info(
        "Pipeline de transformaciones completado. Resultados en: %s",
        output_dir,
    )
