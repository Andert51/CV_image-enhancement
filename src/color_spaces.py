"""
Modulo de conversion y analisis entre modelos de color.

Implementa la conversion de imagenes BGR a los espacios de color:
    - HSV  (Hue, Saturation, Value)
    - YCbCr (Luminancia, Crominancia azul, Crominancia roja)
    - CIE L*a*b* (Luminancia perceptual, crominancia verde-roja, azul-amarilla)

Para cada espacio, se extraen, exportan y visualizan los canales individuales.

Fundamentacion Matematica:

    1. BGR -> HSV:
       Modelo de cono invertido. Hue es el angulo en el circulo cromatico [0, 360].
       Saturation mide la pureza del color [0, 1]. Value es el brillo [0, 1].
       No separa la luminancia de la crominancia de forma lineal, por lo que
       no es adecuado para tareas de procesamiento de luminancia independiente.

    2. BGR -> YCbCr (BT.601):
       Y  = 0.299 R + 0.587 G + 0.114 B         (luminancia)
       Cb = 128 - 0.168736 R - 0.331264 G + 0.5 B (crominancia azul)
       Cr = 128 + 0.5 R - 0.418688 G - 0.081312 B (crominancia roja)
       La separacion Y/CbCr permite comprimir la crominancia (vision humana
       es menos sensible al color que al brillo). Base del estandar JPEG y H.264.

    3. BGR -> CIE L*a*b*:
       L* in [0, 100]: luminancia perceptualmente uniforme.
       a* in [-128, 127]: eje cromatico verde(-) a rojo(+).
       b* in [-128, 127]: eje cromatico azul(-) a amarillo(+).
       Espacio perceptualmente uniforme: distancias euclidianas en L*a*b*
       corresponden a diferencias de color percibidas por el sistema visual humano
       (metrica Delta-E). Fundamental en segmentacion robusta al color.

Referencias:
    Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.).
    Pearson. Cap. 6.
    
    International Telecommunication Union. (2011). BT.601 Studio encoding parameters
    of digital television for standard 4:3 and wide screen 16:9 aspect ratios.

    CIE Publication 15:2004. Colorimetry, 3rd Edition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.utils import (
    ensure_output_dir,
    list_images,
    load_image_bgr,
    save_image,
    validate_bgr,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_FIGURE_DPI: int = 150
_FONT_SIZE: int = 9


# ---------------------------------------------------------------------------
# Definicion de espacios y sus metadatos
# ---------------------------------------------------------------------------


class ColorSpaceDescriptor(NamedTuple):
    """Descriptor de un espacio de color para uso interno.

    Attributes:
        name: Nombre corto del espacio (e.g., "HSV").
        cv_code: Codigo de conversion de OpenCV (e.g., cv2.COLOR_BGR2HSV).
        channel_names: Nombres de los canales en orden (C0, C1, C2).
        channel_cmaps: Colormaps de Matplotlib para visualizacion de cada canal.
        channel_imshow_vmax: Valor maximo para imshow por canal (vmin siempre 0).
            HSV Hue usa 179 (rango real [0,179] en uint8).
        channel_hist_bias: Sesgo a restar del valor del pixel en el histograma.
            0 = sin sesgo (e.g. Luminancia, HSV S/V).
            128 = canal con sesgo (Cr, Cb, a*, b*) — muestra el eje como [-128, 127].
    """

    name: str
    cv_code: int
    channel_names: tuple[str, str, str]
    channel_cmaps: tuple[str, str, str]
    channel_imshow_vmax: tuple[int, int, int]
    channel_hist_bias: tuple[int, int, int]


COLOR_SPACES: list[ColorSpaceDescriptor] = [
    ColorSpaceDescriptor(
        name="HSV",
        cv_code=cv2.COLOR_BGR2HSV,
        # OpenCV Hue: [0, 179] (angulo cromatico en grados / 2)
        # Saturacion y Valor: [0, 255]
        channel_names=("Hue (Matiz)", "Saturation (Saturacion)", "Value (Valor/Brillo)"),
        channel_cmaps=("hsv", "gray", "gray"),
        channel_imshow_vmax=(179, 255, 255),
        channel_hist_bias=(0, 0, 0),
    ),
    ColorSpaceDescriptor(
        name="YCbCr",
        cv_code=cv2.COLOR_BGR2YCrCb,
        # OpenCV COLOR_BGR2YCrCb devuelve [Y, Cr, Cb]
        # Cr y Cb: rango uint8 [0,255] con 128 = neutro (sin crominancia)
        channel_names=("Y (Luminancia)", "Cr (Crom. Roja)", "Cb (Crom. Azul)"),
        channel_cmaps=("gray", "RdBu_r", "RdYlBu_r"),
        channel_imshow_vmax=(255, 255, 255),
        channel_hist_bias=(0, 128, 128),
    ),
    ColorSpaceDescriptor(
        name="LAB",
        cv_code=cv2.COLOR_BGR2Lab,
        # OpenCV COLOR_BGR2Lab uint8: L* en [0,255]; a* y b* en [0,255] con 128=neutro
        # a*: verde(-128) <-> rojo(+127)   b*: azul(-128) <-> amarillo(+127)
        channel_names=("L* (Luminancia CIE)", "a* (Verde\u2194Rojo)", "b* (Azul\u2194Amarillo)"),
        channel_cmaps=("gray", "RdYlGn_r", "RdYlBu_r"),
        channel_imshow_vmax=(255, 255, 255),
        channel_hist_bias=(0, 128, 128),
    ),
]


# ---------------------------------------------------------------------------
# Funciones de Conversion
# ---------------------------------------------------------------------------


def bgr_to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR al espacio de color HSV.

    OpenCV representa el canal Hue en el rango [0, 179] (no [0, 359])
    para almacenamiento en uint8. Saturation y Value estan en [0, 255].

    Args:
        img_bgr: Imagen BGR tri-canal uint8 de forma (H, W, 3).

    Returns:
        Imagen HSV de forma (H, W, 3), dtype uint8.

    Raises:
        TypeError: Si `img_bgr` no es np.ndarray.
        ValueError: Si la imagen no es tri-canal uint8.
    """
    validate_bgr(img_bgr)
    result: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    logger.debug("Conversion BGR->HSV completada. Forma: %s", result.shape)
    return result


def bgr_to_ycbcr(img_bgr: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR al espacio de color YCbCr (BT.601).

    OpenCV implementa COLOR_BGR2YCrCb con el orden de canales [Y, Cr, Cb].
    Este modulo normaliza el orden semantico a [Y, Cb, Cr] para consistencia
    con la nomenclatura ITU-R BT.601.

    Args:
        img_bgr: Imagen BGR tri-canal uint8 de forma (H, W, 3).

    Returns:
        Imagen con canales [Y, Cb, Cr] de forma (H, W, 3), dtype uint8.

    Raises:
        TypeError: Si `img_bgr` no es np.ndarray.
        ValueError: Si la imagen no es tri-canal uint8.
    """
    validate_bgr(img_bgr)
    # OpenCV devuelve [Y, Cr, Cb]; se reordena a [Y, Cb, Cr].
    ycrcb: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycbcr: np.ndarray = ycrcb[:, :, [0, 2, 1]]
    logger.debug("Conversion BGR->YCbCr completada. Forma: %s", ycbcr.shape)
    return ycbcr


def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR al espacio de color CIE L*a*b*.

    OpenCV almacena los valores L*a*b* escalados en uint8:
        L*: [0, 100] escalado a [0, 255]
        a*: [-128, 127] escalado a [0, 255]
        b*: [-128, 127] escalado a [0, 255]

    Args:
        img_bgr: Imagen BGR tri-canal uint8 de forma (H, W, 3).

    Returns:
        Imagen L*a*b* de forma (H, W, 3), dtype uint8.

    Raises:
        TypeError: Si `img_bgr` no es np.ndarray.
        ValueError: Si la imagen no es tri-canal uint8.
    """
    validate_bgr(img_bgr)
    result: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    logger.debug("Conversion BGR->L*a*b* completada. Forma: %s", result.shape)
    return result


# ---------------------------------------------------------------------------
# Extraccion y Visualizacion de Canales
# ---------------------------------------------------------------------------


def extract_channels(
    img_converted: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extrae los tres canales de una imagen convertida como arreglos individuales.

    Args:
        img_converted: Imagen de forma (H, W, 3), dtype uint8.

    Returns:
        Diccionario {'C0': ..., 'C1': ..., 'C2': ...} donde cada valor
        es un arreglo monocanal uint8 de forma (H, W).

    Raises:
        TypeError: Si `img_converted` no es np.ndarray.
        ValueError: Si la imagen no tiene exactamente 3 canales.
    """
    if not isinstance(img_converted, np.ndarray):
        raise TypeError(
            f"Se esperaba np.ndarray, se recibio: {type(img_converted).__name__}"
        )
    if img_converted.ndim != 3 or img_converted.shape[2] != 3:
        raise ValueError(
            f"Se esperaba imagen (H, W, 3), se recibio forma: {img_converted.shape}"
        )
    c0, c1, c2 = cv2.split(img_converted)
    return {"C0": c0, "C1": c1, "C2": c2}


def generate_color_space_figure(
    img_bgr: np.ndarray,
    descriptor: ColorSpaceDescriptor,
    output_path: Path,
    image_name: str = "",
) -> None:
    """Genera la figura comparativa de canales para un espacio de color dado.

    Produce una cuadricula de 2 filas:
        Fila 1: Imagen original BGR  |  Canal 0  |  Canal 1  |  Canal 2
        Fila 2: Histograma de cada canal

    Args:
        img_bgr: Imagen original en formato BGR uint8.
        descriptor: Instancia de ColorSpaceDescriptor con metadatos del espacio.
        output_path: Ruta de destino del archivo PNG.
        image_name: Nombre de la imagen para el titulo.

    Raises:
        TypeError: Si los argumentos no son del tipo esperado.
        IOError: Si la figura no puede guardarse.
    """
    validate_bgr(img_bgr)
    if not isinstance(output_path, Path):
        raise TypeError(
            f"output_path debe ser pathlib.Path, se recibio: {type(output_path).__name__}"
        )

    # Conversion al espacio de color
    img_converted: np.ndarray = cv2.cvtColor(img_bgr, descriptor.cv_code)
    channels: dict[str, np.ndarray] = extract_channels(img_converted)

    img_rgb: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig: Figure
    axes: np.ndarray

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(16, 8),
        dpi=_FIGURE_DPI,
        constrained_layout=True,
    )

    fig.suptitle(
        f"Espacio de Color {descriptor.name}: {image_name}",
        fontsize=_FONT_SIZE + 3,
        fontweight="bold",
    )

    # Fila 0, columna 0: imagen original RGB
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original (RGB)", fontsize=_FONT_SIZE, fontweight="bold")
    axes[0, 0].axis("off")

    # Histograma de luminosidad (valor promedio BGR)
    axes[1, 0].hist(
        img_bgr.mean(axis=2).ravel(),
        bins=256,
        range=(0, 255),
        color="#2C3E50",
        alpha=0.8,
    )
    axes[1, 0].set_title("Histograma Brillo BGR", fontsize=_FONT_SIZE - 1)
    axes[1, 0].set_xlabel("Nivel de intensidad", fontsize=_FONT_SIZE - 1)
    axes[1, 0].tick_params(labelsize=_FONT_SIZE - 2)

    channel_keys = ["C0", "C1", "C2"]
    hist_colors: list[str] = ["#E74C3C", "#27AE60", "#2980B9"]

    for col_idx, (key, ch_name, cmap, hcolor, ch_vmax, ch_bias) in enumerate(
        zip(
            channel_keys,
            descriptor.channel_names,
            descriptor.channel_cmaps,
            hist_colors,
            descriptor.channel_imshow_vmax,
            descriptor.channel_hist_bias,
        ),
        start=1,
    ):
        ch_arr: np.ndarray = channels[key]

        # --- Imagen del canal -------------------------------------------
        # Para canales con sesgo (Cr, Cb, a*, b*): usar colormap divergente
        # centrado en 128 mediante TwoSlopeNorm.
        # Para HSV Hue: vmax=179 para cubrir el ciclo de color completo.
        if ch_bias == 128:
            import matplotlib.colors as mcolors
            norm = mcolors.TwoSlopeNorm(vcenter=128.0, vmin=0.0, vmax=255.0)
            axes[0, col_idx].imshow(ch_arr, cmap=cmap, norm=norm)
        else:
            axes[0, col_idx].imshow(ch_arr, cmap=cmap, vmin=0, vmax=ch_vmax)

        axes[0, col_idx].set_title(
            f"Canal: {ch_name}", fontsize=_FONT_SIZE, fontweight="bold"
        )
        axes[0, col_idx].axis("off")

        # --- Histograma del canal ----------------------------------------
        if ch_bias != 0:
            # Canal con sesgo: mostrar en escala centrada en 0
            # Rango real del canal: [0-bias, vmax-bias] = [-128, 127] para bias=128
            signed_min: int = -ch_bias
            signed_max: int = ch_vmax - ch_bias
            vals: np.ndarray = ch_arr.ravel().astype(np.int16) - ch_bias
            n_bins: int = ch_vmax + 1  # 256 bins
            axes[1, col_idx].hist(
                vals,
                bins=n_bins,
                range=(signed_min, signed_max + 1),
                color=hcolor,
                alpha=0.75,
            )
            axes[1, col_idx].axvline(
                x=0, color="#555555", linewidth=0.8, linestyle="--", label="neutro"
            )
            axes[1, col_idx].set_xlim(signed_min, signed_max)
            axes[1, col_idx].set_xlabel("Valor (0 = neutro)", fontsize=_FONT_SIZE - 1)
        else:
            # Canal sin sesgo: histograma normal [0, ch_vmax]
            n_bins_unbiased: int = ch_vmax + 1
            pdf: np.ndarray = (
                np.bincount(ch_arr.ravel(), minlength=n_bins_unbiased)
                .astype(np.float64)
                / ch_arr.size
            )
            bins: np.ndarray = np.arange(n_bins_unbiased)
            axes[1, col_idx].bar(
                bins, pdf * n_bins_unbiased, color=hcolor, alpha=0.75, width=1.0
            )
            axes[1, col_idx].set_xlim(0, ch_vmax)
            if ch_vmax == 179:  # HSV Hue
                axes[1, col_idx].set_xlabel(
                    "Matiz (deg/2, rango [0, 179])", fontsize=_FONT_SIZE - 1
                )
            else:
                axes[1, col_idx].set_xlabel(
                    "Nivel de intensidad", fontsize=_FONT_SIZE - 1
                )

        axes[1, col_idx].set_title(f"Histograma: {ch_name}", fontsize=_FONT_SIZE - 1)
        axes[1, col_idx].set_ylabel("Frecuencia", fontsize=_FONT_SIZE - 1)
        axes[1, col_idx].tick_params(labelsize=_FONT_SIZE - 2)

    ensure_output_dir(output_path.parent)
    try:
        fig.savefig(str(output_path), bbox_inches="tight")
        logger.info(
            "Figura del espacio %s guardada en: %s", descriptor.name, output_path
        )
    except Exception as exc:
        raise IOError(
            f"No se pudo guardar la figura en '{output_path}': {exc}"
        ) from exc
    finally:
        plt.close(fig)


def export_channel_visualizations(
    img_bgr: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    """Exporta los canales individuales de todos los espacios de color como PNG.

    Para cada espacio en COLOR_SPACES:
        - Convierte la imagen.
        - Guarda cada canal como archivo PNG independiente.
        - Genera la figura comparativa de 4 columnas.

    Args:
        img_bgr: Imagen original en formato BGR uint8.
        output_dir: Directorio raiz para los archivos de salida.
        stem: Nombre base del archivo (sin extension) para los nombres de salida.

    Raises:
        TypeError: Si `img_bgr` no es np.ndarray o `output_dir` no es Path.
        ValueError: Si la imagen no es tri-canal uint8.
    """
    validate_bgr(img_bgr)
    if not isinstance(output_dir, Path):
        raise TypeError(
            f"output_dir debe ser pathlib.Path, se recibio: {type(output_dir).__name__}"
        )

    for descriptor in COLOR_SPACES:
        space_dir: Path = output_dir / "color_spaces" / descriptor.name.lower()
        ensure_output_dir(space_dir)

        try:
            img_converted: np.ndarray = cv2.cvtColor(img_bgr, descriptor.cv_code)
        except cv2.error as exc:
            logger.error(
                "OpenCV no pudo convertir al espacio %s: %s. Se omite.",
                descriptor.name,
                exc,
            )
            continue

        channels: dict[str, np.ndarray] = extract_channels(img_converted)

        for key, ch_name in zip(
            ["C0", "C1", "C2"], descriptor.channel_names
        ):
            # Nombre del archivo: limpia caracteres especiales
            safe_ch_name: str = (
                ch_name.split("(")[0].strip().replace(" ", "_").replace("*", "star")
            )
            ch_path: Path = space_dir / f"{stem}_{descriptor.name}_{safe_ch_name}.png"
            try:
                save_image(channels[key], ch_path)
            except IOError as exc:
                logger.error(
                    "No se pudo guardar el canal '%s': %s", ch_path, exc
                )

        # Figura comparativa del espacio de color
        fig_path: Path = space_dir / f"{stem}_{descriptor.name}_comparison.png"
        try:
            generate_color_space_figure(
                img_bgr=img_bgr,
                descriptor=descriptor,
                output_path=fig_path,
                image_name=stem,
            )
        except (IOError, Exception) as exc:
            logger.error(
                "Error generando figura para espacio %s, imagen '%s': %s",
                descriptor.name,
                stem,
                exc,
            )


# ---------------------------------------------------------------------------
# Funcion de Ejecucion por Lote
# ---------------------------------------------------------------------------


def run_color_analysis(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Ejecuta el pipeline completo de analisis de espacios de color.

    Para cada imagen en `input_dir`:
        1. Carga la imagen en formato BGR.
        2. Convierte a HSV, YCbCr y L*a*b*.
        3. Exporta canales individuales y figuras comparativas.

    Args:
        input_dir: Directorio de imagenes de entrada.
        output_dir: Directorio raiz de salida.

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
        "Iniciando analisis de espacios de color sobre %d imagen(es).", len(images)
    )

    for img_path in images:
        stem: str = img_path.stem
        logger.info("Procesando: %s", img_path.name)

        try:
            bgr: np.ndarray = load_image_bgr(img_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(
                "No se pudo cargar '%s': %s. Se omite.", img_path.name, exc
            )
            continue

        try:
            export_channel_visualizations(bgr, output_dir, stem)
        except Exception as exc:
            logger.error(
                "Error procesando '%s' en espacios de color: %s. Se omite.",
                img_path.name,
                exc,
            )
            continue

    logger.info(
        "Analisis de espacios de color completado. Resultados en: %s", output_dir
    )
