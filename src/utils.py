"""
Modulo de utilidades generales para carga, guardado y manejo de imagenes.

Provee funciones de entrada/salida robustas con manejo de excepciones,
validacion de tipos MIME y soporte para multiples formatos (TIFF, PNG, JPG, BMP).

Notas:
    Este modulo es una dependencia transversal utilizada por todos
    los demas modulos del paquete src/.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_c_stderr():
    """Suprime las advertencias de LibTIFF escritas directamente al stderr de C.

    LibTIFF (usado internamente por OpenCV al leer archivos .tif con metadatos
    no estandar) emite mensajes WARN directamente al descriptor de archivo 2
    (stderr a nivel del sistema operativo), saltandose el sistema de logging de
    Python. Este context manager redirige temporalmente ese descriptor a /dev/null.
    """
    try:
        devnull_fd: int = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd: int = os.dup(2)
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(devnull_fd)
            os.close(saved_stderr_fd)
    except OSError:
        # Si la redireccion falla (e.g. descriptor no disponible), se ignora.
        yield


SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
)


def load_image_gray(path: Path) -> np.ndarray:
    """Carga una imagen desde disco en escala de grises.

    Utiliza cv2.IMREAD_GRAYSCALE para asegurar una representacion
    monocanal de 8 bits (uint8).

    Args:
        path: Ruta absoluta o relativa al archivo de imagen.

    Returns:
        Arreglo NumPy de forma (H, W) con dtype uint8.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
        ValueError: Si el archivo no puede ser decodificado como imagen valida.
        TypeError: Si el argumento `path` no es una instancia de pathlib.Path.
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"Se esperaba un objeto pathlib.Path, se recibio: {type(path).__name__}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"El archivo de imagen no existe en la ruta especificada: {path}"
        )
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "La extension '%s' no esta en la lista de extensiones verificadas. "
            "Se intentara la carga de todas formas.",
            path.suffix,
        )

    with _suppress_c_stderr():
        img: Optional[np.ndarray] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"OpenCV no pudo decodificar el archivo como imagen: {path}. "
            "Verifique que el archivo no este corrupto."
        )
    logger.debug("Imagen cargada en escala de grises: %s | forma=%s", path.name, img.shape)
    return img


def load_image_bgr(path: Path) -> np.ndarray:
    """Carga una imagen desde disco en el espacio de color BGR.

    Args:
        path: Ruta absoluta o relativa al archivo de imagen.

    Returns:
        Arreglo NumPy de forma (H, W, 3) con dtype uint8 en orden BGR.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no puede ser decodificado como imagen.
        TypeError: Si `path` no es pathlib.Path.
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"Se esperaba un objeto pathlib.Path, se recibio: {type(path).__name__}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"El archivo de imagen no existe en la ruta especificada: {path}"
        )

    with _suppress_c_stderr():
        img: Optional[np.ndarray] = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"OpenCV no pudo decodificar el archivo: {path}."
        )

    # Si la imagen tiene un solo canal, convertir a BGR replicando el canal.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        logger.debug(
            "Imagen monocanal convertida a BGR por replicacion de canal: %s", path.name
        )

    logger.debug("Imagen cargada en BGR: %s | forma=%s", path.name, img.shape)
    return img


def save_image(img: np.ndarray, path: Path) -> None:
    """Persiste un arreglo NumPy como imagen en disco.

    El formato de salida es inferido a partir de la extension del archivo
    destino. Crea los directorios intermedios si no existen.

    Args:
        img: Arreglo NumPy representando la imagen.
        path: Ruta de destino con extension valida.

    Raises:
        TypeError: Si los argumentos no son de los tipos esperados.
        IOError: Si OpenCV no puede escribir la imagen en disco.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"Se esperaba np.ndarray como imagen, se recibio: {type(img).__name__}"
        )
    if not isinstance(path, Path):
        raise TypeError(
            f"Se esperaba pathlib.Path como destino, se recibio: {type(path).__name__}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    success: bool = cv2.imwrite(str(path), img)
    if not success:
        raise IOError(
            f"OpenCV no pudo escribir la imagen en: {path}. "
            "Verifique los permisos del directorio y la extension del archivo."
        )
    logger.debug("Imagen guardada exitosamente en: %s", path)


def ensure_output_dir(path: Path) -> None:
    """Crea el directorio de salida y todos sus intermedios si no existen.

    Args:
        path: Ruta del directorio a crear.

    Raises:
        TypeError: Si `path` no es pathlib.Path.
        OSError: Si el directorio no puede ser creado por permisos del S.O.
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"Se esperaba un objeto pathlib.Path, se recibio: {type(path).__name__}"
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Directorio asegurado: %s", path)
    except OSError as exc:
        logger.error("No se pudo crear el directorio '%s': %s", path, exc)
        raise


def list_images(
    folder: Path,
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
) -> list[Path]:
    """Lista todas las imagenes en un directorio dado por extension.

    La busqueda es no-recursiva (nivel de directorio unico).

    Args:
        folder: Directorio a escanear.
        extensions: Tupla de extensiones de archivo validas (con punto incluido).

    Returns:
        Lista ordenada de objetos Path que apuntan a archivos de imagen.

    Raises:
        FileNotFoundError: Si el directorio no existe.
        TypeError: Si `folder` no es pathlib.Path.
    """
    if not isinstance(folder, Path):
        raise TypeError(
            f"Se esperaba pathlib.Path, se recibio: {type(folder).__name__}"
        )
    if not folder.is_dir():
        raise FileNotFoundError(
            f"El directorio especificado no existe o no es un directorio: {folder}"
        )

    images: list[Path] = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    logger.info(
        "Encontradas %d imagenes en '%s'.",
        len(images),
        folder,
    )
    return images


def validate_grayscale(img: np.ndarray) -> None:
    """Valida que un arreglo NumPy sea una imagen en escala de grises valida.

    Args:
        img: Arreglo a validar.

    Raises:
        TypeError: Si `img` no es np.ndarray.
        ValueError: Si la imagen no es monocanal o no tiene dtype uint8.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"Se esperaba np.ndarray, se recibio: {type(img).__name__}"
        )
    if img.ndim != 2:
        raise ValueError(
            f"Se esperaba imagen monocanal bidimensional (H, W), "
            f"se recibio forma: {img.shape}"
        )
    if img.dtype != np.uint8:
        raise ValueError(
            f"Se esperaba dtype uint8, se recibio: {img.dtype}"
        )


def validate_bgr(img: np.ndarray) -> None:
    """Valida que un arreglo NumPy sea una imagen BGR de 3 canales uint8.

    Args:
        img: Arreglo a validar.

    Raises:
        TypeError: Si `img` no es np.ndarray.
        ValueError: Si la imagen no tiene 3 canales o no es uint8.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"Se esperaba np.ndarray, se recibio: {type(img).__name__}"
        )
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Se esperaba imagen tri-canal BGR (H, W, 3), "
            f"se recibio forma: {img.shape}"
        )
    if img.dtype != np.uint8:
        raise ValueError(
            f"Se esperaba dtype uint8, se recibio: {img.dtype}"
        )


def configure_logging(level: int = logging.INFO) -> None:
    """Configura el modulo de logging del proyecto con formato estandarizado.

    Args:
        level: Nivel de registro (e.g., logging.DEBUG, logging.INFO).
    """
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
