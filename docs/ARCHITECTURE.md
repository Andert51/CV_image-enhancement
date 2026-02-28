# Arquitectura del Proyecto: Mejora de Imagenes

## Estructura de Directorios

```
T1_MejoraImagenes/
|
|-- main.py                        # Punto de entrada; delega a src/cli.py
|-- requirements.txt               # Dependencias del proyecto
|
|-- src/
|   |-- __init__.py                # Declaracion del paquete
|   |-- utils.py                   # Utilidades transversales (I/O, logging, validacion)
|   |-- histogram_ops.py           # Modulo 1: histogramas y ecualizacion
|   |-- transformations.py         # Modulo 2: transformaciones puntuales
|   |-- color_spaces.py            # Modulo 3: conversion de espacios de color
|   `-- cli.py                     # CLI Typer con subcomandos
|
|-- data/
|   |-- DIP3E_Original_Images_CH03/   # Imagenes de prueba del libro
|   `-- output/                       # Directorio de resultados (generado en ejecucion)
|       |-- histogram/
|       |   |-- <stem>_eq_global.png
|       |   |-- <stem>_eq_clahe.png
|       |   `-- <stem>_comparison.png
|       |-- transformations/
|       |   |-- <stem>_linear.png
|       |   |-- <stem>_log.png
|       |   |-- <stem>_gamma_<g>.png
|       |   `-- <stem>_transform_grid.png
|       `-- color_spaces/
|           |-- hsv/
|           |   |-- <stem>_HSV_Hue_.png
|           |   |-- <stem>_HSV_Saturation_.png
|           |   |-- <stem>_HSV_Value_.png
|           |   `-- <stem>_HSV_comparison.png
|           |-- ycbcr/ (analogamente)
|           `-- lab/   (analogamente)
|
`-- docs/
    |-- ARCHITECTURE.md            # Este archivo
    `-- analysis_report.md         # Reporte tecnico y matematico
```

## Descripcion de Modulos

### `src/utils.py`
Capa de abstraccion de entrada/salida. Provee:
- Carga de imagenes en escala de grises y BGR con validacion de tipos.
- Guardado de imagenes con creacion automatica de directorios.
- Listado de imagenes por extension.
- Funciones de validacion de arreglos NumPy (monocanal, tri-canal, dtype).
- Configuracion centralizada del modulo `logging`.

### `src/histogram_ops.py`
Modulo 1. Implementa:
- Calculo de histograma normalizado (PDF) y CDF mediante operaciones vectorizadas NumPy.
- Ecualizacion global con LUT derivada de la CDF.
- CLAHE mediante `cv2.createCLAHE`.
- Generacion de figuras 2x3 (imagenes + histogramas con CDF superpuesta).
- Pipeline por lote `run_histogram_analysis`.

### `src/transformations.py`
Modulo 2. Implementa:
- Estiramiento lineal (normalizacion Min-Max).
- Transformacion logaritmica `s = c * log(1 + r)`.
- Correccion gamma `s = c * r^gamma`.
- Panel de curvas T(r) con Matplotlib.
- Cuadricula comparativa 2x5.
- Pipeline por lote `run_transformations`.

### `src/color_spaces.py`
Modulo 3. Implementa:
- Conversion BGR a HSV, YCbCr y CIE L*a*b*.
- Extraccion de canales individuales.
- Exportacion de canales y figuras comparativas 2x4.
- Pipeline por lote `run_color_analysis`.

### `src/cli.py`
Interfaz de linea de comandos construida con Typer. Expone:
- `histogram`: ejecuta el Modulo 1.
- `transform`: ejecuta el Modulo 2.
- `colors`: ejecuta el Modulo 3.
- `all`: ejecuta los tres modulos en secuencia.
- Flag global `--verbose` para activar registro DEBUG.

### `main.py`
Punto de entrada minimalista. Verifica la version de Python (>= 3.10)
y delega a la instancia `app` de Typer.

## Dependencias Tecnologicas

| Libreria | Version minima | Proposito |
|----------|---------------|-----------|
| opencv-python | 4.8.0 | Carga, guardado, conversiones de color, CLAHE |
| numpy | 1.26.0 | Operaciones matriciales vectorizadas |
| matplotlib | 3.8.0 | Generacion de figuras |
| seaborn | 0.13.0 | Estilos estadisticos para graficos |
| scikit-image | 0.22.0 | Metricas de calidad adicionales |
| scipy | 1.11.0 | Procesamiento cientifico avanzado |
| typer[all] | 0.9.0 | CLI robusta con autocompletado |
| Pillow | 10.0.0 | Lectura de TIFF y otros formatos |

## Flujo de Ejecucion

```
Usuario -> main.py -> src/cli.py (Typer)
                         |
             +-----------+-----------+
             |           |           |
    histogram_ops  transformations  color_spaces
             |           |           |
             +-----------+-----------+
                         |
                    src/utils.py
                         |
                    data/output/
```

## Convenciones de Codigo

- Estandar de estilo: PEP 8.
- Tipo de docstrings: formato Google.
- Tipado estatico: `from __future__ import annotations` + type hints en todas las firmas.
- Manejo de errores: bloques `try/except` especificos en todas las funciones de I/O y en los pipelines por lote.
- Backend Matplotlib: `Agg` (no interactivo) para generacion de archivos sin display.
- Logging: modulo estandar `logging` con configuracion centralizada en `utils.configure_logging`.
