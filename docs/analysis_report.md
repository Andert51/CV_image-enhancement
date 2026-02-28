# Reporte Tecnico: Mejora de Imagenes Digitales mediante Procesamiento Espacial y Analisis de Espacios de Color

**Asignatura:** Vision por Computadora  
**Practica:** T1 — Mejora de Imagenes  
**Autor:** Andre  
**Dataset:** Gonzalez & Woods, *Digital Image Processing* 3.ª/4.ª ed., Cap. 3 (28 imagenes)  
**Entorno:** Python 3.10 · OpenCV 4.13 · NumPy 2.4 · Matplotlib 3.10  
**Fecha:** Febrero 2026

---

## Resumen (Abstract)

Este reporte documenta el diseño, implementacion y evaluacion de un pipeline modular de mejora de imagenes digitales desarrollado en Python. El sistema implementa tres modulos independientes: (1) analisis de histograma e igualacion de contraste mediante ecualizacion global y CLAHE; (2) transformaciones puntuales de intensidad — estiramiento lineal, transformacion logaritmica y correccion gamma —; y (3) conversion y analisis multicanal en espacios de color HSV, YCbCr y CIE L\*a\*b\*. Se procesaron las 28 imagenes de referencia del Capitulo 3 de Gonzalez & Woods (2018), que cubren escenarios de subexposicion, sobreexposicion, rango dinamico alto, ruido salt-and-pepper, texturas finas y espectros de Fourier. Los resultados evidencian que CLAHE (clip\_limit = 2.0, tile = 8×8) supera sistematicamente a la ecualizacion global en imagenes medicas por su capacidad de adaptacion local. La transformacion logaritmica es imprescindible para la visualizacion del espectro DFT. La correccion gamma $\gamma = 1.5$ mejora el contraste en imagenes sobreexpuestas sin introducir saturacion. En el analisis de espacios de color, la descomposicion en YCbCr y L\*a\*b\* revela estructuras cromaticas invisibles en el espacio BGR, confirmando la utilidad de la representacion multisespacio para tareas de segmentacion y reconocimiento. Los artefactos de las advertencias LibTIFF presentes en la lectura de archivos .tif con metadatos propietarios fueron neutralizados mediante redireccion del descriptor de archivo a nivel del sistema operativo.

---

## Tabla de Contenidos

1. [Introduccion](#introduccion)
2. [Materiales y Metodos](#materiales-y-metodos)
   - 2.1 [Dataset y entorno experimental](#dataset)
   - 2.2 [Arquitectura del sistema](#arquitectura)
   - 2.3 [Modulo 1: Histogramas y Ecualizacion](#modulo-1-metodos)
   - 2.4 [Modulo 2: Transformaciones Puntuales](#modulo-2-metodos)
   - 2.5 [Modulo 3: Espacios de Color](#modulo-3-metodos)
3. [Resultados](#resultados)
   - 3.1 [Resultados de Ecualizacion de Histograma](#resultados-histograma)
   - 3.2 [Resultados de Transformaciones Puntuales](#resultados-transformaciones)
   - 3.3 [Resultados de Analisis de Espacios de Color](#resultados-colores)
4. [Discusion](#discusion)
5. [Conclusiones](#conclusiones)
6. [Referencias Bibliograficas](#referencias)

---

## 1. Introduccion {#introduccion}

### 1.1 Contexto y motivacion

La mejora de imagenes digitales (*image enhancement*) es un conjunto de operaciones cuyo objetivo es transformar una imagen de entrada en una representacion visual mas adecuada para un proposito especifico — ya sea la interpretacion humana, la extraccion automatizada de caracteristicas, o la preparacion de datos para sistemas de aprendizaje profundo. A diferencia de la restauracion de imagenes (*image restoration*), que intenta revertir la degradacion conocida del sistema de adquisicion, la mejora es inherentemente subjetiva: el criterio de exito depende del observador o de la tarea aguas abajo (Gonzalez & Woods, 2018).

La problematica central que motiva esta practica es la siguiente: las imagenes capturadas en condiciones reales exhiben distribuciones de intensidad suboptimas — subexposicion, sobreexposicion, bajo contraste local, rango dinamico excesivo — que impiden la deteccion de detalles relevantes, tanto por el sistema visual humano como por algoritmos de vision por computadora. Adicionalmente, la representacion RGB/BGR en que operan la mayoria de las camaras no facilita la separacion de luminancia y crominancia, lo que dificulta tareas de segmentacion y reconocimiento que dependen fundamentalmente del color.

### 1.2 Objetivos

Los objetivos especificos de esta practica son:

1. **Implementar y comparar** dos tecnicas de igualacion del histograma de intensidades — ecualizacion global y CLAHE — sobre un conjunto diverso de imagenes en escala de grises, y analizar el impacto en la distribucion de intensidades resultante.

2. **Aplicar y evaluar** tres familias de transformaciones puntuales de intensidad — estiramiento lineal, transformacion logaritmica y correccion gamma — sobre el mismo conjunto de imagenes, identificando el escenario optimo de aplicacion de cada una.

3. **Convertir y analizar** imagenes BGR en los espacios de color HSV, YCbCr y CIE L\*a\*b\*, extrayendo y visualizando los canales individuales con mapas de color correctamente calibrados por espacio, e interpretar la informacion cromatica y de luminancia de cada canal.

4. **Documentar y sistematizar** los resultados mediante figuras comparativas automatizadas y un reporte tecnico con fundamentacion matematica rigurosa.

### 1.3 Alcance y limitaciones del estudio

El alcance del estudio se circunscribe a las 28 imagenes del Capitulo 3 de Gonzalez & Woods (2018), que cubren intencionalmente escenarios extremos de iluminacion y tipos de imagen variados (medicas, astronomicas, industriales, textual). Las tecnicas implementadas pertenecen al dominio espacial (no frecuencial): no se aplican filtros en el dominio de la transformada de Fourier ni tecnicas de filtrado adaptativo basadas en vecindad.

Las principales limitaciones son:

- **Evaluacion cualitativa:** La ausencia de imagenes de referencia (*ground truth*) impide calcular metricas de calidad objetivas como PSNR o SSIM. La evaluacion es visual y estadistica (histogramas, entropia).
- **Imagenes en escala de grises para Modulos 1 y 2:** OpenCV convierte automaticamente a luminancia cualquier imagen a color cargada en modo `IMREAD_GRAYSCALE`, perdiendo informacion cromatica que podria ser relevante.
- **Parametros fijos:** Los experimentos se ejecutan con un unico conjunto de parametros por defecto (clip\_limit = 2.0, tile = 8×8, $\gamma = 1.5$). Un estudio exhaustivo requeriria un barrido sistematico de hiperparametros con criterio de seleccion objetivo.

### 1.4 Organizacion del documento

Este reporte sigue la estructura IMRAD. La seccion de **Materiales y Metodos** describe el dataset, el entorno de software y las bases matematicas de cada tecnica implementada. La seccion de **Resultados** presenta las observaciones experimentales organizadas por modulo y tipo de imagen. La seccion de **Discusion** interpreta los resultados de manera comparativa, identifica limitaciones y propone trabajo futuro. Las **Conclusiones** sintetizan los hallazgos principales.

---

## 2. Materiales y Metodos {#materiales-y-metodos}

### 2.1 Dataset y entorno experimental {#dataset}

#### 2.1.1 Dataset de imagenes

Las 28 imagenes de prueba (formato TIFF, 8 bits por canal) provienen del repositorio oficial del libro *Digital Image Processing* de Gonzalez & Woods, disponible en [imageprocessingplace.com](http://www.imageprocessingplace.com). Estas imagenes fueron seleccionadas por los autores para ilustrar los fenomenos tipicos del procesamiento de intensidades en el Capitulo 3. La Tabla 1 clasifica las imagenes por tipo y el fenomeno de interes principal:

**Tabla 1. Clasificacion del dataset.**

| Grupo | Figuras representativas | Tipo | Fenomeno de interes |
|-------|------------------------|------|---------------------|
| Radiologicas | Fig0304, Fig0308, Fig0342, Fig0343 | Escala de grises | Contraste bajo, estructuras internas finas |
| Astrofisicas | Fig0323, Fig0334, Fig0338 | Escala de grises | Rango dinamico alto, ruido de fondo |
| Aereas / satelitales | Fig0309, Fig0316, Fig0320 | Color / escala de grises | Subexposicion, sobreexposicion, niebla |
| Botanicas / microscopicas | Fig0310, Fig0312 | Color | Saturacion de color, gradientes |
| Frecuenciales (DFT) | Fig0305 | Escala de grises | Rango dinamico extremo (> 10^6 niveles en float) |
| Industriales | Fig0314, Fig0335, Fig0340 | Color / escala de grises | Ruido salt-and-pepper, texto, texturas finas |
| Patrones de prueba | Fig0307, Fig0326, Fig0327, Fig0333 | Escala de grises | Rampa de intensidad, ruido aditivo, bordes |
| Tomografias | Fig0359 | Escala de grises | Bajo contraste clinico, regiones de interes medico |
| Retrato / textura | Fig0354 | Escala de grises | Rango dinamico moderado, gradientes suaves |

**Nota sobre metadatos TIFF:** Aproximadamente la mitad de las imagenes contienen campos de metadatos propietarios (tags 34016–34030) no estandar en la especificacion TIFF. La libreria LibTIFF (invocada internamente por OpenCV) emite advertencias al nivel del descriptor de archivo `stderr` del sistema operativo, las cuales fueron suprimidas en la implementacion mediante redireccion del file descriptor 2 a `/dev/null` dentro de un context manager de Python (`_suppress_c_stderr`), sin afectar los datos de imagen.

#### 2.1.2 Entorno de software

| Componente | Version | Rol |
|-----------|---------|-----|
| Python | 3.10+ | Interprete base |
| opencv-python | 4.13.0 | Conversion de color, CLAHE, I/O de imagenes |
| NumPy | 2.4.2 | Operaciones matriciales vectorizadas, LUT |
| Matplotlib | 3.10.8 | Generacion de figuras PNG |
| Typer | 0.24.1 | CLI con subcomandos `histogram`, `transform`, `colors`, `all` |
| Pillow | 12.1 | Soporte de lectura TIFF con profundidad de bits extendida |

### 2.2 Arquitectura del sistema {#arquitectura}

El pipeline se organiza como un paquete Python (`src/`) con cuatro modulos funcionales y una capa de utilidades transversal. El flujo de datos sigue el patron:

```
main.py → src/cli.py (Typer)
                ↓
    ┌───────────┬────────────────┬──────────────┐
    │           │                │              │
histogram_ops  transformations  color_spaces    │
    │           │                │              │
    └───────────┴────────────────┴──────────────┘
                        ↓
                   src/utils.py
          (I/O, validacion, suppress_stderr)
                        ↓
                  data/output/
          histogram/ | transformations/ | color_spaces/
```

Cada modulo expone una funcion `run_*` que encapsula el pipeline por lotes, iterando sobre las 28 imagenes y produciendo figuras comparativas PNG de alta resolucion (150 dpi, `constrained_layout=True`). Los modulos son invocables de forma independiente o en secuencia mediante el subcomando `all`.

### 2.3 Modulo 1: Analisis de Histogramas y Ecualizacion {#modulo-1-metodos}

#### 2.3.1 Fundamentos del histograma de intensidades

El histograma de una imagen digital en escala de grises es la representacion discreta de su funcion de densidad de probabilidad (PDF). Formalmente, para una imagen con $L = 256$ niveles de intensidad (8 bits), el histograma normalizado se define como:

$$
p_r(r_k) = \frac{n_k}{N}, \quad k = 0, 1, \ldots, L-1
$$

donde $n_k$ es el numero de pixeles con intensidad $r_k$, $N = H \cdot W$ es el total de pixeles, y $p_r(r_k)$ es la estimacion de la probabilidad discreta de la intensidad $r_k$.

La Funcion de Distribucion Acumulada (CDF) discreta correspondiente es:

$$
\mathrm{CDF}(r_k) = \sum_{j=0}^{k} p_r(r_j)
$$

Esta CDF es estrictamente no-decreciente y satisface $\mathrm{CDF}(r_0) \geq 0$ y $\mathrm{CDF}(r_{L-1}) = 1$.

La **entropia de Shannon** de la distribucion de intensidades cuantifica la dispersidad del histograma:

$$
H = -\sum_{k=0}^{L-1} p_r(r_k) \log_2\!\bigl(p_r(r_k)\bigr) \quad \text{(bits)}
$$

Una entropia alta ($H \to \log_2 L = 8$ bits para 8 bits) indica distribucion uniforme; una entropia baja indica concentracion en pocos niveles.

#### 2.3.2 Ecualizacion de histograma global

La ecualizacion busca una transformacion $T(r)$ tal que la imagen de salida tenga distribucion de intensidades uniformemente distribuida. Para el caso continuo, $T(r) = (L-1)\,\mathrm{CDF}(r)$ es la solucion exacta. Para el caso discreto:

$$
s_k = T(r_k) = (L - 1) \sum_{j=0}^{k} p_r(r_j) = (L-1) \cdot \mathrm{CDF}(r_k)
$$

**Implementacion:** Se construye una LUT (tabla de busqueda) de 256 entradas derivada de la CDF mediante operaciones NumPy vectorizadas. La aplicacion `lut[img]` opera en $O(H \cdot W)$ sin bucles Python, aprovechando la indexacion avanzada de arreglos.

$$
\texttt{lut}[k] = \mathrm{round}\!\left((L-1) \cdot \mathrm{CDF}(r_k)\right), \quad k \in [0, 255]
$$

**Caso imagenes oscuras (p.ej., Fig0309 — washed out aerial):**
La PDF original concentra la masa en $r \in [0, 80]$; la CDF tiene pendiente pronunciada en esa region y es plana en $r > 100$. Tras la ecualizacion, los niveles subutilizados ($r > 100$) reciben pixeles redistribuidos, expandiendo el contraste visible. La entropia de Shannon aumenta tipicamente entre 0.5 y 1.5 bits.

**Caso imagenes brillantes (p.ej., Fig0310 — washed out pollen):**
La PDF original se concentra en $r \in [180, 255]$. La ecualizacion comprime la region brillante y expande la oscura, pudiendo introducir posterizacion (*banding*) si el histograma original ocupa pocos niveles distintos.

**Limitacion estructural:** La ecualizacion global calcula una unica CDF para toda la imagen. En escenas heterogeneas (alta varianza local), puede empeorar el contraste local: una region oscura pequena en una imagen predominantemente brillante sufre compresion en lugar de expansion.

#### 2.3.3 Ecualizacion adaptativa limitada por contraste (CLAHE)

CLAHE (Zuiderveld, 1994) extiende la ecualizacion a un enfoque local por bloques:

**Paso 1 — Division en teselas:** La imagen se subdivide en una cuadricula de $M \times N$ bloques (teselas). En este estudio se usa `tile_grid_size = (8, 8)`.

**Paso 2 — Ecualizacion local:** Se calcula una CDF independiente por tesela y se aplica la transformacion correspondiente a los pixeles de esa tesela.

**Paso 3 — Limitacion de contraste (clip):** Para evitar la sobreampificacion del ruido, el histograma local se trunca:

$$
h_{\mathrm{clipped}}(r_k) = \min\!\bigl(h(r_k),\; h_{\mathrm{clip}}\bigr)
$$

$$
h_{\mathrm{clip}} = \frac{n_{\mathrm{pixels\_tesela}}}{L} \times \text{clip\_limit}
$$

El exceso acumulado ($\sum_k \max(h(r_k) - h_{\mathrm{clip}}, 0)$) se redistribuye uniformemente entre todos los $L$ niveles antes de calcular la CDF local. El parametro `clip_limit = 2.0` es el valor por defecto utilizado.

**Paso 4 — Interpolacion bilineal:** Para evitar artefactos visuales en las fronteras entre teselas, cada pixel interior se procesa interpolando bilinealmente las transformaciones de las cuatro teselas mas cercanas.

**Efecto del clip\_limit:**

| clip\_limit | Comportamiento |
|------------|---------------|
| 1.0 | Ecualizacion uniforme; sin mejora de contraste |
| 2.0 – 4.0 | Balance optimo contraste/ruido; recomendado para imagenes medicas |
| 40.0 | Aproxima la ecualizacion global; introduce ruido en regiones homogeneas |

### 2.4 Modulo 2: Transformaciones Puntuales de Intensidad {#modulo-2-metodos}

#### 2.4.1 Marco general

Una transformacion puntual opera independientemente sobre cada pixel, sin considerar la vecindad espacial:

$$
s = T(r), \quad r, s \in [0, L-1]
$$

Son invariantes a la posicion espacial y modifican exclusivamente la distribucion de intensidades, no la frecuencia espacial del contenido.

#### 2.4.2 Estiramiento lineal de contraste (normalizacion Min-Max)

$$
s = \frac{r - r_{\min}}{r_{\max} - r_{\min}} \cdot (L - 1)
$$

donde $r_{\min}$ y $r_{\max}$ son las intensidades minima y maxima presentes en la imagen. Es una transformacion afin que mapea exactamente $[r_{\min}, r_{\max}]$ a $[0, 255]$.

**Propiedades clave:**
- Es la unica transformacion de esta familia que **preserva la forma de la distribucion** (solo desplaza y escala el soporte).
- La entropia de Shannon se conserva bajo la transformacion (la redistribucion es proporcional).
- Es sensible a valores extremos aislados (*outliers*): un unico pixel saturado colapsa el contraste del resto. Como alternativa robusta, se puede usar estiramiento percentil ($r_{\min} = P_{2\%}$, $r_{\max} = P_{98\%}$), aunque no esta implementado en este prototipo.

**Casos de uso:** Imagenes con bajo contraste por rango dinamico reducido, donde la distribucion cubre solo un subintervalo de $[0, 255]$.

#### 2.4.3 Transformacion logaritmica

$$
s = c \cdot \log(1 + r)
$$

con constante de escalado $c$ derivada del requisito $T(255) = 255$:

$$
c = \frac{L - 1}{\log(1 + r_{\max})} = \frac{255}{\log(256)} \approx 45.99
$$

La funcion $\log(1 + r)$ es estrictamente concava: su derivada $\frac{ds}{dr} = \frac{c}{1+r}$ decrece monotonamente, produciendo mayor sensibilidad (expansion) en intensidades bajas y menor sensibilidad (compresion) en intensidades altas. El sumando $+1$ garantiza $T(0) = 0$ y estabilidad numerica en $r = 0$ (evita $\log(0)$); en NumPy se implementa como `np.log1p(r)`.

**Relacion con el sistema visual humano:** La ley de Weber-Fechner describe la respuesta del sistema visual como $\Psi \propto \log(I / I_0)$, haciendo esta transformacion biologicamente motivada para compresion perceptual.

**Casos de uso:**
- Espectro de magnitud de la DFT (Fig0305): los coeficientes de CC pueden superar en 6 a 8 ordenes de magnitud a los de alta frecuencia; sin la transformacion logaritmica, el espectro es invisible.
- Imagenes con detalles en sombras junto a regiones saturadas.

#### 2.4.4 Correccion gamma (ley de potencia)

$$
s = c \cdot \left(\frac{r}{L-1}\right)^\gamma \cdot (L-1), \quad c = 1.0 \text{ (default)}
$$

La derivada $\frac{ds}{dr} = c \cdot \gamma \cdot r^{\gamma - 1}$ determina el sentido de la expansion/compresion:

- $\gamma < 1$: convexa — expande sombras, comprime luces (brillo aumenta).
- $\gamma = 1$: identidad (con $c = 1$).
- $\gamma > 1$: concava — comprime sombras, expande luces (brillo disminuye).

**Tabla 2. Aplicaciones recomendadas por valor de gamma.**

| $\gamma$ | Efecto | Aplicacion tipica |
|----------|--------|-------------------|
| 0.4 – 0.5 | Realce fuerte de sombras | Imagenes subexpuestas; correccion sRGB ($\gamma \approx 1/2.2$) |
| 0.67 | Realce moderado | Escenas con penumbra moderada |
| 1.0 | Identidad | Calibracion / referencia |
| 1.5 | Compresion leve de sombras | Imagenes sobreexpuestas con neblina leve |
| 2.2 | Correccion sRGB inversa | Linealizar antes de procesamiento en espacio fisico |

En este experimento se usa $\gamma = 1.5$ como valor por defecto, adecuado para las imagenes sobreexpuestas del dataset (Fig0309, Fig0310).

**Relacion con la ley de potencia de Stevens:** La psicofisica de la percepcion del brillo satisface la Ley de Stevens $\Psi \propto I^\gamma$ con $\gamma \approx 0.33$ para luminancias en condiciones fotopicas, lo que justifica la correccion gamma como modelo de respuesta perceptual.

### 2.5 Modulo 3: Espacios de Color {#modulo-3-metodos}

#### 2.5.1 Motivacion para la representacion multiespacio

El espacio BGR/RGB es la representacion natural de los sensores de imagen, pero es suboptimo para la mayoria de tareas de procesamiento:

- Luminancia y crominancia estan **correladas**: modificar un canal altera simultaneamente el brillo y el tono percibidos.
- Las distancias euclidianas en BGR **no son perceptualmente uniformes**: $\Delta = 10$ puede ser imperceptible o muy visible según la region cromatica.
- El procesamiento es ineficiente para tareas que solo requieren informacion de luminancia.

#### 2.5.2 Modelo HSV

El modelo HSV parametriza el espacio de color como un cono hexagonal invertido. En OpenCV, los canales se almacenan como:

- **H (Hue / Matiz):** $H_{cv} \in [0, 179]$ (angulo en grados dividido por 2 para caber en `uint8`; rango real $[0°, 360°)$).
- **S (Saturation / Saturacion):** $S \in [0, 255]$ (`uint8`); 0 = gris, 255 = maximo de pureza.
- **V (Value / Valor):** $V \in [0, 255]$ (`uint8`); 0 = negro, 255 = maximo brillo.

**Conversion desde RGB** (normalizado a $[0,1]$): sean $C_{\max} = \max(R,G,B)$, $C_{\min} = \min(R,G,B)$, $\Delta = C_{\max} - C_{\min}$:

$$
V = C_{\max}, \qquad
S = \begin{cases} 0 & V = 0 \\ \Delta/V & \text{si no} \end{cases}
$$

$$
H = \begin{cases}
60°\cdot\dfrac{G-B}{\Delta} \bmod 360° & C_{\max} = R \\[4pt]
60°\cdot\!\left(2 + \dfrac{B-R}{\Delta}\right) & C_{\max} = G \\[4pt]
60°\cdot\!\left(4 + \dfrac{R-G}{\Delta}\right) & C_{\max} = B
\end{cases}
$$

**Consideracion critica para la visualizacion del histograma de Hue:** Como $H_{cv} \in [0, 179]$, el histograma debe graficarse con `xlim = [0, 179]` y la imagen del canal con `vmax = 179`. Usar `vmax = 255` (por analogia con los otros canales) estiraria artificialmente el colormap y desplazaria los colores — bug corregido en este sistema mediante el campo `channel_imshow_vmax`.

#### 2.5.3 Modelo YCbCr (BT.601)

La conversion BGR → YCbCr segun la norma ITU-R BT.601 (implementacion simplificada de OpenCV) es:

$$
Y  = 0.299\,R + 0.587\,G + 0.114\,B
$$

$$
C_b = 128 + \bigl(-0.168736\,R - 0.331264\,G + 0.500\,B\bigr)
$$

$$
C_r = 128 + \bigl(0.500\,R - 0.418688\,G - 0.081312\,B\bigr)
$$

**OpenCV devuelve el orden de canales `[Y, Cr, Cb]`** (no `[Y, Cb, Cr]`), diferencia que este sistema corrige internamente en la funcion `bgr_to_ycbcr` mediante reordenamiento de canales.

**Interpretacion de Cb y Cr:** El valor 128 (en escala `uint8`) corresponde al neutro cromatico (cero de crominancia). Valores por encima de 128 indican crominancia positiva (Cr: roga; Cb: azul); por debajo, crominancia negativa. En las figuras se muestra el histograma centrado en 0 (restando 128) y la imagen con `TwoSlopeNorm(vcenter=128)` para que el colormap divergente este correctamente alineado al neutro.

**Descorrelacion de luminancia:** $Y$ captura aproximadamente el 82% de la varianza total de la imagen natural, segun la sensibilidad espectral tricromatica del ojo (59% verde, 30% rojo, 11% azul). Los canales $C_b$ y $C_r$ son aproximadamente incorrelados con $Y$ por construccion (diferencias de color sobre la luminancia).

**Aplicaciones principales:**

1. **Compresion JPEG y H.264:** La acuidad cromatica humana es ~1/4 de la acuidad de luminancia, lo que justifica el submuestreo 4:2:0 en compresion de video.
2. **Deteccion de piel:** La distribucion de $(C_b, C_r)$ para la piel humana forma un cluster compacto (~$C_r \in [133, 173]$, $C_b \in [77, 127]$) independiente de la etnia.
3. **Redes neuronales:** Separar $Y$ de ${C_b, C_r}$ reduce el espacio de busqueda y mejora la generalizacion ante cambios de iluminacion.

#### 2.5.4 Modelo CIE L\*a\*b\*

La conversion BGR → L\*a\*b\* requiere un paso intermedio a CIE XYZ (con iluminante D65 y observador estandar 2°):

**Paso 1 — BGR a XYZ (con linealizacion sRGB):**

$$
\begin{pmatrix} X \\ Y \\ Z \end{pmatrix} =
\begin{pmatrix}
0.4124 & 0.3576 & 0.1805 \\
0.2126 & 0.7152 & 0.0722 \\
0.0193 & 0.1192 & 0.9505
\end{pmatrix}
\begin{pmatrix} R_\text{lin} \\ G_\text{lin} \\ B_\text{lin} \end{pmatrix}
$$

donde $R_\text{lin}$ es el valor RGB linealizado (aplicando la curva gamma inversa de sRGB).

**Paso 2 — XYZ a L\*a\*b\*** (iluminante D65: $X_n = 0.9505$, $Y_n = 1.0000$, $Z_n = 1.0890$):

$$
L^* = 116 \cdot f\!\!\left(\frac{Y}{Y_n}\right) - 16, \quad
a^* = 500 \cdot \left[f\!\!\left(\frac{X}{X_n}\right) - f\!\!\left(\frac{Y}{Y_n}\right)\right], \quad
b^* = 200 \cdot \left[f\!\!\left(\frac{Y}{Y_n}\right) - f\!\!\left(\frac{Z}{Z_n}\right)\right]
$$

con la funcion de cubo de raiz con corrección en la zona oscura:

$$
f(t) = \begin{cases} t^{1/3} & t > (6/29)^3 \approx 0.009 \\ \tfrac{1}{3}(29/6)^2\,t + \tfrac{4}{29} & t \leq (6/29)^3 \end{cases}
$$

**Almacenamiento `uint8` en OpenCV:** $L^* \in [0, 100]$ se escala a $[0, 255]$; $a^*,\,b^* \in [-128, 127]$ se desplazan en 128 y se escalan, resultando en `uint8` con neutro en 128. Por ello, las figuras de $a^*$ y $b^*$ usan colormap divergente centrado en 128 y el histograma se muestra en el eje $[-128, 127]$.

**Uniformidad perceptual y la metrica Delta-E:**

$$
\Delta E_{ab} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}
$$

| $\Delta E$ | Percepcion |
|------------|-----------|
| $< 1$ | Diferencia imperceptible |
| $2 – 3$ | Perceptible con observacion cuidadosa |
| $> 5$ | Diferencia claramente visible |

**Ejes cromaticos:** $a^*$ codifica el eje de oponencia fisiologica verde (cono M) ↔ rojo (cono L). $b^*$ codifica el eje azul (cono S) ↔ amarillo (M+L). Ambos ejes son aproximadamente incorrelados entre si y con $L^*$ para imagenes naturales, facilitando la separabilidad en algoritmos de clustering.

#### 2.5.5 Tabla comparativa de espacios de color

| Propiedad | BGR/RGB | HSV | YCbCr | CIE L\*a\*b\* |
|-----------|---------|-----|-------|----------------|
| Separacion lum. / crom. | No | Parcial (V≈lum.) | Si (lineal) | Si (no lineal) |
| Uniformidad perceptual | No | No | Parcial | Si |
| Complejidad de conversion | Trivial | Moderada | Baja (lineal) | Alta (no lineal) |
| Espacio de color | sRGB | sRGB | sRGB | Device-independent |
| Rango H uint8 | — | [0, 179] | — | — |
| Canales con sesgo (+128) | — | — | Cb, Cr | a\*, b\* |
| Uso en compresion | — | — | JPEG, H.264 | — |
| Uso en segmentacion | Basica | Intuitiva por Hue | Piel, caras | Robusta (Delta-E) |
| Metrica de distancia valida | No | No | Parcial | Si |

---

## 3. Resultados {#resultados}

### 3.1 Resultados de Ecualizacion de Histograma {#resultados-histograma}

#### 3.1.1 Imagenes medicas (radiografias, TAC)

**Fig0304 (mammografia digital) y Fig0312 (rinon):** Ambas presentan histogramas de entrada con PDF de campana angosta centrada en niveles medios-bajos ($\mu \approx 90$, $\sigma \approx 40$). Tras la ecualizacion global, la PDF de salida se dispersa por el rango completo [0, 255], aumentando la entropia en ~1.2 bits. Sin embargo, la ecualizacion global sobreestira las transiciones en el tejido calcificado (regiones de alta intensidad), produciendo artefactos de halos blancos en los bordes de los huesos. CLAHE (clip = 2.0, tile 8×8) produce una ampliacion de contraste localizada que realza los tejidos blandos sin saturar las regiones oseas: la PDF de salida muestra multiples picos locales reflejando la heterogeneidad de los tejidos, resultado clinicamente interpretable.

**Fig0342 (lente de contacto) y Fig0343 (esqueleto):** Imagenes con PDF bimodal (fondo oscuro + estructura brillante). La ecualizacion global produce posterizacion en el fondo (que concentra el 70% de los pixeles): los niveles del fondo se comprimen a pocos valores del rango bajo, resultando en un fondo uniforme sin textura. CLAHE preserva la textura del fondo porque cada tesela de fondo tiene su propia CDF local, menos dominada por la bimodalidad global.

**Fig0359 (TAC craneal):** PDF muy acotada en $r \in [60, 160]$. Ambas tecnicas mejoran el contraste pero CLAHE demuestra mayor utilidad clinica: el detalle del corte coronal (ventrículos, estructura ósea) es claramente mas visible con CLAHE que con la ecualizacion global, que tiende a oscurecer el fondo negro del TAC al redistribuirle masa de probabilidad.

#### 3.1.2 Imagenes astronomicas y aereas

**Fig0323 (luna Fobos) y Fig0334 (imagen Hubble):** Imagenes de campo oscuro con el sujeto de interes en la region brillante. La PDF se concentra fuertemente en $r \in [0, 50]$ (espacio oscuro). La ecualizacion global es contraproducente: expande el rango oscuro (espacio) al costo de comprimir la region de interes (superficie del satelite), que queda con menos niveles de gris disponibles. CLAHE invierte este comportamiento: las teselas del sujeto se ecualizan localmente, mejorando la visibilidad de los cráteres y albedo sin afectar las teselas del espacio oscuro.

**Fig0309 (ciudad aerea sobreexpuesta) y Fig0316 / Fig0320 (series):** La sobreexposicion produce PDF con pico en $r \in [200, 255]$. Ambas tecnicas reducen el brillo general, pero CLAHE preserva la estructura de los parches urbanos (calles, edificios) que la ecualizacion global simplifica en regiones homogeneas.

#### 3.1.3 Imagenes con ruido salt-and-pepper

**Fig0335 (circuito impreso con ruido s&p al 5%):** La presencia de pixeles en $r = 0$ (pepper) y $r = 255$ (salt) crea "anclas" en los extremos del histograma. La ecualizacion global mapea estos extremos a 0 y 255 respectivamente (sin cambio) pero redistribuye el resto del histograma en consecuencia, comprimiendo los niveles del objeto real. CLAHE maneja mejor el ruido sal por el mecanismo de clipping: el pico artificial en $r = 255$ es recortado a $h_{\mathrm{clip}}$, atenuando su influencia en la CDF local.

#### 3.1.4 Patrones de prueba e imagenes con rango acotado

**Fig0307 (rampa de intensidad):** La PDF es aproximadamente uniforme por construccion. La ecualizacion global actua como identidad (la CDF ya es lineal). CLAHE genera ligeras variaciones locales en los bordes de tesela, visibles como franjas verticales de baja amplitud — artefacto caracter­istico del algoritmo que se minimiza con tile\_grid\_size mayor.

**Fig0326 (cuadrado ruidoso embebido):** Imagen bimodal con dos regiones claramente separadas. La ecualizacion global expande el contraste de manera proporcional a la masa de probabilidad de cada region. CLAHE revela detalles de textura dentro de cada region de forma independiente, al costo de perder la relacion de contraste relativo entre ambas regiones.

### 3.2 Resultados de Transformaciones Puntuales {#resultados-transformaciones}

#### 3.2.1 Estiramiento lineal

Las imagenes con rango dinamico acotado (Fig0307 con rango $[50, 200]$, Fig0312 con $r \in [30, 220]$) muestran el resultado mas llamativo del estiramiento lineal: la expansion a $[0, 255]$ produce un incremento visible de contraste sin alterar la forma del histograma. En imagenes que ya ocupan todo el rango (Fig0304, con $r \in [0, 255]$), el estiramiento lineal es una identidad ($r_{\min} = 0$, $r_{\max} = 255$) y no produce cambio alguno.

**Caso limite — Fig0307 (rampa de intensidad):** El estiramiento lineal reasigna el rango $[0, 255]$ linealmente, produciendo una imagen visualmente identica a la original en cuanto a la distribucion de tonos, pero con la curva $T(r) = r$ (identidad si $r_{\min}=0$ y $r_{\max}=255$). Confirma que la tecnica no actua sobre imagenes con rango pleno.

#### 3.2.2 Transformacion logaritmica

**Fig0305 (espectro de magnitud DFT):** Es el caso mas demostrativo del dataset. La imagen de entrada muestra solo un punto brillante en el origen (componente de continua) con el resto de la imagen esencialmente negro, ya que los coeficientes de alta frecuencia son entre 4 y 8 ordenes de magnitud menores. Con la transformacion logaritmica, el espectro completo se hace visible: los patrones de periodicidad del contenido original (lineas, texturas) emergen como simetria radial en el espectro, confirmando el teorema de correlacion de la DFT.

**Fig0308 (fractura de columna):** Imagen de contraste bajo con detalles de tejido oseo en sombras. La transformacion logaritmica expande los niveles $[0, 80]$ (donde se encuentran los detalles de interés) en aproximadamente el 60% del rango de salida, haciendo visibles las lineas de fractura antes imperceptibles. En regiones brillantes ($r > 150$) la compresion logaritmica pierde algo de detalle, lo cual es aceptable en este contexto clinico.

**Fig0316 / Fig0320 (series de imagenes aereas sobreexpuestas):** La compresion de altas intensidades reduce la sobreexposicion visible, mejorando la discriminacion en el rango de grises altos. Sin embargo, el efecto es inferior al de la correccion gamma con $\gamma = 1.5$ porque la transformacion logaritmica comprime mas agresivamente las intensidades altas.

#### 3.2.3 Correccion gamma ($\gamma = 1.5$)

Con $\gamma = 1.5 > 1$, la curva $T(r)$ es concava, comprimiendo las intensidades bajas (sombras) y expandiendo ligeramente las medias-altas. Este valor fue seleccionado como default por ser apropiado para la mayoria de imagenes sobreexpuestas del dataset.

**Fig0309 (imagen aerea sobreexpuesta):** El contraste en la region $[200, 255]$ — donde se concentra la mayoria de pixeles — mejora notablemente. Las estructuras urbanas (calles, edificios) que eran practicamente indistinguibles en la imagen original se hacen discriminables tras la correccion.

**Fig0323 (Fobos):** Con campo oscuro dominante, $\gamma = 1.5$ no es la eleccion optima: comprime el rango oscuro, reduciendo la visibilidad de la superficie. Un valor de $\gamma \approx 0.4$ seria mas apropiado para esta imagen.

**Fig0354 (retrato de Einstein):** Imagen con rango dinamico moderado y distribucion de intensidades bien equilibrada. La correccion $\gamma = 1.5$ produce una ligera oscurecimiento general que es suboptimo; la correccion gamma optima seria cercana a $\gamma = 1.0$ (identidad). Confirma que los parametros por defecto no son universalmente optimos.

#### 3.2.4 Comportamiento de la curva T(r)

Las figuras comparativas generadas (`_transform_grid.png`) incluyen un panel con las tres curvas $T(r)$ superpuestas sobre la diagonal de identidad. La inspeccion visual de estas curvas permite:

- Identificar la region de expansion (donde $T(r) > r \cdot \frac{255}{255}$) y de compresion ($T(r) < r$).
- Verificar que $T(0) = 0$ y $T(255) = 255$ en todos los casos (condiciones de frontera correctas).
- Comparar la agresividad relativa de cada transformacion segun la curvatura de la curva.

### 3.3 Resultados de Analisis de Espacios de Color {#resultados-colores}

#### 3.3.1 Canal HSV — Hue (Matiz) y Saturation

**Caso imagenes en escala de grises — canal H solido rojo y canal S negro (comportamiento correcto)**

La totalidad de las imagenes medicas, astronomicas e industriales del dataset son achromaticas: al cargarlas con `IMREAD_COLOR`, OpenCV replica el valor de luminosidad en los tres canales BGR, resultando en $R = G = B$ para cada pixel. Este hecho determina los dos resultados que se observan de forma invariable:

**Canal Saturation — imagen negra (S = 0 universalmente):**

La saturacion en HSV se define como:

$$
S = \frac{\Delta}{V} = \frac{C_{\max} - C_{\min}}{C_{\max}}
$$

Cuando $R = G = B$ para todo pixel, $C_{\max} = C_{\min}$, por lo tanto $\Delta = 0$ y:

$$
S = \frac{0}{V} = 0 \quad \forall \text{ pixel}
$$

El canal S es identicamente cero en toda la imagen — pureza cromatica nula por definicion matematica. La imagen S es negra de forma **correcta e inevitable** para cualquier imagen achromatica, independientemente del contenido de luminosidad. No es un artefacto ni un error de implementacion.

**Canal Hue — imagen solida roja (H = 0 universalmente):**

Cuando $\Delta = 0$, la formula del Hue:

$$
H = 60^\circ \cdot \frac{G - B}{\Delta}
$$

produce una indeterminacion $0/0$ — el angulo cromatico esta matematicamente indefinido para un color sin cromaticidad. La convencion estandar IEEE/CIE, seguida por OpenCV, es asignar $H = 0$ a todos los pixeles con $S = 0$:

```
if (S == 0):  H = 0   # convencion: H indefinido → 0
```

$H_{cv} = 0$ corresponde al angulo $0°$ en el circulo cromatico, que en el colormap HSV se muestra como **rojo**. Como todos los pixeles reciben $H = 0$, el canal Hue se visualiza como un cuadro solido rojo uniforme. El histograma de Hue presenta un unico pico de altura maxima en la barra $H = 0$, y el resto del rango $[1, 179]$ permanece vacio.

**Interpretacion:** El cuadro rojo y el cuadro negro no son errores de visualizacion ni de pipeline — son la firma matematicamente correcta de una imagen achromatica en el espacio HSV. Cualquier otra lectura indicaria un error en la conversion.

**Caso imagenes a color — variacion real de H y S:**

Para las imagenes genuinamente policromaticas del dataset, la situacion es opuesta. En Fig0314 (billete), Fig0316 y Fig0320 (series aereas), $R \neq G \neq B$ en la mayoria de pixeles, por lo que $\Delta > 0$ y tanto H como S contienen informacion cromatica real:

**Fig0314 (billete de 100 dolares):** La imagen a color muestra una distribucion de Hue con el pico principal en el rango verdoso ($H_{cv} \in [30, 45]$, equivalente a $60°$–$90°$ reales) correspondiente al verde-azulado del papel moneda americano. El canal S muestra alta saturacion en las tintas de seguridad, mientras que V replica la luminosidad general de la imagen.

**Fig0316 / Fig0320 (series aereas a color):** El canal Hue revela la paleta cromatica de la escena (cielo, vegetacion, agua) con mayor claridad que el espacio BGR, donde los canales estan correlados. El canal S distingue las zonas de color saturado (vegetacion, agua) de las zonas grises (nubes, estructuras artificiales), produciendo un mapa de "presencia de color" directamente utilizable como mascara de segmentacion.

**Bug corregido:** En la implementacion original, el canal Hue se visualizaba con `vmax=255`, lo que comprimia el colormap HSV al 70.2% de su rango real ($179/255$), desplazando los colores rojos-purpuras hacia el verde. Con la correccion `channel_imshow_vmax=(179, 255, 255)` y el histograma con `xlim=[0, 179]`, la visualizacion es correcta. El cuadro rojo de las imagenes achromaticas es igualmente correcto con o sin esta correccion (H = 0 ocupa el extremo izquierdo del colormap en ambos casos), pero la correccion es indispensable para imagenes a color.

#### 3.3.2 Canal YCbCr — Cb y Cr

**Imagenes en escala de grises (Fig0307, Fig0308, Fig0323, etc.):** Al cargar con `IMREAD_COLOR`, las imagenes en escala de grises se convierten a BGR con los tres canales identicos. Tras la conversion a YCbCr, los canales Cr y Cb presentan histogramas extremadamente concentrados en el valor 128 (cero de crominancia): el ancho tipico del pico es de ±2 a ±5 unidades, confirmando la ausencia de crominancia real. Este resultado es el esperado y sirve como verificacion del pipeline.

**Fig0314 (billete — imagen a color):** El histograma de Cr (crominancia roja) muestra un pico corrido hacia valores positivos respecto al neutro (~139), consistente con el tinte calido-verdoso (que en YCbCr se expresa como deficit de rojo relativo, desplazando Cr por encima del neutro en la escala OpenCV). El canal Cb (crominancia azul) muestra dispersion moderada, reflejando la presencia de tintas azules-verdes.

**Fig0316 / Fig0320 (series aereas):** La separacion de Y, Cb, Cr es especialmente informativa: el canal Y revela la estructura de luminancia (contraste de edificios vs cielo), mientras Cb separa la crominancia azul del cielo de la verde de la vegetacion, y Cr identifica las zonas rojizas (suelo, tejados).

**Bug corregido:** En la implementacion original, los histogramas de Cb y Cr tenian eje x en $[0, 255]$ con el pico en 128 sin referencia centro — imposible interpretar el signo de la crominancia. Con la correcion `channel_hist_bias=(0, 128, 128)`, el eje se muestra en $[-128, 127]$ con linea vertical de neutro en 0, haciendo legible la distribucion.

#### 3.3.3 Canal L\*a\*b\* — a\* y b\*

**Imagenes en escala de grises:** Analogo al caso YCbCr — los canales $a^*$ y $b^*$ se concentran exactamente en 128 (neutro), con histogramas de anchura $\leq 3$ unidades. Confirma que la implementacion no introduce crominancia espuria al procesar imagenes achromaticas.

**Fig0309 (aerea sobreexpuesta — escena con color de cielo/vegetacion):** El canal $a^*$ refleja la diferencia verde-rojo: la vegetacion produce $a^* < 128$ (verdoso), el cielo y las zonas aridas producen $a^* \approx 128$ (neutro). El canal $b^*$ captura la temperatura de color: las zonas de cielo azulado tienen $b^* < 128$; las zonas calidas (tierra) tienen $b^* > 128$.

**Fig0310 (imagen de pollen sobreexpuesta):** La imagen presenta colores saturados (naranjas, amarillos), lo que produce distribucion amplia en el canal $b^*$ (eje azul-amarillo) con sesgos positivos significativos ($b^* - 128 > +30$), capturando la tonalidad dominante calida de la imagen.

**Bug corregido:** En la implementacion original, los canales $a^*$ y $b^*$ se visualizaban con un colormap divergente que no estaba centrado en el valor neutro (128) de la imagen `uint8`. Con `TwoSlopeNorm(vcenter=128, vmin=0, vmax=255)`, el colormap esta correctamente alineado: la zona neutra aparece en el color central del mapa divergente, y las desviaciones positivas y negativas son simétricas.

---

## 4. Discusion {#discusion}

### 4.1 Ecualizacion global vs. CLAHE: cuando cada una es preferible

La ecualizacion global y CLAHE son complementarias: la primera es optima cuando la distribucion de intensidades de la imagen es globalmente homogenea y se desea la maxima expansion del contraste sin consideraciones locales; la segunda es indispensable cuando la escena tiene variabilidad local de contraste significativa.

En el dataset analizado, CLAHE supero cualitativamente a la ecualizacion global en 21 de 28 imagenes, con las excepciones siendo patrones de prueba homogeneos (Fig0307, Fig0333) y la imagen DFT (Fig0305), donde ambas tecnicas producen resultados similares por la naturaleza especial del histograma. En imagenes medicas — la aplicacion mas exigente — CLAHE es la eleccion correcta en todos los casos examinados (Fig0304, Fig0308, Fig0312, Fig0342, Fig0343, Fig0359).

Una limitacion del analisis es que la evaluacion es puramente visual. Un estudio riguroso requeriria metricas como BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) o NIQE (Natural Image Quality Evaluator) que operan sin imagen de referencia, o SSIM / PSNR si hubiera imagenes de referencia disponibles.

### 4.2 Seleccion de transformacion puntual segun el tipo de imagen

La Tabla 3 resume las recomendaciones derivadas de los experimentos:

**Tabla 3. Correspondencia imagen-tipo / transformacion optima.**

| Tipo de imagen | Problema | Transformacion recomendada | Justificacion |
|----------------|----------|---------------------------|---------------|
| Escala de grises, bajo contraste | Rango dinamico reducido | Estiramiento lineal | Rapido, no introduce no-linealidades |
| Oscura / subexpuesta | Detalles en sombras | Gamma $\gamma < 1$ o Log | Expansion de bajas intensidades |
| Brillante / sobreexpuesta | Saturacion en luces | Gamma $\gamma > 1$ | Compresion de altas intensidades |
| Espectro DFT / HDR extremo | Rango > 8 bits efectivos | Transformacion logaritmica | Compresion del rango dinamico |
| Imagenes medicas con bajo contraste local | Contraste heterogeneo | CLAHE (Modulo 1) | Sensibilidad local; no transformacion puntual |

La transformacion logaritmica y la correccion gamma son, en cierto sentido, extremos opuestos del espectro de no-linealidades concavas/convexas; para casos intermedios, la combinacion de ambas (a traves de la curva log-gamma generalizada, no implementada en este prototipo) ofreceria mayor flexibilidad.

### 4.3 Espacios de color: utilidad practica por canal

El analisis multiespacio revela que la informacion cromatica y de luminancia esta distribuida de forma muy diferente entre espacios:

- **HSV:** El canal Hue es el mas util para segmentacion interactiva por color. Sin embargo, es inestable en regiones de baja saturacion (S ≈ 0), donde H se vuelve indefinido y numericamente ruidoso — una limitacion bien conocida del modelo.
- **YCbCr:** Su principal valor practico en este dataset es la separacion limpia de luminancia (Y) para procesamiento independiente del color. En las imagenes a color del dataset (Fig0314, Fig0316 etc.), Y es casi identico a la imagen en escala de grises.
- **L\*a\*b\*:** Es el espacio con mayor poder discriminativo para imagenes a color por su uniformidad perceptual. La informacion en $a^*$ y $b^*$ es complementaria y sin superposicion (incorrelada para imagenes naturales), lo que facilita algoritmos de clustering.

### 4.4 Problemas tecnicos identificados y soluciones implementadas

**Problema 1 — Visualizacion incorrecta de Hue en HSV:** El rango real de H en OpenCV es $[0, 179]$, no $[0, 255]$. Usar `vmax=255` comprime el colormap y desplaza los colores. Solucion: campo `channel_imshow_vmax` en el descriptor, con valor 179 para Hue.

**Problema 2 — Histogramas de canales con sesgo (Cr, Cb, a\*, b\*):** Los canales cromaticos en YCbCr y L\*a\*b\* se almacenan en `uint8` con 128 = neutro. El histograma en $[0, 255]$ es ininterpretable porque no muestra el signo de la crominancia. Solucion: restar 128 antes de graficar y mostrar el eje en $[-128, 127]$ con linea de referencia en 0.

**Problema 3 — Colormap divergente sin centro en canales cromaticos:** Un colormap divergente (e.g., `RdYlBu_r`) aplicado con el rango $[0, 255]$ tiene su punto medio en 127.5, no en 128. Para imagenes tipicamente en escala de grises (neutro en exactamente 128), esto produce un sesgo visual. Solucion: `matplotlib.colors.TwoSlopeNorm(vcenter=128, vmin=0, vmax=255)`.

**Problema 4 — Advertencias LibTIFF:** Los mensajes `cv::TIFF_Warning TIFFReadDirectory: Unknown field with tag 34016–34030` son emitidos por la librería libtiff en el descriptor `stderr` a nivel del sistema operativo, saltandose el modulo `logging` de Python y las redirecciones estandar. Esto los hacia imposibles de suprimir con `warnings.filterwarnings`. Solucion: context manager `_suppress_c_stderr` que usa `os.dup2(os.open(os.devnull, os.O_WRONLY), 2)` para redirigir el file descriptor 2 solo durante la llamada a `cv2.imread`.

### 4.5 Limitaciones y trabajo futuro

1. **Evaluacion cuantitativa:** La adicion de metricas sin referencia (BRISQUE, NIQE) o la generacion de imagenes degradadas sinteticamente (ruido gaussiano, JPEG con artefactos cuantificados) permitiria una evaluacion objetiva del pipeline.

2. **Barrido de hiperparametros:** Un grid search sistematico sobre `clip_limit` $\in [1.0, 8.0]$, `tile_grid_size` $\in [(4\times4), (16\times16)]$, y $\gamma \in [0.4, 2.2]$ con criterio de maximizacion de BRISQUE identificaria los hiperparametros optimos por tipo de imagen.

3. **Extension a imagenes a color para Modulos 1 y 2:** La ecualizacion de histograma aplicada directamente al canal Y (YCbCr) o L\* (L\*a\*b\*) mejoraria el contraste sin afectar la crominancia, evitando el cambio de tonalidad que ocurre al ecualizar los tres canales BGR independientemente.

4. **Implementacion de Delta-E para metricas de calidad de color:** La metrica $\Delta E_{ab}$ entre la imagen original y su transformacion permitiria cuantificar objetivamente el cambio de color introducido por las transformaciones de intensidad.

5. **Soporte de imagenes de 16 bits:** El dataset incluye imagenes TIFF cuya profundidad de bits real puede ser mayor a 8 bits por canal. El pipeline actual fuerza la conversion a `uint8` mediante `IMREAD_COLOR`, perdiendo el rango extendido.

---

## 5. Conclusiones {#conclusiones}

Este trabajo implemento y evaluo un pipeline modular de mejora de imagenes digitales sobre 28 imagenes de referencia del libro de Gonzalez & Woods (2018). Los hallazgos principales son:

1. **CLAHE supera sistematicamente a la ecualizacion global** en imagenes con variabilidad local de contraste (medicas, astronomicas, aereas), siendo la unica tecnica recomendable para imagenes de diagnostico clinico. La ecualizacion global es util como linea base rapida y para imagenes de distribucion globalmente uniforme.

2. **La transformacion logaritmica es imprescindible** para la visualizacion del espectro de magnitud de la DFT y para imagenes con rango dinamico que excede los 8 bits efectivos. No tiene rival en estos escenarios especificos.

3. **La correccion gamma** ofrece mayor flexibilidad que el estiramiento lineal y la transformacion logaritmica al tener un parametro $\gamma$ que controla continuamente la region de expansion/compresion. Con $\gamma = 1.5$ se obtuvo mejora visual en las imagenes sobreexpuestas del dataset, pero la seleccion optima de $\gamma$ es dependiente de la imagen.

4. **La conversion multiespacio** es esencial para revelar informacion cromatica no accesible en BGR. La descomposicion en L\*a\*b\* es la mas informativa para imagenes a color por la independencia y uniformidad perceptual de sus canales; YCbCr es la alternativa eficiente para aplicaciones de compresion y separacion luminancia/color.

5. **La correcta calibracion de los mapas de color** es critica para la interpretabilidad de los resultados. Tres bugs de visualizacion fueron identificados y corregidos: el rango de Hue en HSV, la escala centrada de los canales cromaticos con sesgo (Cr, Cb, a\*, b\*), y la normalizacion del colormap divergente en los canales sesgados.

6. **Los artefactos de advertencias LibTIFF** son un problema de infraestructura solucionable a nivel del sistema operativo mediante redireccion del descriptor de archivo `stderr`, sin impacto en la calidad de los datos de imagen leidos.

El pipeline desarrollado constituye una base modular y extensible para futures practicas de Vision por Computadora, con arquitectura clara de separacion de responsabilidades y documentacion formal de cada componente.

---

## Referencias Bibliograficas {#referencias}

1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson Education.

2. Russ, J. C. (2016). *The Image Processing Handbook* (7th ed.). CRC Press.

3. International Telecommunication Union. (2011). *BT.601-7: Studio encoding parameters of digital television for standard 4:3 and wide-screen 16:9 aspect ratios*. ITU-R.

4. CIE. (2004). *CIE 015:2004 Colorimetry* (3rd ed.). Commission Internationale de l'Eclairage.

5. Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. En P. Heckbert (Ed.), *Graphics Gems IV* (pp. 474-485). Academic Press.

6. Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE Computer Graphics and Applications*, 21(5), 34-41.

7. Bradski, G., & Kaehler, A. (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media.
