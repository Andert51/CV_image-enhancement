<div align="center">

# 🖼️ Image Enhancement Pipeline

### Computer Vision · Practice 1

A modular image-processing pipeline that implements **histogram equalization**, **point-intensity transformations**, and **color-space conversions** — built for the *Digital Image Processing* course.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-11557C?style=flat)](https://matplotlib.org/)

</div>

---

## ✨ Features

| Module | What it does |
|--------|-------------|
| **Histogram** | Global histogram equalization + adaptive CLAHE with comparison grids |
| **Transformations** | Linear stretch, logarithmic mapping, and gamma (power-law) correction |
| **Color Spaces** | Channel-level decomposition into **HSV**, **YCbCr**, and **CIE L\*a\*b\*** |

All results are saved automatically to `data/output/` with clean naming conventions.

---

## 📂 Project Structure

```
T1_MejoraImagenes/
├── main.py                      # Entry point — delegates to src/cli.py
├── requirements.txt
│
├── src/
│   ├── cli.py                   # Typer CLI (histogram | transform | colors | all)
│   ├── histogram_ops.py         # Module 1 · Histograms & equalization
│   ├── transformations.py       # Module 2 · Point-intensity transformations
│   ├── color_spaces.py          # Module 3 · Color-space conversions
│   └── utils.py                 # Shared I/O, logging, and validation helpers
│
├── data/
│   ├── DIP3E_Original_Images_CH03/   # Test images (Gonzalez & Woods, 4th ed.)
│   └── output/
│       ├── histogram/
│       ├── transformations/
│       └── color_spaces/
│           ├── hsv/
│           ├── lab/
│           └── ycbcr/
│
└── docs/
    ├── ARCHITECTURE.md
    └── REPORTE_IMARD.md
```

---

## 🚀 Getting Started

### 1 · Clone & create virtual environment

```bash
git clone https://github.com/Andert51/CV_image-enhancement.git
cd CV_image-enhancement

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Requires Python ≥ 3.10**

---

## 🖥️ Usage

```
python main.py [OPTIONS] COMMAND
```

| Command | Description |
|---------|-------------|
| `all` | Run the full pipeline (histogram → transform → colors) |
| `histogram` | Histogram analysis and equalization only |
| `transform` | Point-intensity transformations only |
| `colors` | Color-space conversion and analysis only |

### Global flag

```bash
python main.py --verbose COMMAND   # Enable DEBUG logging
```

### Examples

```bash
# Run everything with defaults
python main.py all

# Histogram with a custom CLAHE clip limit
python main.py histogram --clip-limit 3.0

# Gamma correction with γ = 0.5 (brighten dark images)
python main.py transform --gamma 0.5

# Color-space analysis only
python main.py colors

# Help for any subcommand
python main.py transform --help
```

---

## 🔬 Module Details

### 📊 Module 1 — Histogram Equalization

Implements histogram equalization using the CDF-derived look-up table:

$$s_k = (L-1) \sum_{j=0}^{k} p_r(r_j)$$

Also applies **CLAHE** (`cv2.createCLAHE`) for adaptive local contrast enhancement. Outputs comparison figures showing original, equalized, and CLAHE results side-by-side with overlay histograms.

---

### 🌗 Module 2 — Point-Intensity Transformations

Three intensity mappings $s = T(r)$, all operating on 8-bit grayscale images:

| Transformation | Formula |
|---------------|---------|
| Linear Stretch | $s = \dfrac{r - r_{\min}}{r_{\max} - r_{\min}} \cdot 255$ |
| Logarithmic | $s = c \cdot \log(1 + r)\,,\quad c = \dfrac{255}{\log(256)}$ |
| Gamma / Power-Law | $s = c \cdot r^{\,\gamma}$ |

- $\gamma < 1$ → expands shadows (brightens dark images)
- $\gamma > 1$ → expands highlights (darkens bright images)
- $\gamma = 1$ → identity

Outputs a 2×5 comparison grid plus a $T(r)$ curves panel.

---

### 🎨 Module 3 — Color Space Conversions

Converts BGR images to three color spaces and exports each channel individually:

| Space | Channels |
|-------|---------|
| **HSV** | Hue · Saturation · Value |
| **YCbCr** | Luma · Cb · Cr |
| **CIE L\*a\*b\*** | Lightness · a\* · b\* |

Each conversion produces per-channel images plus a 2×4 comparison figure.

---

## 📦 Dependencies

| Library | Min. Version | Purpose |
|---------|-------------|---------|
| `opencv-python` | 4.8.0 | Image I/O, color conversions, CLAHE |
| `numpy` | 1.26.0 | Vectorized array operations |
| `matplotlib` | 3.8.0 | Figure generation (non-interactive `Agg` backend) |
| `seaborn` | 0.13.0 | Statistical plot styling |
| `scikit-image` | 0.22.0 | Additional image quality metrics |
| `scipy` | 1.11.0 | Advanced scientific processing |
| `typer[all]` | 0.9.0 | CLI framework |

---

## 📖 Reference

> Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
> — Sections 3.2 (intensity transformations) and 3.3 (histogram processing).

---

<div align="center">
Made with 🧠 and ☕ for the Computer Vision course
</div>
