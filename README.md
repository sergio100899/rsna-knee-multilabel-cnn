# Clasificación multietiqueta de anomalías de rodilla en resonancia magnética con CNN y aprendizaje débilmente supervisado

**Examen parcial de Redes Neuronales y Aprendizaje Profundo · Pregunta 1 · UNI FIIS 2026-II**

Predicción de 12 hallazgos en resonancias de rodilla ([RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)). Solo 58 de 4407 estudios tienen etiqueta de radiólogo; el resto se etiqueta a partir de los informes en texto libre (etiquetas débiles). Se comparan dos arquitecturas: una CNN 2.5D desde cero y ConvNeXt-Tiny preentrenada en ImageNet.

## Estructura

```
.
├── config.py    hallazgos y rutas comunes
├── data/        dataset y archivos generados · data/README.md
├── etiquetas/   etiquetas débiles · etiquetas/README.md
├── imagenes/    dataset de imágenes y folds · imagenes/README.md
└── informe/     informe en PDF (formato NeurIPS)
```

## Instalación

Requisitos: Python 3.13 y GPU NVIDIA con CUDA 12.8.

**Con uv**

```bash
uv sync
```

**Con pip**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Descargar el dataset

1. Aceptar las reglas de la competición en <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules>.
2. Guardar el token de la API de Kaggle (<https://www.kaggle.com/settings> → *Create New Token*) en `~/.kaggle/kaggle.json`.
3. Descargar (~265 GB) y descomprimir:

```bash
pip install kaggle
kaggle competitions download -c rsna-knee-abnormality-detection -p data
unzip -q data/rsna-knee-abnormality-detection.zip -d data/rsna
```

Estructura del dataset: [data/README.md](data/README.md)

## Orden de ejecución

Los comandos se ejecutan desde la raíz del repositorio.

1. Etiquetas débiles: [etiquetas/README.md](etiquetas/README.md)
2. Dataset de imágenes: [imagenes/README.md](imagenes/README.md)
