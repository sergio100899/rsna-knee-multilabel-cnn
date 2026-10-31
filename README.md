# Multilabel classification of knee MRI abnormalities with CNNs and weakly supervised learning

**English** · [Español](README.es.md)

**Midterm exam, Neural Networks and Deep Learning · Question 1 · UNI FIIS 2026-II**

Prediction of 12 findings in knee MRI ([RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)). Only 58 of 4407 studies have radiologist labels; the rest are labeled from the free-text reports (weak labels). Two architectures are compared: a 2.5D CNN trained from scratch and ConvNeXt-Tiny pretrained on ImageNet.

## Structure

```
.
├── config.py        findings and shared paths
├── data/            dataset and generated files · data/README.md
├── main.py          the whole pipeline in one command
├── etiquetas/       weak labels · etiquetas/README.md
├── imagenes/        image dataset and folds · imagenes/README.md
├── entrenamiento/   teacher-student training of both architectures · entrenamiento/README.md
├── evaluacion/      gold metrics, bias and Grad-CAM · evaluacion/README.md
└── informe/         report in PDF (NeurIPS format)
```

## Installation

Requirements: Python 3.13 and an NVIDIA GPU with CUDA 12.8.

**With uv**

```bash
uv sync
```

**With pip**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Download the dataset

1. Accept the competition rules at <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules>.
2. Save your Kaggle API token (<https://www.kaggle.com/settings> → *Create New Token*) in `~/.kaggle/kaggle.json`.
3. Download (~265 GB) and unzip:

```bash
pip install kaggle
kaggle competitions download -c rsna-knee-abnormality-detection -p data
unzip -q data/rsna-knee-abnormality-detection.zip -d data/rsna
```

Dataset structure: [data/README.md](data/README.md)

## How to run

Commands are run from the repository root. Everything in one command:

```bash
uv run python main.py   # with uv
python main.py          # with pip (with the .venv environment activated)
```

Or step by step:

1. Weak labels: [etiquetas/README.md](etiquetas/README.md)
2. Image dataset: [imagenes/README.md](imagenes/README.md)
3. Training: [entrenamiento/README.md](entrenamiento/README.md)
4. Evaluation: [evaluacion/README.md](evaluacion/README.md)
