# data/

Resonancias magnéticas de rodilla en DICOM de la competición RSNA Knee Abnormality Detection, con el informe radiológico de cada estudio y 12 etiquetas de radiólogo en 58 de ellos.

Estructura al descomprimir el zip en `data/rsna/`:

```
data/rsna/
├── train.csv                 un estudio por fila: informe (Report) y 12 etiquetas (solo 58 estudios)
├── train_series.csv          una serie por fila: plano y tipo de secuencia
├── train_series/
│   └── <estudio>/<serie>/<corte>.dcm
├── test.csv
├── test_series.csv
├── test_series/
│   └── <estudio>/<serie>/<corte>.dcm
└── sample_submission.csv
```
