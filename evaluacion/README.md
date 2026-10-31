# evaluacion/

Evalúa los modelos de `data/runs/` en los 58 estudios gold. Tablas y figuras en `data/resultados/`.

```bash
uv run python -m evaluacion.metricas   # con uv; con pip: python -m evaluacion.metricas
uv run python -m evaluacion.sesgo
uv run python -m evaluacion.gradcam
```

| Script | Qué hace | Salida |
|---|---|---|
| `metricas.py` | AUC por hallazgo y macro con IC95 (bootstrap), umbral de Youden elegido fuera de fold → precisión, recall y F1; prevalencia y hallazgos raros | `auc.csv`, `resumen.csv`, `metricas_umbral.csv`, `umbrales.csv`, `prevalencia.png`, `auc_por_hallazgo.png` |
| `sesgo.py` | AUC fuera de fold de ConvNeXt por idioma del informe y por fabricante del escáner | `sesgo.csv`, `sesgo.png` |
| `gradcam.py` | Mapas HiResCAM en LCA, menisco medial, derrame y quiste de Baker, y % del mapa en cada plano | `gradcam_*.png`, `gradcam_planos.csv` |

Cada modelo es el ensamble de sus 5 folds (promedio de probabilidades). Las predicciones se calculan una vez y se guardan en `predicciones.npz` dentro de cada run.
