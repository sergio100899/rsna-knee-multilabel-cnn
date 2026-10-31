# evaluacion/ (evaluation)

**English** · [Español](README.es.md)

Evaluates the models in `data/runs/` on the 58 gold studies. Tables and figures go to `data/resultados/`.

```bash
uv run python -m evaluacion.metricas   # with uv; with pip: python -m evaluacion.metricas
uv run python -m evaluacion.sesgo
uv run python -m evaluacion.gradcam
```

| Script | What it does | Output |
|---|---|---|
| `metricas.py` | Per-finding and macro AUC with 95 % CI (bootstrap), Youden threshold chosen out of fold → precision, recall and F1; prevalence and rare findings | `auc.csv`, `resumen.csv`, `metricas_umbral.csv`, `umbrales.csv`, `prevalencia.png`, `auc_por_hallazgo.png` |
| `sesgo.py` | Out-of-fold AUC of ConvNeXt by report language and scanner manufacturer | `sesgo.csv`, `sesgo.png` |
| `gradcam.py` | HiResCAM maps for ACL, medial meniscus, effusion and Baker's cyst, and % of the map in each plane | `gradcam_*.png`, `gradcam_planos.csv` |

Each model is the ensemble of its 5 folds (mean of probabilities). Predictions are computed once and stored in `predicciones.npz` inside each run.
