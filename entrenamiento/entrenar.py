"""Maestro-alumno en 5 folds: maestro, etiquetas, alumno, convnext (o todo)."""
import gc
import sys

import numpy as np
import pandas as pd
import torch

from config import ETIQUETAS, INDICE, LABELS, RUNS
from entrenamiento import comun as C

FOLDS = [0, 1, 2, 3, 4]
DEBILES = "etiquetas_hibrido_gemma"            # híbrido reglas + embeddings (etiquetas/embeddings.py)
NUEVAS = "etiquetas_maestro"                   # maestro-alumno sobre DEBILES
ALFA = 0.5                                     # peso de la etiqueta débil en la mezcla


def run(etapa, k):
    return f"{RUNS}/{etapa}_f{k}"


def liberar():
    gc.collect()
    torch.cuda.empty_cache()


def entrenar_fold(etapa, k):
    perdida = C.PerdidaFocal(gamma=2.0)
    if etapa == "maestro":
        datos = C.preparar_datos(fold_val=k, etiquetas=DEBILES, etiquetas_val=DEBILES, clase_dataset=C.RodillaDatasetZoom)
        C.entrenar(C.RedZoom(), datos, run(etapa, k), epocas=30, lr=1e-3, wd=1e-4, perdida_fn=perdida)
    elif etapa == "alumno":
        datos = C.preparar_datos(fold_val=k, etiquetas=NUEVAS, etiquetas_val=DEBILES, clase_dataset=C.RodillaDatasetZoom)
        C.entrenar(C.RedZoom(), datos, run(etapa, k), epocas=30, lr=1e-3, wd=1e-4, perdida_fn=perdida)
    elif etapa == "convnext":
        datos = C.preparar_datos(fold_val=k, etiquetas=NUEVAS, etiquetas_val=DEBILES, clase_dataset=C.RodillaDatasetZoom,
                                 batch=2)
        C.entrenar(C.RedPreentrenada("convnext_tiny"), datos, run(etapa, k), epocas=15, lr=1e-3, lr_backbone=1e-4,
                   wd=1e-4, perdida_fn=perdida, acumular=4)
    else:
        raise SystemExit(f"etapa desconocida: {etapa}")
    del datos
    liberar()


def a_escala_debil(p, ref_maestro, ref_debil):
    """Cada valor del maestro → el valor de la etiqueta débil en el mismo percentil (silver como referencia)."""
    u = np.searchsorted(np.sort(ref_maestro), p, side="right") / len(ref_maestro)
    return np.quantile(ref_debil, np.clip(u, 0, 1))


def etiquetas_maestro():
    indice = pd.read_csv(INDICE, dtype={"StudyInstanceUID": str}, keep_default_na=False).set_index("StudyInstanceUID")
    silver = indice[indice.split == "silver"]
    ids, oof = [], []
    for k in FOLDS:
        datos = C.preparar_datos(fold_val=k, etiquetas=DEBILES, etiquetas_val=DEBILES, clase_dataset=C.RodillaDatasetZoom)
        p = C.cargar_predicciones(run("maestro", k), datos, C.RedZoom)
        assert datos["val_ids"] == silver.index[silver.fold == k].tolist()
        ids += datos["val_ids"]
        oof.append(p["val"])
        del datos
        liberar()
    maestro = pd.DataFrame(np.concatenate(oof), index=ids, columns=LABELS)
    debiles = pd.read_csv(f"{ETIQUETAS}/{DEBILES}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS]
    gold_ids = pd.read_csv(f"{ETIQUETAS}/gold.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID
    for l in LABELS:
        maestro[l] = a_escala_debil(maestro[l].to_numpy(), maestro[l].to_numpy(), debiles.loc[ids, l].to_numpy())
    nuevas = ALFA * debiles.loc[ids] + (1 - ALFA) * maestro
    salida = pd.concat([nuevas, debiles.loc[gold_ids]]).rename_axis("StudyInstanceUID")   # filas gold: no se entrenan
    salida.reset_index().to_csv(f"{ETIQUETAS}/{NUEVAS}.csv", index=False)
    print(f"{len(nuevas)} estudios silver → {ETIQUETAS}/{NUEVAS}.csv")


def main(args=None):
    args = sys.argv[1:] if args is None else args
    etapa, folds = args[0], [int(k) for k in args[1:]] or FOLDS
    etapas = ["maestro", "etiquetas", "alumno", "convnext"] if etapa == "todo" else [etapa]
    for e in etapas:
        if e == "etiquetas":
            print("\n=== etiquetas del maestro ===", flush=True)
            etiquetas_maestro()
            continue
        for k in folds:
            print(f"\n=== {e} · fold {k} ===", flush=True)
            entrenar_fold(e, k)


if __name__ == "__main__":
    main()
