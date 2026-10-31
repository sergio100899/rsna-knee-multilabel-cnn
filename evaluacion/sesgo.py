"""AUC fuera de fold de ConvNeXt por idioma del informe y por fabricante."""
import os

import matplotlib
import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect

from config import ETIQUETAS, INDICE, LABELS, RESULTADOS, RUNS, TRAIN_CSV
from entrenamiento import comun as C
from entrenamiento.entrenar import DEBILES, FOLDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DetectorFactory.seed = 0
NOMBRES = {"en": "inglés", "es": "español", "tr": "turco", "hr": "croata/serbio", "el": "griego", "de": "alemán",
           "bg": "búlgaro", "nl": "neerlandés", "fr": "francés"}


def idioma(texto):
    try:
        return detect(str(texto))
    except Exception:
        return "?"


def main():
    os.makedirs(RESULTADOS, exist_ok=True)
    indice = pd.read_csv(INDICE, dtype={"StudyInstanceUID": str}, keep_default_na=False).set_index("StudyInstanceUID")
    informes = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").Report
    ids, P = [], []
    for k in FOLDS:                                    # orden de validación de preparar_datos: el del índice
        ids += indice.index[(indice.split == "silver") & (indice.fold == k)].tolist()
        P.append(np.load(f"{RUNS}/convnext_f{k}/predicciones.npz")["val"])
    P = pd.DataFrame(np.concatenate(P), index=ids, columns=LABELS)
    ref = pd.read_csv(f"{ETIQUETAS}/{DEBILES}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    Y = (ref.loc[ids, LABELS] >= 0.5).astype(int)
    global_ = C.auc_macro(Y.values, P.values)

    grupos = {"idioma": informes.loc[ids].map(idioma).map(lambda c: NOMBRES.get(c, c)),
              "fabricante": indice.loc[ids, "manufacturer"]}
    filas = []
    for tipo, s in grupos.items():
        for valor, ii in s.groupby(s).groups.items():
            if len(ii) >= 150:
                filas.append({"tipo": tipo, "grupo": valor, "estudios": len(ii),
                              "AUC OOF": C.auc_macro(Y.loc[ii].values, P.loc[ii].values)})
    t = pd.DataFrame(filas).sort_values(["tipo", "estudios"], ascending=[False, False])
    t.to_csv(f"{RESULTADOS}/sesgo.csv", index=False)
    print(f"AUC OOF global: {global_:.3f}\n")
    print(t.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]}, sharey=True)
    for a, tipo, color in zip(ax, ["idioma", "fabricante"], ["#2a78d6", "#1baf7a"]):
        g = t[t.tipo == tipo]
        a.bar(range(len(g)), g["AUC OOF"], color=color)
        a.set_xticks(range(len(g)), [f"{v}\n{n}" for v, n in zip(g.grupo, g.estudios)], rotation=30, ha="right")
        a.axhline(global_, color="gray", ls=":", lw=1)
        a.set_title(f"por {tipo}")
    ax[0].set(ylabel="AUC fuera de fold (ConvNeXt)", ylim=(0.7, 0.9))
    plt.tight_layout()
    plt.savefig(f"{RESULTADOS}/sesgo.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
