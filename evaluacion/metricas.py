"""AUC con IC95 y P/R/F1 (umbral de Youden) de los dos ensambles en gold."""
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from config import LABELS, RESULTADOS
from entrenamiento import comun as C
from entrenamiento.entrenar import DEBILES, FOLDS, run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELOS = {"CNN desde cero": ("alumno", C.RedZoom),
           "ConvNeXt-Tiny": ("convnext", lambda: C.RedPreentrenada("convnext_tiny", preentrenada=False))}


def cargar():
    """Etiquetas OOF (débiles binarizadas) y gold, y por modelo: predicción OOF y ensamble en gold."""
    y_oof, oof, gold = [], {n: [] for n in MODELOS}, {n: [] for n in MODELOS}
    for k in FOLDS:
        datos = C.preparar_datos(fold_val=k, etiquetas=DEBILES, etiquetas_val=DEBILES, clase_dataset=C.RodillaDatasetZoom,
                                 batch=2)
        y_oof.append(datos["y_val"])
        y_gold = datos["y_gold"]
        for n, (etapa, crear) in MODELOS.items():
            p = C.cargar_predicciones(run(etapa, k), datos, crear)
            oof[n].append(p["val"])
            gold[n].append(p["gold"])
        del datos
    P = {n: {"oof": np.concatenate(oof[n]), "gold": np.mean(gold[n], 0)} for n in MODELOS}
    return np.concatenate(y_oof), y_gold, P


def ic95(f, y, p, repeticiones=2000, semilla=0):
    rng, v = np.random.default_rng(semilla), []
    for _ in range(repeticiones):
        i = rng.integers(0, len(y), len(y))
        try:
            v.append(f(y[i], p[i]))
        except ValueError:                                          # remuestreo sin positivos o sin negativos
            pass
    return np.percentile(v, [2.5, 97.5])


def umbral_youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def metricas(y, p, umbrales):
    filas = []
    for j, l in enumerate(LABELS):
        pred = (p[:, j] >= umbrales[j]).astype(int)
        filas.append({"hallazgo": l, "umbral": umbrales[j],
                      "precisión": precision_score(y[:, j], pred, zero_division=0),
                      "recall": recall_score(y[:, j], pred, zero_division=0),
                      "F1": f1_score(y[:, j], pred, zero_division=0),
                      "especificidad": ((pred == 0) & (y[:, j] == 0)).sum() / max((y[:, j] == 0).sum(), 1)})
    t = pd.DataFrame(filas).set_index("hallazgo")
    t.loc["MACRO"] = t.mean()
    return t


def main():
    os.makedirs(RESULTADOS, exist_ok=True)
    y_oof, y_gold, P = cargar()

    prev = pd.DataFrame({"silver %": 100 * y_oof.mean(0), "gold %": 100 * y_gold.mean(0)}, index=LABELS)
    raros = list(prev.index[prev["silver %"] < 15])                        # decidido en silver, no en gold
    frecuentes = [l for l in LABELS if l not in raros]

    auc = pd.DataFrame(index=LABELS + ["MACRO"])
    for n in MODELOS:
        auc[n] = [roc_auc_score(y_gold[:, j], P[n]["gold"][:, j]) for j in range(12)] + [C.auc_macro(y_gold, P[n]["gold"])]
        auc[f"{n} IC95"] = (["{:.2f}–{:.2f}".format(*ic95(roc_auc_score, y_gold[:, j], P[n]["gold"][:, j])) for j in range(12)]
                            + ["{:.3f}–{:.3f}".format(*ic95(C.auc_macro, y_gold, P[n]["gold"]))])
    auc.insert(0, "positivos gold", list(y_gold.sum(0)) + [y_gold.sum()])
    auc.to_csv(f"{RESULTADOS}/auc.csv")

    umbrales = {n: np.array([umbral_youden(y_oof[:, j], P[n]["oof"][:, j]) for j in range(12)]) for n in MODELOS}
    pd.DataFrame(umbrales, index=LABELS).to_csv(f"{RESULTADOS}/umbrales.csv")
    detalle = pd.concat({n: metricas(y_gold, P[n]["gold"], umbrales[n]) for n in MODELOS}, axis=1)
    detalle.to_csv(f"{RESULTADOS}/metricas_umbral.csv")

    resumen = []
    for n in MODELOS:
        m = detalle[n].loc["MACRO"]
        resumen.append({"modelo": n, "AUC macro gold": auc.loc["MACRO", n], "IC95": auc.loc["MACRO", f"{n} IC95"],
                        "AUC macro OOF": C.auc_macro(y_oof, P[n]["oof"]),
                        **m[["precisión", "recall", "F1", "especificidad"]].to_dict(),
                        "AUC gold raros": auc.loc[raros, n].mean(), "AUC gold frecuentes": auc.loc[frecuentes, n].mean()})
    resumen = pd.DataFrame(resumen).set_index("modelo")
    resumen.to_csv(f"{RESULTADOS}/resumen.csv")
    print(f"raros (< 15 % en silver): {raros}\n")
    print(resumen.round(3).to_string())

    o = prev.sort_values("silver %").index
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.2, prev.loc[o, "silver %"], 0.4, color="#2a78d6", label="silver (etiqueta débil ≥ 0,5)")
    ax.bar(x + 0.2, prev.loc[o, "gold %"], 0.4, color="#eb6834", label="gold (radiólogo)")
    ax.axhline(15, color="gray", ls=":", lw=1)
    ax.set_xticks(x, o, rotation=30, ha="right")
    ax.set(ylabel="% de estudios positivos", title="Prevalencia por hallazgo (línea: umbral de 'raro', 15 %)")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{RESULTADOS}/prevalencia.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, (n, color) in enumerate(zip(MODELOS, ["#2a78d6", "#8e5ad6"])):
        ax.bar(x + (i - 0.5) * 0.4, auc.loc[LABELS, n], 0.4, color=color, label=f"{n} (macro {auc.loc['MACRO', n]:.3f})")
    ax.set_xticks(x, LABELS, rotation=30, ha="right")
    ax.set(ylabel="AUC en gold", ylim=(0.4, 1))
    ax.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{RESULTADOS}/auc_por_hallazgo.png", dpi=150)
    plt.close()
    print(f"\ntablas y figuras → {RESULTADOS}/")


if __name__ == "__main__":
    main()
