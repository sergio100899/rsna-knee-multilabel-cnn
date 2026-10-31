"""Compara las etiquetas débiles con las 58 gold."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from config import LABELS, TRAIN_CSV

ES = {"ACL": "LCA", "MCL": "LCM", "Medial Meniscus": "Menisco medial", "Lateral Meniscus": "Menisco lateral",
      "Medial OA": "Artrosis medial", "Lateral OA": "Artrosis lateral", "PF OA": "Artrosis PF", "Effusion": "Derrame",
      "Synovitis": "Sinovitis", "Baker's": "Baker", "Contusion": "Contusión", "Fracture": "Fractura"}


def evaluate(path, thr=.5):
    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    gold = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS].astype(int)
    pred = pd.read_csv(path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    ids = gold.index.intersection(pred.dropna(subset=LABELS).index)       # estudios con predicción en las 12
    rows = []
    for l in LABELS:
        y, p = gold.loc[ids, l].to_numpy(), pred.loc[ids, l].astype(float).to_numpy()
        b = (p >= thr).astype(int)
        rows.append({"hallazgo": ES[l], "positivos gold": int(y.sum()), "AUC": roc_auc_score(y, p), "acuerdo": (b == y).mean(),
                     "sensibilidad": b[y == 1].mean(), "especificidad": 1 - b[y == 0].mean(),
                     "kappa": cohen_kappa_score(y, b)})
    t = pd.DataFrame(rows).set_index("hallazgo")
    t.loc["MEDIA (macro)"] = t.mean()
    t.loc["MEDIA (macro)", "positivos gold"] = gold.loc[ids].sum().sum()
    t["ruido (1 − acuerdo)"] = 1 - t["acuerdo"]
    return t, len(ids)


def main(rutas=None):
    pd.set_option("display.width", 200)
    summary = {}
    for path in rutas or sys.argv[1:]:
        t, n = evaluate(path)
        print(f"\n=== {path} · {n} estudios gold ===")
        print(t.round(3).to_string())
        summary[os.path.basename(path)] = t.loc["MEDIA (macro)", ["AUC", "acuerdo", "sensibilidad", "especificidad", "kappa"]]
    if len(summary) > 1:
        print("\n=== resumen (medias macro) ===")
        print(pd.DataFrame(summary).T.round(3).to_string())


if __name__ == "__main__":
    main()
