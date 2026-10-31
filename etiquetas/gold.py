"""Extrae de train.csv los 58 estudios con etiqueta de radiólogo."""
import os

import pandas as pd

from config import ETIQUETAS, LABELS, TRAIN_CSV


def main():
    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    gold = train[train[LABELS].notna().all(axis=1)][["StudyInstanceUID", *LABELS]]
    gold[LABELS] = gold[LABELS].astype(int)
    os.makedirs(ETIQUETAS, exist_ok=True)
    gold.to_csv(f"{ETIQUETAS}/gold.csv", index=False)
    print(f"{len(gold)} estudios gold → {ETIQUETAS}/gold.csv")


if __name__ == "__main__":
    main()
