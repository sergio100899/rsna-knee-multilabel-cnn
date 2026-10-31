"""Etiquetas con EmbeddingGemma + regresión logística, combinadas con las reglas."""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import ETIQUETAS as OUT, LABELS, MODELO_EMBEDDINGS, MODELO_EMBEDDINGS_HF, TRAIN_CSV
from etiquetas.reglas import HIGH, PLAIN, clauses, score_clause

PREFIJO = "task: classification | query: "                   # prompt de EmbeddingGemma para clasificación


def embed(texts):
    """Vectores normalizados de EmbeddingGemma (con sus capas densas finales, vía sentence-transformers).
    Se guardan en caché: la segunda ejecución no vuelve a pasar por la GPU."""
    cache = f"{OUT}/clausulas_gemma.npy"
    if os.path.exists(cache):
        E = np.load(cache)
        if len(E) == len(texts):
            return E.astype(np.float32)
    if not os.path.isdir(MODELO_EMBEDDINGS):
        from huggingface_hub import snapshot_download
        print(f"descargando {MODELO_EMBEDDINGS_HF} en {MODELO_EMBEDDINGS} (1,2 GB)")
        snapshot_download(MODELO_EMBEDDINGS_HF, local_dir=MODELO_EMBEDDINGS)
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(MODELO_EMBEDDINGS, device=dev, model_kwargs={"dtype": torch.float32})
    E = m.encode([PREFIJO + t for t in texts], batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=True)
    np.save(cache, E.astype(np.float16))
    return E.astype(np.float16).astype(np.float32)                 # igual que al leer la caché: mismo resultado siempre


def main():
    tag = "_gemma"
    rng = np.random.default_rng(0)
    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    gold = set(train[train[LABELS].notna().all(axis=1)].StudyInstanceUID)
    rows = []
    for u, r in zip(train.StudyInstanceUID, train.Report.fillna("")):
        for c in clauses(r):
            rows.append({"StudyInstanceUID": u, "clause": c, **score_clause(c)})
    C = pd.DataFrame(rows)
    print(f"{len(C)} cláusulas de {C.StudyInstanceUID.nunique()} informes")
    E = embed(C.clause.tolist())
    silver = ~C.StudyInstanceUID.isin(gold).to_numpy()

    preds = pd.DataFrame({"StudyInstanceUID": C.StudyInstanceUID})
    for l in LABELS:
        s = C[l]
        pos = silver & s.isin([HIGH, PLAIN]).to_numpy()
        hard = silver & (s.notna() & ~s.isin([HIGH, PLAIN])).to_numpy()        # negadas o de bajo grado
        other = np.where(silver & s.isna().to_numpy())[0]
        easy = np.zeros(len(C), bool)                # muestra de 4 negativas fáciles por positiva, clases equilibradas
        easy[rng.choice(other, min(len(other), 4 * max(pos.sum(), 1)), replace=False)] = True
        idx = pos | hard | easy
        clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")
        clf.fit(E[idx], pos[idx].astype(int))
        preds[l] = clf.predict_proba(E)[:, 1]
        print(f"  {l:17s} positivas {int(pos.sum()):5d} · negativas difíciles {int(hard.sum()):5d} · otras {int(easy.sum()):5d}")

    emb = preds.groupby("StudyInstanceUID")[LABELS].max().reindex(train.StudyInstanceUID).fillna(0.0)
    emb.rename_axis("StudyInstanceUID").reset_index().to_csv(f"{OUT}/etiquetas_embeddings{tag}.csv", index=False)
    rules = pd.read_csv(f"{OUT}/etiquetas_reglas.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    hyb = np.sqrt(rules.loc[emb.index, LABELS] * emb[LABELS])
    hyb.rename_axis("StudyInstanceUID").reset_index().to_csv(f"{OUT}/etiquetas_hibrido{tag}.csv", index=False)
    print(f"→ {OUT}/etiquetas_embeddings{tag}.csv y {OUT}/etiquetas_hibrido{tag}.csv")


if __name__ == "__main__":
    main()
