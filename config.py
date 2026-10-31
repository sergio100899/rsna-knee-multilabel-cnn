"""Hallazgos y rutas compartidos por todo el repositorio."""

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

RSNA = "data/rsna"                                              # zip de Kaggle descomprimido
TRAIN_CSV = f"{RSNA}/train.csv"
ETIQUETAS = "data/etiquetas"                                    # etiquetas débiles generadas
MODELO_EMBEDDINGS = "data/modelos/embeddinggemma-300m"
MODELO_EMBEDDINGS_HF = "unsloth/embeddinggemma-300m"            # copia sin restricciones de google/embeddinggemma-300m
