"""Hallazgos y rutas compartidos por todo el repositorio. Las rutas son relativas a la raíz: todo se ejecuta desde ahí."""
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

DATA = "data"
RSNA = f"{DATA}/rsna"                                           # zip de Kaggle descomprimido
TRAIN_CSV = f"{RSNA}/train.csv"

ETIQUETAS = f"{DATA}/etiquetas"                                 # etiquetas débiles generadas y gold.csv
MODELO_EMBEDDINGS = f"{DATA}/modelos/embeddinggemma-300m"
MODELO_EMBEDDINGS_HF = "unsloth/embeddinggemma-300m"            # copia sin restricciones de google/embeddinggemma-300m

IMAGENES = f"{DATA}/imagenes"                                   # un .npz por estudio: sag, cor, ax (cortes, 320, 320)
INDICE = f"{DATA}/indice.csv"                                   # un estudio por fila: gold/silver, fold, escáner, lado…
SERIES = f"{DATA}/series.csv"                                   # cabecera DICOM de cada serie
MODELO_LATERALIDAD = f"{DATA}/modelos/lateralidad_cnn.pt"

RUNS = f"{DATA}/runs"                                           # un directorio por modelo: checkpoint, modelo.pt, historial
RESULTADOS = f"{DATA}/resultados"                               # tablas y figuras de la evaluación
