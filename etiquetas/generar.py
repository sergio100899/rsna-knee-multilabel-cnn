"""Genera gold, reglas e híbrido, y los evalúa contra gold."""
from config import ETIQUETAS
from etiquetas import embeddings, evaluar, gold, reglas


def main():
    print("\n## 0. Gold")
    gold.main()
    print("\n## 1. Reglas")
    reglas.main()
    print("\n## 2. Híbrido reglas + embeddings")
    embeddings.main()
    print("\n## 3. Validación contra gold")
    evaluar.main([f"{ETIQUETAS}/etiquetas_reglas.csv", f"{ETIQUETAS}/etiquetas_hibrido_gemma.csv"])


if __name__ == "__main__":
    main()
