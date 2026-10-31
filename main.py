"""Ejecuta todo el pipeline: etiquetas, imágenes, entrenamiento y evaluación."""
from entrenamiento import entrenar
from etiquetas import generar as etiquetas
from evaluacion import gradcam, metricas, sesgo
from imagenes import generar as imagenes


def main():
    print("\n# 1. Etiquetas")
    etiquetas.main()
    print("\n# 2. Imágenes")
    imagenes.main()
    print("\n# 3. Entrenamiento")
    entrenar.main(["todo"])
    print("\n# 4. Evaluación")
    metricas.main()
    sesgo.main()
    gradcam.main()


if __name__ == "__main__":
    main()
