"""Prepara los volúmenes por plano y completa la lateralidad."""
from imagenes import lateralidad, preparar


def main():
    print("\n## 1. Preparar imágenes")
    preparar.main()
    print("\n## 2. Lado de la rodilla")
    lateralidad.main()


if __name__ == "__main__":
    main()
