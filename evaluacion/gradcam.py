"""Mapas HiResCAM de los modelos del fold 0 en cuatro hallazgos."""
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import LABELS, RESULTADOS
from entrenamiento import comun as C
from entrenamiento.entrenar import DEBILES, run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HALLAZGOS = ["ACL", "Medial Meniscus", "Effusion", "Baker's"]
MODELOS = {
    "ConvNeXt-Tiny": dict(run=run("convnext", 0), crear=lambda: C.RedPreentrenada("convnext_tiny", preentrenada=False),
                          capa=lambda r: r.backbone[5]),
    "nuestra CNN": dict(run=run("alumno", 0), crear=lambda: C.RedZoom(), capa=lambda r: r.etapas[2]),
}
N_PLANO = 12
PLANOS = ["sagital", "coronal", "axial"]


def cargar_redes():
    redes = {}
    for n, m in MODELOS.items():
        red = m["crear"]()
        red.load_state_dict(torch.load(f"{m['run']}/modelo.pt", map_location="cpu"))
        redes[n] = red.to(C.dev).eval()
    return redes


def hirescam(red, capa, x, j):
    """x: (72, 3, 224, 224) de un estudio → mapas (72, 14, 14) ≥ 0 y la probabilidad del hallazgo j."""
    guardado = {}

    def gancho(_, __, salida):
        salida.retain_grad()
        guardado["A"] = salida

    h = capa(red).register_forward_hook(gancho)
    try:
        red.zero_grad()
        logit = red(x[None].to(C.dev))[0, j]
        logit.backward()
    finally:
        h.remove()
    A, G = guardado["A"].detach(), guardado["A"].grad
    return F.relu((G * A).sum(1)).cpu().numpy(), torch.sigmoid(logit).item()


def masa_por_ventana(cam):
    """Fracción del mapa en cada una de las 36 ventanas (vista completa + zoom de la misma ventana)."""
    m = cam.sum((1, 2))
    m = m[:36] + m[36:]
    return m / (m.sum() + 1e-12)


def figura(redes, ds, s, j, ruta, k=3):
    x = ds[s][0]
    fig, ax = plt.subplots(2, k * len(MODELOS), figsize=(3.1 * k * len(MODELOS), 6.4))
    for b, (nombre, red) in enumerate(redes.items()):
        cam, p = hirescam(red, MODELOS[nombre]["capa"], x, j)
        cam = cam / (cam.max() + 1e-12)
        masa = masa_por_ventana(cam)
        for c, i in enumerate(np.argsort(masa)[::-1][:k]):
            for fila, idx in enumerate([i, i + 36]):                 # completa, zoom
                a = ax[fila, b * k + c]
                a.imshow(x[idx, 1], cmap="gray")
                m = F.interpolate(torch.from_numpy(cam[idx])[None, None], size=x.shape[-2:], mode="bilinear")[0, 0].numpy()
                a.imshow(m, cmap="jet", alpha=np.where(m > 0.3, 0.45, 0.0), vmin=0, vmax=1)
                a.set_xticks([]), a.set_yticks([])
                if fila == 0:
                    plano, pos = divmod(i, N_PLANO)
                    a.set_title(f"{nombre} · p={p:.2f}\n{PLANOS[plano]} {100 * pos / (N_PLANO - 1):.0f} % · "
                                f"{100 * masa[i]:.0f} % del mapa", fontsize=9)
            ax[1, b * k + c].set_xlabel("zoom 100 mm", fontsize=8)
    fig.suptitle(f"{LABELS[j]} · estudio gold positivo según el radiólogo", fontsize=12)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def main():
    os.makedirs(RESULTADOS, exist_ok=True)
    datos = C.preparar_datos(fold_val=0, etiquetas=DEBILES, etiquetas_val=DEBILES, clase_dataset=C.RodillaDatasetZoom, batch=2)
    ds, Y = datos["dl_gold"].dataset, datos["y_gold"]
    P = {n: C.cargar_predicciones(m["run"], datos, m["crear"])["gold"] for n, m in MODELOS.items()}
    del datos
    redes = cargar_redes()
    filas = []
    for h in HALLAZGOS:
        j = LABELS.index(h)
        pos = np.where(Y[:, j] == 1)[0]
        ejemplo = pos[np.argmax(np.mean([P[n][pos, j] for n in P], 0))]       # el positivo más claro para ambos
        figura(redes, ds, ejemplo, j, f"{RESULTADOS}/gradcam_{h.lower().replace(' ', '_').replace(chr(39), '')}.png")
        for nombre, red in redes.items():
            planos = []
            for s in pos:
                m = hirescam(red, MODELOS[nombre]["capa"], ds[s][0], j)[0].sum((1, 2))
                m = m / (m.sum() + 1e-12)
                planos.append(m[:36].reshape(3, N_PLANO).sum(1) + m[36:].reshape(3, N_PLANO).sum(1))
            filas.append({"hallazgo": h, "modelo": nombre, "positivos": len(pos),
                          **{f"% {p}": 100 * v for p, v in zip(PLANOS, np.mean(planos, 0))}})
    t = pd.DataFrame(filas)
    t.to_csv(f"{RESULTADOS}/gradcam_planos.csv", index=False)
    print(t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
