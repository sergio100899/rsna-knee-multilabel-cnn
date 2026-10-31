"""CNN pequeña para el lado de la rodilla cuando no viene en el DICOM ni en el informe."""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from config import DATA, IMAGENES, INDICE, MODELO_LATERALIDAD

RES = 128                                                   # resolución de entrada
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def centrales(vol, k=3):
    c = len(vol) // 2
    idx = np.clip(np.arange(c - k // 2, c + k // 2 + 1), 0, len(vol) - 1)
    return vol[idx]


def entrada(uid):
    with np.load(f"{IMAGENES}/{uid}.npz") as d:
        x = np.concatenate([centrales(d[k]) for k in ("cor", "ax")])           # (6, 320, 320)
    t = F.interpolate(torch.from_numpy(x).float()[None], size=(RES, RES), mode="bilinear", antialias=True)[0]
    return t.round().clamp(0, 255).to(torch.uint8).numpy()


def bloque(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                         nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class CNNLado(nn.Module):
    def __init__(self, c_in=6):
        super().__init__()
        self.features = nn.Sequential(bloque(c_in, 32), bloque(32, 64), bloque(64, 128), bloque(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(x))[:, 0]


def aumentar(x, y):
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    y = torch.where(flip, 1 - y, y)
    dx, dy = np.random.randint(-8, 9, 2)
    x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(-2, -1))
    gain = 1 + 0.2 * (torch.rand(len(x), 1, 1, 1, device=x.device) - 0.5)
    return (x * gain).clamp(0, 1), y


@torch.no_grad()
def predecir(model, Xn, bs=512):
    model.eval()
    out = []
    for i in range(0, len(Xn), bs):
        x = torch.from_numpy(Xn[i:i + bs]).to(DEV).float() / 255
        p = (torch.sigmoid(model(x)) + 1 - torch.sigmoid(model(x.flip(-1)))) / 2      # TTA con el reflejo
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def entrenar(Xtr, ytr, epocas=25, bs=128, lr=2e-3):
    model = CNNLado().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pasos = epocas * int(np.ceil(len(Xtr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=pasos)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr).float()
    for _ in range(epocas):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            x, y = aumentar(Xt[b].to(DEV).float() / 255, yt[b].to(DEV))
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    indice = pd.read_csv(INDICE, dtype={"StudyInstanceUID": str}, keep_default_na=False)
    print("lado:", indice.fuente_lado.replace("", "desconocido").value_counts().to_dict())

    cache = f"{DATA}/lateralidad_entrada.npz"
    ids = list(indice.StudyInstanceUID)
    if os.path.exists(cache) and list(np.load(cache)["ids"]) == ids:
        X = np.load(cache)["X"]
    else:
        torch.set_num_threads(1)
        with ProcessPoolExecutor(min(16, os.cpu_count())) as ex:
            X = np.stack(list(ex.map(entrada, ids, chunksize=16)))
        np.savez(cache, X=X, ids=np.array(ids))
        torch.set_num_threads(os.cpu_count())

    conocido = indice.lado.isin(["R", "L"]).to_numpy()
    Xk, yk = X[conocido], (indice.lado[conocido] == "L").to_numpy().astype(np.float32)
    if len(yk) >= 50:                                        # validación cruzada (se salta con muestras de prueba pequeñas)
        oof = np.zeros(len(Xk))
        for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xk, yk)):
            oof[va] = predecir(entrenar(Xk[tr], yk[tr]), Xk[va])
            print(f"   fold {k}: acierto {((oof[va] > .5) == yk[va]).mean():.4f}")
        print(f"acierto (validación cruzada, {len(yk)} estudios): {((oof > .5) == yk).mean():.4f}")

    final = entrenar(Xk, yk)
    os.makedirs(os.path.dirname(MODELO_LATERALIDAD), exist_ok=True)
    torch.save(final.state_dict(), MODELO_LATERALIDAD)
    p_izq = predecir(final, X)
    indice["p_izquierda_cnn"] = p_izq.round(4)
    indice["lado_final"] = np.where(conocido, indice.lado, np.where(p_izq > .5, "L", "R"))   # el dato DICOM/informe manda
    indice["fuente_lado_final"] = np.where(conocido, indice.fuente_lado, "cnn")
    indice.to_csv(INDICE, index=False)
    print(f"lado completado por la CNN: {(~conocido).sum()} estudios → {INDICE} · modelo → {MODELO_LATERALIDAD}")


if __name__ == "__main__":
    main()
