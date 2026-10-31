"""Dataset 2.5D, las dos redes, pérdida focal y bucle de entrenamiento."""
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import ETIQUETAS, IMAGENES, INDICE, LABELS

PLANOS = ["sag", "cor", "ax"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- datos ----------------

def cargar_npz(path, intentos=3):
    """Lee el .npz completo; reintenta (en esta máquina hubo lecturas corruptas puntuales de archivos sanos)."""
    for t in range(intentos):
        try:
            with np.load(path) as d:
                return {k: d[k] for k in d.files}
        except Exception:
            if t == intentos - 1:
                raise


class RodillaDataset(Dataset):
    """Un estudio → x (36, 3, size, size), plano (36,), pos (36,), y (12,)."""

    def __init__(self, ids, etiquetas, lados, entrenar, n_cortes=12, size=224):
        self.ids, self.etiquetas, self.lados, self.entrenar = ids, etiquetas, lados, entrenar
        self.n_cortes, self.size = n_cortes, size

    def __len__(self):
        return len(self.ids)

    def indices(self, n):
        """n_cortes índices repartidos en la serie; al entrenar, todos desplazados al azar hasta medio paso."""
        base = np.linspace(0, n - 1, self.n_cortes)
        if self.entrenar:
            paso = (n - 1) / (self.n_cortes - 1)
            base = base + np.random.uniform(-paso / 2, paso / 2)
        return np.clip(base.round().astype(int), 0, n - 1)

    def __getitem__(self, i):
        uid = self.ids[i]
        v = cargar_npz(f"{IMAGENES}/{uid}.npz")
        if self.lados[uid] == "L":                                     # todas las rodillas como derechas
            v = {"sag": v["sag"][::-1], "cor": v["cor"][..., ::-1], "ax": v["ax"][..., ::-1]}
        x, plano, pos = [], [], []
        for k, nombre in enumerate(PLANOS):
            vol = v[nombre]
            n = len(vol)
            idx = self.indices(n)
            ventana = np.stack([vol[np.clip(idx - 1, 0, n - 1)], vol[idx], vol[np.clip(idx + 1, 0, n - 1)]], 1)
            x.append(ventana)
            plano += [k] * len(idx)
            pos.append(idx / (n - 1))
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate(x))).float().div_(255)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
        y = torch.tensor(self.etiquetas.loc[uid].to_numpy(), dtype=torch.float32)
        return x, torch.tensor(plano), torch.tensor(np.concatenate(pos), dtype=torch.float32), y


class RodillaDatasetZoom(RodillaDataset):
    """Un estudio → x (72, 3, size, size): 36 ventanas completas + las mismas 36 con zoom central.
    `aumentar_plano(x)`: gancho de aumentación por plano, solo al entrenar; no hace nada (la aumentación no mejoró).
    `size_completa` (por defecto = size): la vista completa a otra resolución, p. ej. 320 = la nativa del dataset. Como las
    dos vistas van en un mismo tensor, el zoom (size × size) queda en la esquina superior izquierda de un lienzo de
    size_completa × size_completa con relleno de ceros, que RedZoom(size_zoom=size) descarta."""

    def __init__(self, *args, zoom_mm=100, size_completa=None, **kw):
        super().__init__(*args, **kw)
        self.zoom_px = round(320 * zoom_mm / 150)                     # el dataset guarda 150 mm en 320 px
        self.size_completa = size_completa or self.size

    def aumentar_plano(self, x):
        return x

    def __getitem__(self, i):
        uid = self.ids[i]
        v = cargar_npz(f"{IMAGENES}/{uid}.npz")
        if self.lados[uid] == "L":
            v = {"sag": v["sag"][::-1], "cor": v["cor"][..., ::-1], "ax": v["ax"][..., ::-1]}
        x, plano, pos = [], [], []
        for k, nombre in enumerate(PLANOS):
            vol = v[nombre]
            n = len(vol)
            idx = self.indices(n)
            ventana = np.stack([vol[np.clip(idx - 1, 0, n - 1)], vol[idx], vol[np.clip(idx + 1, 0, n - 1)]], 1)
            ventana = torch.from_numpy(np.ascontiguousarray(ventana)).float().div_(255)
            x.append(self.aumentar_plano(ventana) if self.entrenar else ventana)
            plano += [k] * len(idx)
            pos.append(idx / (n - 1))
        x = torch.cat(x)                                              # (36, 3, 320, 320)
        a = (x.shape[-1] - self.zoom_px) // 2
        s = self.size_completa
        completa = x if s == x.shape[-1] else F.interpolate(x, size=(s, s), mode="bilinear", antialias=True)
        zoom = F.interpolate(x[..., a:a + self.zoom_px, a:a + self.zoom_px], size=(self.size, self.size),
                             mode="bilinear", antialias=True)
        if s != self.size:
            zoom = F.pad(zoom, (0, s - self.size, 0, s - self.size))
        y = torch.tensor(self.etiquetas.loc[uid].to_numpy(), dtype=torch.float32)
        return (torch.cat([completa, zoom]), torch.tensor(plano * 2),
                torch.tensor(np.concatenate(pos * 2), dtype=torch.float32), y)


def preparar_datos(etiquetas="etiquetas_hibrido_gemma", fold_val=0, n_cortes=12, size=224, batch=8, workers=8,
                   etiquetas_val="etiquetas_hibrido_gemma", clase_dataset=None, folds_fuera=(), **kw_dataset):
    """Índice, etiquetas, partición (folds ≠ fold_val para entrenar, fold_val para validar, gold aparte) y DataLoaders.
    `etiquetas`: con qué etiquetas débiles se entrena. `etiquetas_val`: la referencia de la validación, fija para que
    el AUC de validación sea comparable aunque se entrene con otra fuente de etiquetas.
    `clase_dataset` (por defecto RodillaDataset) y `kw_dataset`: para probar otra forma de armar la entrada (p. ej. zoom).
    `folds_fuera`: folds que tampoco se usan para entrenar."""
    indice = pd.read_csv(INDICE, dtype={"StudyInstanceUID": str}, keep_default_na=False).set_index("StudyInstanceUID")
    debiles = pd.read_csv(f"{ETIQUETAS}/{etiquetas}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS]
    referencia = pd.read_csv(f"{ETIQUETAS}/{etiquetas_val}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS]
    gold = pd.read_csv(f"{ETIQUETAS}/gold.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS]
    silver = indice[indice.split == "silver"]
    train_ids = silver.index[(silver.fold != fold_val) & ~silver.fold.isin(folds_fuera)].tolist()
    val_ids = silver.index[silver.fold == fold_val].tolist()
    gold_ids = gold.index.tolist()
    assert not set(gold_ids) & set(train_ids + val_ids)          # los gold nunca entran al entrenamiento
    lados = indice.lado_final.to_dict()
    kw = dict(lados=lados, n_cortes=n_cortes, size=size, **kw_dataset)
    clase = clase_dataset or RodillaDataset
    ds_train = clase(train_ids, debiles, entrenar=True, **kw)
    ds_val = clase(val_ids, referencia, entrenar=False, **kw)
    ds_gold = clase(gold_ids, gold.astype(float), entrenar=False, **kw)
    y_val_blandas = referencia.loc[val_ids].to_numpy()
    return {
        "indice": indice, "debiles": debiles, "gold": gold,
        "train_ids": train_ids, "val_ids": val_ids, "gold_ids": gold_ids,
        "dl_train": DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                               pin_memory=True, persistent_workers=True),
        "dl_val": DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=4),
        "dl_gold": DataLoader(ds_gold, batch_size=batch, shuffle=False, num_workers=4),
        "y_val_blandas": y_val_blandas,                              # para la pérdida de validación
        "y_val": (y_val_blandas >= 0.5).astype(int),                 # débiles binarizadas, para el AUC
        "y_gold": gold.loc[gold_ids].to_numpy().astype(int),
    }


# ---------------- piezas de la red ----------------

def conv_bn_relu(c_in, c_out, stride=1):
    """Conv 3×3 → BatchNorm → ReLU: la unidad básica que se repite en toda la red."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Stem(nn.Module):
    """3 conv 3×3 (la primera con stride 2) + max-pool: 3 × 224 × 224 → 32 × 56 × 56."""

    def __init__(self, c_in=3, c_out=32):
        super().__init__()
        self.conv1 = conv_bn_relu(c_in, c_out, stride=2)
        self.conv2 = conv_bn_relu(c_out, c_out)
        self.conv3 = conv_bn_relu(c_out, c_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.pool(self.conv3(self.conv2(self.conv1(x))))


class Etapa(nn.Module):
    """Etapa v0: dos conv 3×3 seguidas, sin atajo. La primera puede reducir el tamaño (stride 2) y cambiar los canales."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = conv_bn_relu(c_in, c_out, stride=stride)
        self.conv2 = conv_bn_relu(c_out, c_out)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.conv2(self.conv1(x))




class PoolingGlobal(nn.Module):
    """(N, C, h, w) → (N, C) con modo "media" o "maximo", o (N, 2C) con "media+maximo"."""

    def __init__(self, modo="media+maximo"):
        super().__init__()
        assert modo in ("media", "maximo", "media+maximo")
        self.modo = modo

    def forward(self, x):
        media, maximo = x.mean(dim=(2, 3)), x.amax(dim=(2, 3))
        if self.modo == "media":
            return media
        if self.modo == "maximo":
            return maximo
        return torch.cat([media, maximo], dim=1)

    def dim_salida(self, canales):
        return 2 * canales if self.modo == "media+maximo" else canales


class CabezaEstudio(nn.Module):
    """(B, n_imagenes, D) → (B, 12): junta las imágenes de cada estudio (media y/o máximo) y clasifica."""

    def __init__(self, dim, n_salidas=12, modo="media+maximo", dropout=0.3):
        super().__init__()
        assert modo in ("media", "maximo", "media+maximo")
        self.modo = modo
        self.dropout = nn.Dropout(dropout)
        self.clasificador = nn.Linear(2 * dim if modo == "media+maximo" else dim, n_salidas)

    def forward(self, v):
        media, maximo = v.mean(dim=1), v.amax(dim=1)
        e = media if self.modo == "media" else maximo if self.modo == "maximo" else torch.cat([media, maximo], dim=1)
        return self.clasificador(self.dropout(e))


class RedCNN(nn.Module):
    """Stem + 4 etapas hechas con `bloque(c_in, c_out, stride)` + pooling global + cabeza del estudio.
    `bloques_por_etapa`: profundidad. En cada etapa solo el primer bloque reduce el tamaño (stride) y cambia los canales.
    Con bloque=Etapa y 1 bloque por etapa es la versión base."""

    def __init__(self, bloque, ancho=32, bloques_por_etapa=1, pooling="media+maximo", agregacion="media+maximo", dropout=0.3):
        super().__init__()
        c = [ancho, ancho, 2 * ancho, 4 * ancho, 8 * ancho]                  # 32, 32, 64, 128, 256
        self.stem = Stem(3, c[0])
        strides = [1, 2, 2, 2]
        if bloques_por_etapa == 1:                                          # mismos nombres que las versiones anteriores
            self.etapas = nn.Sequential(*[bloque(c[k], c[k + 1], stride=strides[k]) for k in range(4)])
        else:
            self.etapas = nn.Sequential(*[
                nn.Sequential(bloque(c[k], c[k + 1], stride=strides[k]),
                              *[bloque(c[k + 1], c[k + 1], stride=1) for _ in range(bloques_por_etapa - 1)])
                for k in range(4)])
        self.pool = PoolingGlobal(pooling)
        self.cabeza = CabezaEstudio(self.pool.dim_salida(c[4]), modo=agregacion, dropout=dropout)

    def vector_por_imagen(self, imgs):
        return self.pool(self.etapas(self.stem(imgs)))

    def forward(self, x):                                                   # x: (B, 36, 3, H, W)
        B, n = x.shape[:2]
        return self.cabeza(self.vector_por_imagen(x.flatten(0, 1)).view(B, n, -1))


class RedPiramide(RedCNN):
    """v2: v0 + pirámide. Pooling global de las salidas de las etapas `niveles`,
    cada una con su LayerNorm, concatenadas. Con niveles=(4,) es la v0 (más una LayerNorm)."""

    def __init__(self, bloque=Etapa, niveles=(2, 3, 4), ancho=32, **kw):
        super().__init__(bloque, ancho=ancho, **kw)
        self.niveles = tuple(niveles)
        canales = {1: ancho, 2: 2 * ancho, 3: 4 * ancho, 4: 8 * ancho}
        dims = [self.pool.dim_salida(canales[n]) for n in self.niveles]
        self.normas = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.cabeza = CabezaEstudio(sum(dims), modo=self.cabeza.modo, dropout=self.cabeza.dropout.p)

    def vector_por_imagen(self, imgs):
        h, partes = self.stem(imgs), []
        for k, etapa in enumerate(self.etapas, start=1):
            h = etapa(h)
            if k in self.niveles:
                partes.append(self.normas[len(partes)](self.pool(h)))
        return torch.cat(partes, dim=1)


class RedZoom(RedPiramide):
    """v2 pirámide con dos vistas por ventana (completa y zoom), procesadas por la misma CNN.
    Entrada (B, 72, 3, H, W): 36 ventanas completas y después las mismas 36 con zoom; la cabeza recibe [v_completa, v_zoom].
    `size_zoom`: si las vistas tienen distinta resolución (RodillaDatasetZoom con size_completa), el zoom se recorta a
    size_zoom × size_zoom y cada vista pasa por la CNN por separado. Los pesos no cambian."""

    def __init__(self, size_zoom=None, **kw):
        super().__init__(**kw)
        self.size_zoom = size_zoom
        dim = self.cabeza.clasificador.in_features // 2
        self.cabeza = CabezaEstudio(2 * dim, modo=self.cabeza.modo, dropout=self.cabeza.dropout.p)

    def vectores(self, x):                                            # (B, 72, 3, H, W) → (B, 36, 2·D)
        B, n = x.shape[:2]
        if self.size_zoom is None or self.size_zoom == x.shape[-1]:
            v = self.vector_por_imagen(x.flatten(0, 1)).view(B, n, -1)
            return torch.cat([v[:, :n // 2], v[:, n // 2:]], dim=2)
        s = self.size_zoom
        completa = self.vector_por_imagen(x[:, :n // 2].flatten(0, 1)).view(B, n // 2, -1)
        zoom = self.vector_por_imagen(x[:, n // 2:, :, :s, :s].reshape(-1, *x.shape[2:3], s, s)).view(B, n // 2, -1)
        return torch.cat([completa, zoom], dim=2)

    def forward(self, x):
        return self.cabeza(self.vectores(x))











# ---- arquitectura preentrenada en ImageNet: transfer learning ----

class RedPreentrenada(nn.Module):
    """Backbone CNN de torchvision preentrenado en ImageNet, sin su clasificador, con las mismas piezas que RedZoom:
    pooling media+máximo con LayerNorm por imagen, vista completa + zoom concatenadas y CabezaEstudio (12 salidas sigmoides).
    Entrada (B, 72, 3, 224, 224) en [0, 1]: los 3 canales RGB son 3 cortes vecinos (2.5D) y se normalizan con la media y
    la desviación de ImageNet. `self.backbone` puede llevar su propio lr: entrenar(..., lr_backbone=...).
    `preentrenada=False` crea la misma red sin descargar pesos (para cargar un modelo ya entrenado)."""

    BACKBONES = {"resnet50": ("resnet50", "ResNet50_Weights", 2048),
                 "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", 1280),
                 "densenet121": ("densenet121", "DenseNet121_Weights", 1024),
                 "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", 768)}

    def __init__(self, nombre="resnet50", preentrenada=True, dropout=0.3):
        super().__init__()
        import torchvision
        fn, pesos, canales = self.BACKBONES[nombre]
        m = getattr(torchvision.models, fn)(weights=getattr(torchvision.models, pesos).DEFAULT if preentrenada else None)
        self.nombre = nombre
        if nombre.startswith("resnet"):
            self.backbone = nn.Sequential(*list(m.children())[:-2])          # sin avgpool ni fc
        elif nombre.startswith("densenet"):
            self.backbone = nn.Sequential(m.features, nn.ReLU())             # torchvision aplica esta ReLU en forward
        else:
            self.backbone = m.features                                       # efficientnet, convnext
        self.pool = PoolingGlobal("media+maximo")
        self.norma = nn.LayerNorm(self.pool.dim_salida(canales))
        self.cabeza = CabezaEstudio(2 * self.pool.dim_salida(canales), dropout=dropout)       # [v_completa, v_zoom]
        self.register_buffer("media", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("desv", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def vector_por_imagen(self, imgs):
        return self.norma(self.pool(self.backbone((imgs - self.media) / self.desv)))

    def vectores(self, x):                                            # (B, 72, 3, H, W) → (B, 36, 2·D)
        B, n = x.shape[:2]
        v = self.vector_por_imagen(x.flatten(0, 1)).view(B, n, -1)
        return torch.cat([v[:, :n // 2], v[:, n // 2:]], dim=2)

    def forward(self, x):
        return self.cabeza(self.vectores(x))


# ---------------- entrenamiento y evaluación ----------------

@torch.no_grad()
def predecir(modelo, loader):
    """Probabilidades (n_estudios, 12) de un DataLoader, con el modelo en modo evaluación."""
    modelo.eval()
    salida = []
    for x, _, _, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            salida.append(torch.sigmoid(modelo(x.to(dev, non_blocking=True)).float()).cpu())
    return torch.cat(salida).numpy()


def auc_macro(y, p):
    """Media de las AUC de los hallazgos que tienen positivos y negativos (la métrica de la competición)."""
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]))


def perdida_bce(p, y):
    """BCE media entre probabilidades predichas y etiquetas (0–1), la misma que optimiza el entrenamiento."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


class PerdidaFocal(nn.Module):
    """Focal binaria por hallazgo, con etiquetas blandas: (1 − p_t)^γ · BCE, promediada. γ = 0 es la BCE."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        p_t = y * p + (1 - y) * (1 - p)                       # probabilidad de la respuesta correcta
        perdida = (1 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            perdida = (y * self.alpha + (1 - y) * (1 - self.alpha)) * perdida
        return perdida.mean()


def actualizar_ema(ema, red, decaimiento):
    """ema ← decaimiento · ema + (1 − decaimiento) · red (pesos); los buffers de BatchNorm se copian tal cual."""
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), red.parameters()):
            pe.lerp_(p, 1 - decaimiento)
        for be, b in zip(ema.buffers(), red.buffers()):
            be.copy_(b)


def entrenar(red, datos, run, epocas=30, lr=1e-3, wd=1e-4, perdida_fn=None, semilla=0, ema=None, lr_backbone=None, acumular=1):
    """AdamW + OneCycle + BCE, validación (fold y gold) en cada época, checkpoint reanudable en `run`.
    Si el entrenamiento ya terminó, solo carga el modelo y devuelve el historial.
    `ema` (p. ej. 0.999): además guarda una media móvil exponencial de los pesos. No cambia el entrenamiento; se evalúan
    los dos (columnas "sin EMA" = pesos normales) y `modelo.pt` es la EMA (`modelo_sin_ema.pt`, los normales).
    `lr_backbone`: lr máximo distinto para `red.backbone` (redes preentrenadas); el resto usa `lr`.
    `acumular`: suma los gradientes de tantos lotes antes de cada paso del optimizador (lote efectivo = batch · acumular),
    para redes grandes que no entran en memoria con el lote completo."""
    torch.manual_seed(semilla)
    os.makedirs(run, exist_ok=True)
    red = red.to(dev)
    dl_train = datos["dl_train"]
    if lr_backbone is None:
        grupos, max_lr = red.parameters(), lr
    else:
        ids_backbone = {id(p) for p in red.backbone.parameters()}
        grupos = [{"params": list(red.backbone.parameters()), "lr": lr_backbone},
                  {"params": [p for p in red.parameters() if id(p) not in ids_backbone], "lr": lr}]
        max_lr = [lr_backbone, lr]
    opt = torch.optim.AdamW(grupos, lr=lr, weight_decay=wd)
    pasos = len(dl_train) // acumular
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=epocas * pasos, pct_start=0.1)
    perdida_fn = perdida_fn or nn.BCEWithLogitsLoss()
    red_ema = copy.deepcopy(red).eval() if ema else None

    historial, inicio, ckpt = [], 0, f"{run}/checkpoint.pt"
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=dev, weights_only=False)
        red.load_state_dict(c["modelo"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        if ema:
            red_ema.load_state_dict(c["ema"])
        historial, inicio = c["historial"], c["epoca"] + 1
        print(f"checkpoint encontrado: {'ya terminado' if inicio >= epocas else f'se retoma en la época {inicio + 1}'}")

    for ep in range(inicio, epocas):
        red.train()
        t, perdidas = time.time(), []
        barra = tqdm(dl_train, desc=f"época {ep + 1}/{epocas}", unit="lote", leave=False)
        opt.zero_grad(set_to_none=True)
        for i, (x, _, _, y) in enumerate(barra):
            if i >= pasos * acumular:                                # lotes sobrantes que no completan un paso
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = red(x)
            loss = perdida_fn(logits.float(), y)
            (loss / acumular).backward()
            if (i + 1) % acumular == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema:                                              # al principio decae más rápido (calentamiento)
                    paso = sched.last_epoch
                    actualizar_ema(red_ema, red, min(ema, (1 + paso) / (10 + paso)))
            perdidas.append(loss.item())
            barra.set_postfix(perdida_train=f"{np.mean(perdidas[-50:]):.4f}")
        evaluada = red_ema if ema else red
        p_val, p_gold = predecir(evaluada, datos["dl_val"]), predecir(evaluada, datos["dl_gold"])
        historial.append({"época": ep + 1, "pérdida": np.mean(perdidas), "pérdida val": perdida_bce(p_val, datos["y_val_blandas"]),
                          "lr": sched.get_last_lr()[0], "AUC val (fold 0)": auc_macro(datos["y_val"], p_val),
                          "AUC gold": auc_macro(datos["y_gold"], p_gold), "minutos": (time.time() - t) / 60})
        if ema:
            q_val, q_gold = predecir(red, datos["dl_val"]), predecir(red, datos["dl_gold"])
            historial[-1].update({"AUC val sin EMA": auc_macro(datos["y_val"], q_val),
                                  "AUC gold sin EMA": auc_macro(datos["y_gold"], q_gold)})
        estado = {"modelo": red.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "historial": historial, "epoca": ep}
        if ema:
            estado["ema"] = red_ema.state_dict()
        torch.save(estado, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
        h = historial[-1]
        tqdm.write(f"época {h['época']:2d}/{epocas} · pérdida train {h['pérdida']:.4f} · pérdida val {h['pérdida val']:.4f} · "
                   f"AUC val {h['AUC val (fold 0)']:.3f} · AUC gold {h['AUC gold']:.3f} · {h['minutos']:.1f} min"
                   + (f" · sin EMA: val {h['AUC val sin EMA']:.3f}, gold {h['AUC gold sin EMA']:.3f}" if ema else ""))
        if ep + 1 == int(os.environ.get("PARAR_EN", 0)):  # pausa: el checkpoint queda y se retoma relanzando sin PARAR_EN
            print(f"pausado en la época {ep + 1} (PARAR_EN)", flush=True)
            return pd.DataFrame(historial)

    pd.DataFrame(historial).to_csv(f"{run}/historial.csv", index=False)
    if ema:
        torch.save(red.state_dict(), f"{run}/modelo_sin_ema.pt")
    torch.save((red_ema if ema else red).state_dict(), f"{run}/modelo.pt")
    return pd.DataFrame(historial)








def cargar_predicciones(run, datos, red_fn):
    """Probabilidades del modelo de `run` en validación (su fold) y en gold; se calculan una vez y se guardan en el run."""
    try:
        with np.load(f"{run}/predicciones.npz") as d:
            return {"val": d["val"], "gold": d["gold"]}
    except FileNotFoundError:
        red = red_fn()
        red.load_state_dict(torch.load(f"{run}/modelo.pt", map_location="cpu"))
        red.to(dev)
        p = {"val": predecir(red, datos["dl_val"]), "gold": predecir(red, datos["dl_gold"])}
        np.savez(f"{run}/predicciones.npz", **p)
        return p
