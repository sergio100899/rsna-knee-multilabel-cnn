"""DICOM → un volumen por plano (150 mm, 320 px) por estudio, índice y folds."""
import glob
import os
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from config import IMAGENES, INDICE, LABELS, RSNA, SERIES, TRAIN_CSV

FOV_MM = 150                                                # lado del recorte central, en milímetros
SIZE = 320                                                  # píxeles de salida (~0,47 mm/píxel)
WORKERS = min(16, os.cpu_count())
PLANES = {"Sagittal": "sag", "Coronal": "cor", "Axial": "ax"}
TAGS = ["SeriesNumber", "MRAcquisitionType", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
        "ReceiveCoilName", "Laterality", "SeriesDescription"]

# ---------------- cabeceras y escáner ----------------

def cabecera(args):
    series, files = args
    d = pydicom.dcmread(files[0], stop_before_pixels=True)
    return {"SeriesInstanceUID": series, "n_slices": len(files),
            **{t: "" if d.get(t) is None else str(d.get(t)) for t in TAGS}}


FABRICANTES = {"siemens": "Siemens", "ge ": "GE", "gehc": "GE", "ge medical": "GE", "philips": "Philips",
               "toshiba": "Canon/Toshiba", "canon": "Canon/Toshiba", "hitachi": "Hitachi/Fujifilm",
               "fujifilm": "Hitachi/Fujifilm", "united imaging": "United Imaging", "uih": "United Imaging",
               "esaote": "Esaote"}


def fabricante(m):
    m = (str(m) + " ").lower()
    return next((v for k, v in FABRICANTES.items() if k in m), str(m).strip() or "?")


def moda(x):
    x = x.dropna()
    x = x[x.astype(str).str.strip() != ""]
    return x.mode().iloc[0] if len(x) else ""


def por_estudio(series):
    """Escáner, fabricante, campo y lado DICOM de cada estudio (la moda de sus series)."""
    s = series.copy()
    s["manufacturer"] = s.Manufacturer.map(fabricante)
    s["field_T"] = pd.to_numeric(s.MagneticFieldStrength, errors="coerce").round(1)
    s["scanner"] = (s.manufacturer + " | " + s.ManufacturerModelName.str.strip() + " | "
                    + s.field_T.map(lambda v: f"{v}T" if pd.notna(v) else "?T") + " | " + s.ReceiveCoilName.str.strip())
    return s.groupby("StudyInstanceUID").agg(scanner=("scanner", moda), manufacturer=("manufacturer", moda),
                                             field_T=("field_T", moda), lat_dicom=("Laterality", moda))


# ---------------- elección de series ----------------

def elegir_series(g):
    elegidas = {}
    for plano, k in PLANES.items():
        c = g[g.Anatomical_Plane == plano]
        if c.empty:
            continue
        c = c.assign(tipo=2 * (c.Fluid_Sensitive == "1") + (c.Fat_Suppression == "1"),
                     es3d=c.MRAcquisitionType.str.upper() == "3D",
                     dist30=(c.n_slices.astype(int) - 30).abs(),
                     num=pd.to_numeric(c.SeriesNumber, errors="coerce").fillna(0))
        elegidas[k] = c.sort_values(["tipo", "es3d", "dist30", "num", "SeriesInstanceUID"],
                                    ascending=[False, True, True, False, True]).iloc[0]
    return elegidas


# ---------------- lado de la rodilla desde el informe ----------------

def norm(s):
    s = unicodedata.normalize("NFKD", str(s).replace("İ", "i").replace("I", "i").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("µ", "μ").replace("ς", "σ").replace("∆", "δ")


DERECHA = ["right knee", "rt knee", "knee right", "knee, right", "rodilla derecha", "sag diz", "desnog koljena",
           "desno koljeno", "desnog kolena", "desno koleno", "desnom koljenu", "δεξιου γονατοσ", "δεξιο γονατο",
           "δεξιοσ γονατοσ", "knie rechts", "rechtes knie", "rechten knie", "rechter knie", "kniegelenk rechts",
           "rechtes kniegelenk", "rechten kniegelenk", "дясна колянна", "дясната колянна", "дясно коляно",
           "дясното коляно", "genou droit"]
IZQUIERDA = ["left knee", "lt knee", "knee left", "knee, left", "rodilla izquierda", "sol diz", "lijevog koljena",
             "lijevo koljeno", "levog kolena", "levo koleno", "lijevom koljenu", "αριστερου γονατοσ", "αριστερο γονατο",
             "αριστεροσ γονατοσ", "knie links", "linkes knie", "linken knie", "linker knie", "kniegelenk links",
             "linkes kniegelenk", "linken kniegelenk", "лява колянна", "лявата колянна", "ляво коляно",
             "лявото коляно", "genou gauche"]
RX_D = re.compile("|".join(map(re.escape, map(norm, DERECHA))))
RX_I = re.compile("|".join(map(re.escape, map(norm, IZQUIERDA))))


def lado_informe(report):
    r = norm(report)
    d, i = len(RX_D.findall(r)), len(RX_I.findall(r))
    return "R" if d > i else "L" if i > d else ""


# ---------------- lectura y preprocesado de una serie ----------------

def leer_serie(files):
    ds = [pydicom.dcmread(f) for f in files]
    normal = np.cross(*np.reshape(np.array(ds[0].ImageOrientationPatient, float), (2, 3)))
    pos = np.array([np.dot(np.array(d.ImagePositionPatient, float), normal) for d in ds])
    ds = [ds[i] for i in np.argsort(pos)]
    forma = max({d.pixel_array.shape for d in ds}, key=[d.pixel_array.shape for d in ds].count)
    ds = [d for d in ds if d.pixel_array.shape == forma]                     # descarta cortes de otro tamaño (raros)
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1) or 1)
                    + float(getattr(d, "RescaleIntercept", 0) or 0) for d in ds])
    if str(getattr(ds[0], "PhotometricInterpretation", "")) == "MONOCHROME1":
        vol = vol.max() - vol
    return vol, float(ds[0].PixelSpacing[0])


def preprocesar(vol, px, fov_mm=FOV_MM, size=SIZE):
    n = int(round(fov_mm / px))                                             # píxeles que ocupan 150 mm
    S, H, W = vol.shape
    out = np.zeros((S, n, n), np.float32)
    top, left = (H - n) // 2, (W - n) // 2
    sy, sx, dy, dx = max(0, top), max(0, left), max(0, -top), max(0, -left)
    h, w = min(n - dy, H - sy), min(n - dx, W - sx)
    out[:, dy:dy + h, dx:dx + w] = vol[:, sy:sy + h, sx:sx + w]
    lo, hi = np.percentile(out, [1, 99.5])
    out = np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)
    t = F.interpolate(torch.from_numpy(out)[:, None], size=(size, size), mode="bilinear", antialias=True)[:, 0]
    return (t.numpy() * 255).round().astype(np.uint8)


def procesar_estudio(args):
    uid, eleccion = args
    destino = f"{IMAGENES}/{uid}.npz"
    if os.path.exists(destino):
        return uid, "ya existía"
    torch.set_num_threads(1)
    out = {}
    for k, files in eleccion.items():
        try:
            vol, px = leer_serie(files)
            out[k] = preprocesar(vol, px)
        except Exception as e:                                              # una serie ilegible no tumba el estudio
            return uid, f"error en {k}: {e}"
    np.savez_compressed(destino + ".tmp.npz", **out)
    os.replace(destino + ".tmp.npz", destino)                               # escritura atómica
    return uid, "ok"


def main():
    os.makedirs(IMAGENES, exist_ok=True)
    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    series_csv = pd.read_csv(f"{RSNA}/train_series.csv", dtype=str)
    gold = set(train[train[LABELS].notna().all(axis=1)].StudyInstanceUID)
    dicoms = defaultdict(list)                              # SeriesInstanceUID → rutas de sus DICOM
    for f in sorted(glob.glob(f"{RSNA}/train_series/*/*/*.dcm")):
        dicoms[f.split("/")[-2]].append(f)
    series_csv = series_csv[series_csv.SeriesInstanceUID.isin(dicoms)]
    print(f"{len(train)} estudios ({len(gold)} gold) · {len(series_csv)} series · {sum(map(len, dicoms.values()))} DICOM")

    print("1/4 cabeceras DICOM…", flush=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        heads = pd.DataFrame(list(ex.map(cabecera, [(s, dicoms[s]) for s in series_csv.SeriesInstanceUID], chunksize=64)))
    series = series_csv.merge(heads, on="SeriesInstanceUID")
    series.to_csv(SERIES, index=False)
    estudio = por_estudio(series)
    print(f"   {estudio.scanner.nunique()} escáneres distintos")

    eleccion = {u: elegir_series(g) for u, g in series.groupby("StudyInstanceUID")}
    lado_dicom = estudio.lat_dicom.str.upper().map({"R": "R", "RIGHT": "R", "L": "L", "LEFT": "L"}).fillna("")
    inf = dict(zip(train.StudyInstanceUID, train.Report.fillna("").map(lado_informe)))
    lado = {u: (lado_dicom.get(u, "") or inf.get(u, "")) for u in train.StudyInstanceUID}
    fuente = {u: "dicom" if lado_dicom.get(u, "") else "informe" if inf.get(u, "") else "" for u in train.StudyInstanceUID}

    print("2/4 imágenes (la parte larga)…", flush=True)
    trabajos = [(u, {k: dicoms[r.SeriesInstanceUID] for k, r in el.items()}) for u, el in eleccion.items()]
    estados = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, (uid, estado) in enumerate(ex.map(procesar_estudio, trabajos, chunksize=4), 1):
            estados.append(estado.split(":")[0])
            if i % 250 == 0 or i == len(trabajos):
                print(f"   {i}/{len(trabajos)}", flush=True)
    print("  ", pd.Series(estados).value_counts().to_dict())

    print("3/4 folds por escáner…", flush=True)
    ok = sorted(f[:-4] for f in os.listdir(IMAGENES) if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    silver = [u for u in ok if u not in gold]
    fold = dict.fromkeys(ok, -1)
    if len(silver) >= 5:
        for k, (_, va) in enumerate(GroupKFold(5).split(silver, groups=estudio.loc[silver, "scanner"].to_numpy())):
            for i in va:
                fold[silver[i]] = k

    print("4/4 índice…", flush=True)
    filas = []
    for u in ok:
        with np.load(f"{IMAGENES}/{u}.npz") as d:
            n = {f"n_{k}": d[k].shape[0] if k in d.files else 0 for k in PLANES.values()}
        el = eleccion[u]
        filas.append({"StudyInstanceUID": u, "split": "gold" if u in gold else "silver", "fold": fold[u],
                      "scanner": estudio.scanner[u], "manufacturer": estudio.manufacturer[u],
                      "field_T": estudio.field_T[u], "lado": lado[u], "fuente_lado": fuente[u], **n,
                      "plano_sin_fs": ",".join(k for k, r in el.items() if r.Fluid_Sensitive != "1"),
                      **{f"serie_{k}": el[k].SeriesInstanceUID if k in el else "" for k in PLANES.values()}})
    indice = pd.DataFrame(filas)
    indice.to_csv(INDICE, index=False)
    print(f"→ {INDICE}: {len(indice)} estudios · {(indice.split == 'gold').sum()} gold · "
          f"lado conocido {(indice.lado != '').mean():.0%}")


if __name__ == "__main__":
    main()
