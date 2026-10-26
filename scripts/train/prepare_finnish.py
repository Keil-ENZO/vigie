"""Prépare le jeu « navires de la côte finlandaise » au format YOLO.

Annotations : Mäyrä et Jokinen, Institut finlandais de l'environnement (SYKE),
Zenodo 10.5281/zenodo.10046342, CC BY 4.0. Boîtes alignées sur la grille UTM
des tuiles Sentinel-2, une couche par date.

Imagerie : la bande `visual` (TCI) L2A des mêmes passes, lue sur le STAC
d'Element 84, c'est-à-dire exactement ce que voit le pipeline Vigie. Les
annotations ont été faites sur du L1C, le décalage radiométrique est assumé.

Split géographique, celui des auteurs pour le test :
  - train : 34VEM (archipel de Turku), 35VLG (golfe de Finlande)
  - val   : 34WFT (baie de Botnie)
  - test  : 34VEN (mer de Botnie), 34VER (Kvarken)

Usage : uv run python scripts/train/prepare_finnish.py --out data
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from shapely import wkb

STAC = "https://earth-search.aws.element84.com/v1"
ZENODO = "https://zenodo.org/api/records/10046342/files/{name}.gpkg/content"
SPLITS = {"34VEM": "train", "35VLG": "train", "34WFT": "val", "34VEN": "test", "34VER": "test"}
# 34VEN recouvre 34VEM sur ~9,5 km au sud ; les auteurs n'y ont rien annoté.
# On exclut toute tuile qui descend sous cette ordonnée UTM.
VEN_MIN_NORTHING = 6_700_020.0
TILE = 640
STRIDE = {"train": TILE - 96, "val": TILE, "test": TILE}
MIN_WATER = 0.02  # part d'eau SCL minimale pour garder une tuile
MAX_NODATA = 0.5
KEEP_EMPTY_TRAIN = 0.3  # part des tuiles vides gardées en train
MIN_KEEP = 0.4  # part de la boîte conservée après découpe à la tuile
PAD_M = 500.0  # marge autour de l'enveloppe des annotations


def gpkg_geom(blob: bytes):
    """Décode une géométrie GeoPackage : en-tête GP puis WKB."""
    flags = blob[3]
    env = (flags >> 1) & 0b111
    size = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
    return wkb.loads(blob[8 + size:])


def read_layers(path: Path) -> dict[str, dict]:
    con = sqlite3.connect(path)
    out = {}
    for (name, srs) in con.execute(
        "select table_name, srs_id from gpkg_geometry_columns order by table_name"
    ).fetchall():
        boxes = [gpkg_geom(g).bounds for (g,) in con.execute(f'select geom from "{name}"')]
        out[name] = {"srs": srs, "boxes": np.array(boxes, dtype=np.float64).reshape(-1, 4)}
    con.close()
    return out


def find_item(tile: str, day: str) -> dict:
    z, band, sq = int(tile[:2]), tile[2], tile[3:]
    d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    body = {
        "collections": ["sentinel-2-l2a"], "limit": 10,
        "datetime": f"{d}T00:00:00Z/{d}T23:59:59Z",
        "query": {"mgrs:utm_zone": {"eq": z}, "mgrs:latitude_band": {"eq": band},
                  "mgrs:grid_square": {"eq": sq}},
    }
    feats = httpx.post(f"{STAC}/search", json=body, timeout=60).json()["features"]
    if not feats:
        raise RuntimeError(f"aucune passe L2A pour {tile} {d}")
    # Plusieurs retraitements possibles : on prend la ligne de base la plus récente.
    return max(feats, key=lambda f: f["properties"].get("s2:processing_baseline", ""))


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def clip_boxes(boxes: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Boîtes pixel de l'image entière vers la tuile, au format YOLO normalisé."""
    if len(boxes) == 0:
        return np.zeros((0, 4))
    b = boxes - [x0, y0, x0, y0]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, w)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, h)
    carea = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0)
    keep = (carea > 0) & (carea >= MIN_KEEP * area)
    c = c[keep]
    xc, yc = (c[:, 0] + c[:, 2]) / 2 / w, (c[:, 1] + c[:, 3]) / 2 / h
    bw, bh = (c[:, 2] - c[:, 0]) / w, (c[:, 3] - c[:, 1]) / h
    return np.stack([xc, yc, bw, bh], axis=1)


def erase_small(crop: np.ndarray, boxes: np.ndarray, x0: int, y0: int, min_diag: float) -> tuple[np.ndarray, np.ndarray]:
    """Efface par inpainting les boîtes de diagonale < min_diag pixels et les retire
    des étiquettes. YOLO n'a pas de zone « ignorée » : sans effacement, un petit
    bateau non étiqueté deviendrait un exemple négatif contradictoire."""
    diag = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    small = boxes[diag < min_diag]
    mask = np.zeros(crop.shape[:2], np.uint8)
    for x1, y1, x2, y2 in small - [x0, y0, x0, y0]:
        if x2 < 0 or y2 < 0 or x1 >= TILE or y1 >= TILE:
            continue
        mask[max(0, int(y1) - 1):int(np.ceil(y2)) + 1, max(0, int(x1) - 1):int(np.ceil(x2)) + 1] = 255
    if mask.any():
        crop = cv2.inpaint(np.ascontiguousarray(crop), mask, 3, cv2.INPAINT_TELEA)
    return crop, boxes[diag >= min_diag]


def process_scene(tile: str, day: str, layer: dict, envelope: tuple, root: Path, rng: random.Random,
                  variant: str = "", keep_empty: float = KEEP_EMPTY_TRAIN, min_diag: float = 0.0) -> dict | None:
    split = SPLITS[tile]
    if variant and split != "train":
        return None  # une variante ne refait que le train, val et test restent ceux du jeu de base
    item = find_item(tile, day)
    cache = root / "raw" / "s2"
    cache.mkdir(parents=True, exist_ok=True)
    tci = download(item["assets"]["visual"]["href"], cache / f"{item['id']}_TCI.tif")
    scl = download(item["assets"]["scl"]["href"], cache / f"{item['id']}_SCL.tif")
    with rasterio.open(tci) as src:
        if src.crs.to_epsg() != layer["srs"]:
            raise RuntimeError(f"{tile} {day} : CRS {src.crs} différent de EPSG:{layer['srs']}")
        rgb = np.moveaxis(src.read(), 0, -1)
        inv = ~src.transform
        tr = src.transform
        h, w = src.height, src.width
    with rasterio.open(scl) as s:
        water = s.read(1, out_shape=(h, w), resampling=Resampling.nearest) == 6
    bx = layer["boxes"]
    px1, py1 = inv * (bx[:, 0], bx[:, 3])  # coin haut gauche : x min, y max
    px2, py2 = inv * (bx[:, 2], bx[:, 1])
    boxes = np.stack([px1, py1, px2, py2], axis=1)
    ex1, ey1, ex2, ey2 = envelope
    ds_name = f"finnish_{variant}" if variant else "finnish"
    out_img = root / "datasets" / ds_name / "images" / split
    out_lbl = root / "datasets" / ds_name / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    stride = STRIDE[split]
    stats = {"tile": tile, "date": day, "split": split, "item": item["id"],
             "baseline": item["properties"].get("s2:processing_baseline"),
             "annotations": len(bx), "tiles": 0, "tiles_with_boats": 0, "boxes": 0}
    for y0 in range(0, h - TILE + 1, stride):
        for x0 in range(0, w - TILE + 1, stride):
            # Emprise de la tuile en UTM : elle doit tenir dans la zone annotée.
            ux1, uy2 = tr * (x0, y0)
            ux2, uy1 = tr * (x0 + TILE, y0 + TILE)
            if ux1 < ex1 - PAD_M or ux2 > ex2 + PAD_M or uy1 < ey1 - PAD_M or uy2 > ey2 + PAD_M:
                continue
            if tile == "34VEN" and uy1 < VEN_MIN_NORTHING:
                continue
            crop = rgb[y0:y0 + TILE, x0:x0 + TILE]
            if (crop.max(axis=2) == 0).mean() > MAX_NODATA:
                continue
            if water[y0:y0 + TILE, x0:x0 + TILE].mean() < MIN_WATER:
                continue
            tile_boxes = boxes
            if min_diag > 0:
                crop, tile_boxes = erase_small(crop, boxes, x0, y0, min_diag)
            yolo = clip_boxes(tile_boxes, x0, y0, TILE, TILE)
            if len(yolo) == 0 and split == "train" and rng.random() > keep_empty:
                continue
            name = f"{tile}_{day}_{y0:05d}_{x0:05d}"
            cv2.imwrite(str(out_img / f"{name}.png"), np.ascontiguousarray(crop[:, :, ::-1]))
            (out_lbl / f"{name}.txt").write_text(
                "".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for a, b, c, d in yolo))
            stats["tiles"] += 1
            stats["tiles_with_boats"] += int(len(yolo) > 0)
            stats["boxes"] += len(yolo)
    print(json.dumps(stats), flush=True)
    return stats


def add_rot90(ds: Path) -> int:
    """Copie tournée de 90° de chaque tuile de train. Avec les flips d'Ultralytics,
    on couvre les huit orientations ; la rotation libre gonflerait les boîtes."""
    img_dir, lbl_dir = ds / "images" / "train", ds / "labels" / "train"
    n = 0
    for ip in sorted(img_dir.glob("*.png")):
        if ip.stem.endswith("_r90"):
            continue
        out = img_dir / f"{ip.stem}_r90.png"
        lbl = (lbl_dir / f"{ip.stem}.txt").read_text().split("\n")
        rows = [r.split() for r in lbl if r.strip()]
        # Rotation antihoraire : (x, y) devient (y, 1 - x), largeur et hauteur s'échangent.
        rot = "".join(f"{c} {float(y):.6f} {1 - float(x):.6f} {float(h):.6f} {float(w):.6f}\n"
                      for c, x, y, w, h in rows)
        (lbl_dir / f"{ip.stem}_r90.txt").write_text(rot)
        if not out.exists():
            cv2.imwrite(str(out), cv2.rotate(cv2.imread(str(ip)), cv2.ROTATE_90_COUNTERCLOCKWISE))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--rot90-only", action="store_true", help="ajoute seulement les tuiles tournées")
    ap.add_argument("--variant", default="", help="nom d'une variante du train, val et test inchangés")
    ap.add_argument("--keep-empty", type=float, default=KEEP_EMPTY_TRAIN)
    ap.add_argument("--min-diag", type=float, default=0.0, help="efface les navires plus petits, en pixels")
    args = ap.parse_args()
    ds = args.out / "datasets" / (f"finnish_{args.variant}" if args.variant else "finnish")
    if args.rot90_only:
        print(add_rot90(ds), "tuiles tournées")
        return
    raw = args.out / "raw" / "finnish"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = []
    for tile in SPLITS:
        gpkg = download(ZENODO.format(name=tile), raw / f"{tile}.gpkg")
        layers = read_layers(gpkg)
        allb = np.concatenate([lay["boxes"] for lay in layers.values()])
        envelope = (allb[:, 0].min(), allb[:, 1].min(), allb[:, 2].max(), allb[:, 3].max())
        for day, lay in layers.items():
            jobs.append((tile, day, lay, envelope))
    with ThreadPoolExecutor(args.jobs) as ex:
        stats = list(ex.map(lambda j: process_scene(*j, args.out, random.Random(f"{args.seed}{j[0]}{j[1]}"),
                                                    args.variant, args.keep_empty, args.min_diag), jobs))
    stats = [s for s in stats if s]
    print(add_rot90(ds), "tuiles tournées")
    base = args.out / "datasets" / "finnish"
    (ds / "data.yaml").write_text(
        f"path: {ds}\ntrain: images/train\nval: {base / 'images/val'}\ntest: {base / 'images/test'}\n"
        "names:\n  0: boat\n")
    summary = {}
    for s in stats:
        agg = summary.setdefault(s["split"], {"scenes": 0, "tiles": 0, "tiles_with_boats": 0, "boxes": 0})
        agg["scenes"] += 1
        for k in ("tiles", "tiles_with_boats", "boxes"):
            agg[k] += s[k]
    (ds / "manifest.json").write_text(json.dumps({
        "source": "https://doi.org/10.5281/zenodo.10046342", "license": "CC-BY-4.0",
        "imagery": "Sentinel-2 L2A TCI, Element 84 earth-search v1", "tile": TILE, "stride": STRIDE,
        "min_water": MIN_WATER, "keep_empty_train": args.keep_empty, "min_diag": args.min_diag,
        "variant": args.variant, "seed": args.seed,
        "scenes": stats, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
