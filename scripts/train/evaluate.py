"""Évalue un ou plusieurs modèles YOLO sur un split d'un jeu au format YOLO.

Métriques, toutes calculées ici et pas par Ultralytics, pour que les deux modèles
passent par le même code :
  - mAP50 et mAP50-95, interpolation 101 points façon COCO ;
  - précision, rappel, F1 au seuil 0,20 (celui du pipeline) et au meilleur F1 ;
  - les mêmes en « centre » : une prédiction est juste si son centre tombe dans une
    boîte de vérité libre. Ce critère ne dépend pas de la convention de boîte
    (sillage inclus ou non) et dit si le navire est trouvé ;
  - rappel par taille de navire, au seuil 0,20, critère centre.

NMS à IoU 0,45, comme `vigie.detector`. Sortie : JSON + tableau Markdown, et les
prédictions brutes (npz) pour les figures.

Usage :
  uv run python scripts/train/evaluate.py --data .../data.yaml --split test \
      --weights models/defendintel-yolov8s-s2.pt models/vigie-s2-v1.pt --out runs/eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from vigie.thermal import wait_if_hot

IOUS = np.linspace(0.5, 0.95, 10)
PIPELINE_CONF = 0.20
SIZES = {"petit (<8 px)": (0, 8), "moyen (8-20 px)": (8, 20), "grand (>20 px)": (20, 1e9)}


def load_split(data_yaml: Path, split: str) -> tuple[list[Path], list[np.ndarray]]:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    img_dir = root / cfg[split]
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".tif"})
    gts = []
    for p in images:
        lbl = Path(str(p.parent).replace("/images/", "/labels/")) / f"{p.stem}.txt"
        rows = np.loadtxt(lbl, ndmin=2) if lbl.exists() and lbl.stat().st_size else np.zeros((0, 5))
        gts.append(rows[:, 1:5])
    return images, gts


def xywhn_to_xyxy(b: np.ndarray, w: int, h: int) -> np.ndarray:
    if len(b) == 0:
        return np.zeros((0, 4))
    x, y, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
    return np.stack([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], axis=1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def center_inside(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Matrice pred x gt : 1 si le centre de la prédiction est dans la boîte vraie.
    La valeur est la proximité au centre vrai, pour départager plusieurs candidats."""
    cx, cy = (pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2
    inside = ((cx[:, None] >= gt[None, :, 0]) & (cx[:, None] <= gt[None, :, 2])
              & (cy[:, None] >= gt[None, :, 1]) & (cy[:, None] <= gt[None, :, 3]))
    gcx, gcy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
    d = np.hypot(cx[:, None] - gcx[None, :], cy[:, None] - gcy[None, :])
    return np.where(inside, 1.0 / (1.0 + d), 0.0)


def greedy_match(score: np.ndarray, thr: float) -> np.ndarray:
    """Prédictions triées par confiance décroissante. Rend l'indice de la vérité
    appariée, ou -1."""
    n_pred, n_gt = score.shape
    out = np.full(n_pred, -1)
    taken = np.zeros(n_gt, bool)
    for i in range(n_pred):
        s = np.where(taken, -1.0, score[i])
        j = int(np.argmax(s)) if n_gt else -1
        if j >= 0 and s[j] >= thr and s[j] > 0:
            out[i] = j
            taken[j] = True
    return out


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(1 - tp[order])
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def prf_at(tp: np.ndarray, conf: np.ndarray, n_gt: int, thr: float) -> dict:
    m = conf >= thr
    t = float(tp[m].sum())
    p = t / max(m.sum(), 1)
    r = t / max(n_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"conf": round(thr, 3), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": int(t), "fp": int(m.sum() - t), "fn": int(n_gt - t)}


def best_f1(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> dict:
    order = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[order])
    n = np.arange(1, len(order) + 1)
    p, r = tpc / n, tpc / max(n_gt, 1)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    if len(f1) == 0:
        return prf_at(tp, conf, n_gt, 1.0)
    i = int(np.argmax(f1))
    return prf_at(tp, conf, n_gt, float(conf[order][i]))


def predict_all(weights: Path, images: list[Path], imgsz: int, batch: int, device) -> list[np.ndarray]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    preds = []
    for i in range(0, len(images), batch):
        wait_if_hot()
        res = model.predict([str(p) for p in images[i:i + batch]], conf=0.001, iou=0.45, imgsz=imgsz,
                            max_det=300, device=device, verbose=False)
        for r in res:
            b = r.boxes
            preds.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], axis=1)
                         if b is not None and len(b) else np.zeros((0, 5)))
    return preds


def evaluate(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    conf_all, tp_iou, tp_ctr, size_hits = [], [], [], {k: [0, 0] for k in SIZES}
    n_gt = 0
    for pred, gt in zip(preds, gts):
        order = np.argsort(-pred[:, 4], kind="stable")
        pred = pred[order]
        n_gt += len(gt)
        conf_all.append(pred[:, 4])
        if len(gt) == 0 or len(pred) == 0:
            tp_iou.append(np.zeros((len(pred), len(IOUS))))
            tp_ctr.append(np.zeros(len(pred)))
            for k, (lo, hi) in SIZES.items():
                diag = np.hypot(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0)
                size_hits[k][1] += int(((diag >= lo) & (diag < hi)).sum())
            continue
        iou = box_iou(pred[:, :4], gt)
        tp_iou.append(np.stack([greedy_match(iou, t) >= 0 for t in IOUS], axis=1).astype(float))
        ctr_match = greedy_match(center_inside(pred[:, :4], gt), 1e-12)
        tp_ctr.append((ctr_match >= 0).astype(float))
        # Rappel par taille au seuil du pipeline : appariement refait sur les seules
        # prédictions retenues par le pipeline.
        keep = pred[:, 4] >= PIPELINE_CONF
        m = greedy_match(center_inside(pred[keep, :4], gt), 1e-12) if keep.any() else np.zeros(0, int)
        found = np.zeros(len(gt), bool)
        found[m[m >= 0]] = True
        diag = np.hypot(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1])
        for k, (lo, hi) in SIZES.items():
            sel = (diag >= lo) & (diag < hi)
            size_hits[k][0] += int(found[sel].sum())
            size_hits[k][1] += int(sel.sum())
    conf = np.concatenate(conf_all) if conf_all else np.zeros(0)
    tpi = np.concatenate(tp_iou) if tp_iou else np.zeros((0, len(IOUS)))
    tpc = np.concatenate(tp_ctr) if tp_ctr else np.zeros(0)
    aps = [average_precision(tpi[:, k], conf, n_gt) for k in range(len(IOUS))]
    return {
        "images": len(gts), "ground_truth": n_gt, "predictions_conf_ge_0.001": len(conf),
        "iou": {"map50": round(aps[0], 4), "map50_95": round(float(np.mean(aps)), 4),
                "at_0.20": prf_at(tpi[:, 0], conf, n_gt, PIPELINE_CONF),
                "best_f1": best_f1(tpi[:, 0], conf, n_gt)},
        "center": {"ap": round(average_precision(tpc, conf, n_gt), 4),
                   "at_0.20": prf_at(tpc, conf, n_gt, PIPELINE_CONF),
                   "best_f1": best_f1(tpc, conf, n_gt)},
        "recall_by_size_at_0.20": {k: {"recall": round(h / max(n, 1), 4), "n": n} for k, (h, n) in
                                   size_hits.items()},
    }


def markdown(results: dict[str, dict]) -> str:
    rows = [
        ("| Modèle | mAP50 | mAP50-95 | P@0,20 | R@0,20 | F1@0,20 | meilleur F1 (seuil) "
         "| AP centre | P centre@0,20 | R centre@0,20 | F1 centre max (seuil) |"),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        i, c = r["iou"], r["center"]
        rows.append(
            f"| {name} | {i['map50']:.3f} | {i['map50_95']:.3f} | {i['at_0.20']['precision']:.3f} "
            f"| {i['at_0.20']['recall']:.3f} | {i['at_0.20']['f1']:.3f} "
            f"| {i['best_f1']['f1']:.3f} ({i['best_f1']['conf']:.2f}) | {c['ap']:.3f} "
            f"| {c['at_0.20']['precision']:.3f} | {c['at_0.20']['recall']:.3f} "
            f"| {c['best_f1']['f1']:.3f} ({c['best_f1']['conf']:.2f}) |")
    rows += ["", "Rappel par taille de boîte vraie (diagonale), critère centre, seuil 0,20 :", "",
             "| Modèle | " + " | ".join(SIZES) + " |", "|---|" + "---|" * len(SIZES)]
    for name, r in results.items():
        s = r["recall_by_size_at_0.20"]
        rows.append(f"| {name} | " + " | ".join(f"{s[k]['recall']:.3f} (n={s[k]['n']})" for k in SIZES) + " |")
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    images, gts_n = load_split(args.data, args.split)
    import cv2

    h, w = cv2.imread(str(images[0])).shape[:2]
    gts = [xywhn_to_xyxy(g, w, h) for g in gts_n]
    results = {}
    for wpath in args.weights:
        t0 = time.perf_counter()
        preds = predict_all(wpath, images, args.imgsz, args.batch, args.device)
        np.savez_compressed(args.out / f"preds_{wpath.stem}_{args.split}.npz",
                            names=np.array([p.name for p in images]), **{f"p{i}": p for i, p in enumerate(preds)})
        r = evaluate(preds, gts)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        r["weights"] = str(wpath)
        results[wpath.stem] = r
        print(wpath.stem, json.dumps(r["iou"]), flush=True)
    meta = {"data": str(args.data), "split": args.split, "imgsz": args.imgsz, "nms_iou": 0.45}
    (args.out / f"metrics_{args.split}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    md = f"Split `{args.split}` de `{args.data}`, {len(images)} tuiles, {sum(map(len, gts))} navires.\n\n"
    md += markdown(results)
    (args.out / f"metrics_{args.split}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
