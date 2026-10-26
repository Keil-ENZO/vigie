"""Validation AIS sur plusieurs passes danoises : un tableau par scène et un total.

    uv run python scripts/validate_ais.py [--conf 0.2] [--model models/x.pt]

Chaque ligne : zone, passe, fenêtre. Le script scanne, croise avec l'AIS, et écrit
docs/validation-ais.md et docs/validation-ais.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vigie import ais, detector, scan, sentinel

CASES = [
    ("Skagen, mouillage", (10.50, 57.55, 10.80, 57.72), "S2B_32VNJ_20261018_0_L2A"),
    ("Aarhus, baie", (10.18, 56.08, 10.40, 56.20), "S2B_32VNH_20261018_0_L2A"),
    ("Fredericia, Petit Belt", (9.65, 55.50, 9.85, 55.62), "S2B_32UNG_20261018_0_L2A"),
    ("Øresund, Copenhague", (12.55, 55.58, 12.80, 55.75), "S2C_33UUB_20260927_0_L2A"),
    ("Grand Belt, pont", (10.85, 55.25, 11.10, 55.40), "S2C_32UPG_20260930_0_L2A"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--model", type=Path, default=detector.DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=Path("docs/validation-ais"))
    args = ap.parse_args()
    rows, pooled_err = [], []
    for name, bbox, scene_id in CASES:
        try:
            ais.ensure_day(f"{scene_id.split('_')[2][:4]}-{scene_id.split('_')[2][4:6]}-{scene_id.split('_')[2][6:8]}")
        except LookupError as exc:
            print(f"{name} : {exc}")
            continue
        sc = sentinel.get_scene(scene_id)
        detector.DEFAULT_MODEL = args.model
        meta = scan.run(bbox, scene=sc, params=detector.DetectParams(confidence=args.conf), name=name)
        res = ais.crosscheck(meta["id"])
        s = res["summary"]
        meta2, det = scan.load_scan(meta["id"])
        for f in det["features"]:
            p = f["properties"]
            if p.get("ais", {}).get("status") == "matched" and p["ais"].get("length_m"):
                pooled_err.append(p["length_m"] - p["ais"]["length_m"])
        big_det = sum(1 for f in det["features"] if f["properties"].get("ais", {}).get("status") == "matched"
                      and (f["properties"]["ais"].get("length_m") or 0) >= 50)
        rows.append({"zone": name, "scene": scene_id, "scan_id": meta["id"], "ais": s["ais_vessels"],
                     "ais_50m": s["ais_50m"], "detections": s["detections"], "matched": s["matched"],
                     "dark": s["dark"], "precision": round(s["matched"] / s["detections"], 3) if s["detections"] else None,
                     "recall_50m": s["recall_ais_50m"], "matched_50m": big_det,
                     "len_mae": s["length_error_m"]["median_abs"]})
        print(rows[-1])
    tot = {k: sum(r[k] for r in rows) for k in ("ais", "ais_50m", "detections", "matched", "dark", "matched_50m")}
    tot["precision"] = round(tot["matched"] / tot["detections"], 3) if tot["detections"] else None
    tot["recall_50m"] = round(tot["matched_50m"] / tot["ais_50m"], 3) if tot["ais_50m"] else None
    tot["len_mae"] = round(float(np.median(np.abs(pooled_err))), 1) if pooled_err else None
    tot["len_bias"] = round(float(np.median(pooled_err)), 1) if pooled_err else None
    args.out.parent.mkdir(exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({"model": args.model.name, "conf": args.conf,
                                                         "rows": rows, "total": tot}, indent=2, ensure_ascii=False))
    lines = [f"# Validation AIS, modèle `{args.model.name}`, seuil {args.conf}", "",
             "| Zone | Passe | AIS | AIS > 50 m | Détections | Appariées | Sombres | Précision | Rappel > 50 m | Longueur, err. méd. |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda v: "–" if v is None else f"{v * 100:.0f} %"  # noqa: E731
    for r in rows:
        lines.append(f"| {r['zone']} | {r['scene'][:24]} | {r['ais']} | {r['ais_50m']} | {r['detections']} | {r['matched']} | "
                     f"{r['dark']} | {pct(r['precision'])} | {pct(r['recall_50m'])} | {r['len_mae']} m |")
    lines.append(f"| **Total** | | {tot['ais']} | {tot['ais_50m']} | {tot['detections']} | {tot['matched']} | {tot['dark']} | "
                 f"**{pct(tot['precision'])}** | **{pct(tot['recall_50m'])}** | {tot['len_mae']} m, biais {tot['len_bias']} m |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
