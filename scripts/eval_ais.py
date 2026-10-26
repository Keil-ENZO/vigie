"""Évalue un scan croisé avec l'AIS : longueurs, sillages, catégories du VLM.

    uv run python scripts/eval_ais.py <scan_id>
"""

import json
import sys
from collections import Counter

import numpy as np

from vigie import scan

meta, det = scan.load_scan(sys.argv[1])
ps = [f["properties"] for f in det["features"]]
m = [p for p in ps if p.get("ais", {}).get("status") == "matched"]
print(f"scan {meta['id']} {meta.get('name')} : {len(ps)} détections, {len(m)} appariées AIS")

err = [p["length_m"] - p["ais"]["length_m"] for p in m if p["ais"].get("length_m")]
if err:
    e = np.array(err)
    rel = np.array([abs(p["length_m"] - p["ais"]["length_m"]) / p["ais"]["length_m"] for p in m if p["ais"].get("length_m")])
    print(f"longueur : erreur absolue médiane {np.median(np.abs(e)):.0f} m, biais {np.median(e):+.0f} m, "
          f"erreur relative médiane {np.median(rel) * 100:.0f} %")

truth = [(p.get("underway", False), (p["ais"].get("sog_kn") or 0) >= 1.0) for p in m]
tp = sum(a and b for a, b in truth)
fp = sum(a and not b for a, b in truth)
fn = sum(b and not a for a, b in truth)
tn = sum(not a and not b for a, b in truth)
print(f"en route (sillage vs SOG >= 1 nd) : VP {tp}, FP {fp}, FN {fn}, VN {tn}")

pairs = [(p["analysis"]["category"], p["ais"]["category"]) for p in m if p.get("analysis") and "category" in p["analysis"]]
if pairs:
    ok = sum(a == b for a, b in pairs)
    print(f"VLM catégorie : {ok}/{len(pairs)} justes ({ok / len(pairs) * 100:.0f} %), "
          f"inconnu {sum(a == 'unknown' for a, _ in pairs)}")
    print("  confusions :", dict(Counter(f"{b}->{a}" for a, b in pairs if a != b)))
print(json.dumps(det.get("ais_summary", {}), ensure_ascii=False))
