"""L'analyste : un VLM local regarde chaque navire, puis rédige un rapport de situation.

Deux API au choix. Ollama par défaut. Sinon toute API compatible OpenAI : LM Studio,
vLLM, llama.cpp server, OpenRouter, OpenAI. Le modèle doit accepter les images.

    VIGIE_LLM_API   ollama (défaut) ou openai
    VIGIE_LLM_URL   http://localhost:11434 pour Ollama, http://localhost:1234/v1 pour LM Studio
    VIGIE_LLM_KEY   clé API, seulement si le serveur en demande une
    VIGIE_VLM       nom du modèle

À 10 m par pixel, un VLM ne lit pas un nom de coque. Il voit une silhouette, un
sillage, un contexte. On lui donne donc la longueur mesurée et on lui demande de
dire « inconnu » plutôt que d'inventer. Le rapport se fonde sur les chiffres de la
détection, pas sur l'imagination du modèle.
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx

from . import scan
from .thermal import wait_if_hot

API = os.environ.get("VIGIE_LLM_API", "ollama").lower()
if API not in ("ollama", "openai"):
    raise ValueError(f"VIGIE_LLM_API={API} : ollama ou openai")
URL = os.environ.get("VIGIE_LLM_URL") or os.environ.get("VIGIE_OLLAMA") or (
    "http://localhost:11434" if API == "ollama" else "http://localhost:1234/v1")
URL = URL.rstrip("/")
KEY = os.environ.get("VIGIE_LLM_KEY")
MODEL = os.environ.get("VIGIE_VLM", "gemma4:26b-a4b-it-q4_K_M")
CATEGORIES = ["cargo", "tanker", "passenger", "military", "fishing", "pleasure", "service", "unknown"]

VESSEL_SCHEMA = {
    "type": "object",
    "properties": {
        "is_vessel": {"type": "boolean"},
        "category": {"type": "string", "enum": CATEGORIES},
        "moving": {"type": "boolean"},
        "confidence": {"type": "number"},
        "note": {"type": "string"},
    },
    "required": ["is_vessel", "category", "moving", "confidence", "note"],
    "additionalProperties": False,
}

VESSEL_PROMPT = """You are a maritime imagery analyst. This is a Sentinel-2 true-colour crop, 10 m per pixel,
480 m wide, upscaled 4x. The detected object is in the centre.
Measured on the pixels: hull length about {length} m (10 m pixels, so ±10 m), axis {heading}° from north.
{wake} Detector confidence {conf}.
Context: {context}.

Answer in JSON:
- is_vessel: false if it is a pier, rock, buoy, cloud, glint or artefact.
- category: pick from visual cues and size. Use "unknown" when the evidence is weak.
  Visual cues at 10 m resolution:
  * tanker: long plain deck, often red, green or dark, white superstructure at the stern, 100-330 m.
  * cargo: bulk carrier or container ship. Segmented deck (hatches) or many coloured blocks
    (containers), superstructure at the stern, 90-400 m.
  * passenger: ferry or cruise ship, white over most of its length, superstructure spanning
    the hull, usually underway on a fixed route near a port.
  * military: uniform grey hull, at a naval base or in formation.
  * fishing: 15-40 m, often working in groups.
  * pleasure: under 30 m, near marinas or coasts.
  * service: tug, pilot, workboat, 15-40 m, near ports or alongside a big ship.
  Most large ships waiting at an offshore anchorage are cargo ships or tankers, not ferries.
- moving: true only if a wake is visible.
- confidence: 0 to 1, your confidence in the category.
- note: one short sentence in French, what you see. No speculation about names or flags."""

REPORT_PROMPT = """Tu es analyste en renseignement maritime. Rédige un court rapport de situation en français,
à partir des seules données JSON ci-dessous : une passe Sentinel-2 et les navires détectés.
Structure en Markdown : un titre, « Synthèse » en 2-3 phrases, « Activité » en puces (navires en route avec leur cap,
au mouillage, répartition par type), « Points d'attention » (les plus grands, les détections douteuses),
« Limites » (résolution de 10 m, une seule passe, catégories incertaines).
Si la clé « ais » est présente, ajoute une section « Croisement AIS » : navires identifiés (noms tels quels),
navires sombres, c'est-à-dire vus mais sans AIS, à vérifier. Sans AIS, ne donne aucun nom.
N'invente aucun fait absent des données. 300 mots maximum.

Données :
{data}"""


def _headers() -> dict:
    return {"Authorization": f"Bearer {KEY}"} if KEY else {}


def _chat(content: str, images: list[str] | None = None, schema: dict | None = None, timeout: float = 300) -> str:
    if API == "openai":
        return _chat_openai(content, images, schema, timeout)
    msg = {"role": "user", "content": content}
    if images:
        msg["images"] = images
    body = {"model": MODEL, "stream": False, "think": False, "messages": [msg],
            "options": {"temperature": 0.2}, "keep_alive": "5m"}
    if schema:
        body["format"] = schema
    r = httpx.post(f"{URL}/api/chat", json=body, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _chat_openai(content: str, images: list[str] | None, schema: dict | None, timeout: float) -> str:
    parts = [{"type": "text", "text": content}]
    parts += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in images or []]
    body = {"model": MODEL, "temperature": 0.2, "messages": [{"role": "user", "content": parts}]}
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "vessel", "schema": schema, "strict": True}}
    r = httpx.post(f"{URL}/chat/completions", json=body, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def _json(raw: str) -> dict:
    """Certains serveurs ignorent le schéma et entourent le JSON de ```json."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(raw)


def models() -> list[str]:
    """Les modèles que le serveur annonce, vide s'il est injoignable."""
    try:
        if API == "openai":
            r = httpx.get(f"{URL}/models", headers=_headers(), timeout=3)
            return [m["id"] for m in r.json().get("data", [])]
        r = httpx.get(f"{URL}/api/tags", headers=_headers(), timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return []


def available() -> bool:
    names = models()
    return MODEL in names or f"{MODEL}:latest" in names


def classify(crop: Path, props: dict, context: str) -> dict:
    img = base64.b64encode(crop.read_bytes()).decode()
    prompt = VESSEL_PROMPT.format(length=props.get("length_m"), width=props.get("width_m") or "?",
                                  heading=props.get("heading_deg") if props.get("heading_deg") is not None else "?",
                                  conf=props.get("confidence"), context=context,
                                  wake=(f"A wake of about {props['wake_m']:.0f} m was measured: the vessel is underway, "
                                        f"course {props['course_deg']:.0f}°." if props.get("wake_m") else
                                        "No clear wake was measured."))
    raw = _chat(prompt, [img], VESSEL_SCHEMA)
    out = _json(raw)
    out["category"] = out.get("category") if out.get("category") in CATEGORIES else "unknown"
    out["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0))))
    return out


def summarize(meta: dict, features: list[dict], vessels: dict) -> dict:
    kept = [f["properties"] | {"analysis": vessels.get(f["properties"]["id"])} for f in features]
    ais_facts = None
    if any("ais" in p for p in kept):
        ais_facts = {
            "apparies": [{"id": p["id"], "nom": p["ais"].get("name"), "type_ais": p["ais"].get("ship_type"),
                          "longueur_declaree_m": p["ais"].get("length_m"), "statut": p["ais"].get("status")}
                         for p in kept if p.get("ais", {}).get("status") == "matched"],
            "sombres": [p["id"] for p in kept if p.get("ais", {}).get("status") == "dark"],
        }
    real = [p for p in kept if not p["analysis"] or p["analysis"].get("is_vessel", True)]
    return {
        "zone": meta.get("name") or meta["bbox"], "passe": meta["scene"]["id"],
        "date_utc": meta["scene"]["datetime"], "nuages_pct": meta["scene"]["cloud_cover"],
        "detections": len(kept), "navires_retenus": len(real),
        "rejetes_par_vlm": len(kept) - len(real),
        # « en route » vient du sillage mesuré, plus sûr que l'avis du VLM.
        "en_route": sum(1 for p in real if p.get("underway")),
        "par_type": dict(Counter((p["analysis"] or {}).get("category", "unknown") for p in real)),
        "ais": ais_facts,
        "navires": [{k: p.get(k) for k in ("id", "length_m", "heading_deg", "confidence", "lat", "lon")}
                    | {"type": (p["analysis"] or {}).get("category"), "en_route": bool(p.get("underway")),
                       "cap_deg": p.get("course_deg"), "sillage_m": p.get("wake_m")}
                    for p in sorted(real, key=lambda p: -(p.get("length_m") or 0))],
    }


def analyse(scan_id: str, progress=lambda done, total, msg: None) -> dict:
    meta, det = scan.load_scan(scan_id)
    d = scan.scan_dir(scan_id)
    feats = [f for f in det["features"] if not f["properties"].get("fixed")]
    context = f"area {meta.get('name') or 'unnamed'}, bbox {meta['bbox']}, acquired {meta['scene']['datetime'][:10]}"
    vessels, t0 = {}, time.perf_counter()
    for i, f in enumerate(feats):
        wait_if_hot(log=lambda m: progress(i, len(feats) + 1, m))
        p = f["properties"]
        progress(i, len(feats) + 1, f"analyse de {p['id']}")
        try:
            vessels[p["id"]] = classify(d / "crops" / f"{p['id']}.png", p, context)
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            vessels[p["id"]] = {"error": str(exc)}
    progress(len(feats), len(feats) + 1, "rédaction du rapport")
    facts = summarize(meta, feats, vessels)
    report = _chat(REPORT_PROMPT.format(data=json.dumps(facts, ensure_ascii=False)), timeout=600)
    result = {"model": MODEL, "seconds": round(time.perf_counter() - t0, 1), "facts": facts,
              "vessels": vessels, "report_md": report.strip()}
    (d / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (d / "report.md").write_text(result["report_md"])
    progress(len(feats) + 1, len(feats) + 1, "terminé")
    return result
