"""Ligne de commande : `vigie scan`, `vigie analyse`, `vigie serve`, `vigie demo`, `vigie watch`, `vigie fetch-model`."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Vigie, surveillance maritime locale.", no_args_is_help=True)
console = Console()
# Un .pt est un pickle : la révision est figée et l'empreinte vérifiée avant tout chargement.
MODEL_REV = "67f6ef787f726bc874b35a96541c6b61732f699d"
MODEL_URL = f"https://huggingface.co/DefendIntelligence/vessel-detection/resolve/{MODEL_REV}/models/best.pt"
MODEL_SHA256 = "db4ddd5e9603d31051dcee57a8f035180cedd30834a41da44619d8c020695cf9"
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "skagen"


@app.command("fetch-model")
def fetch_model():
    """Télécharge le modèle de référence, YOLOv8s Sentinel-2 de DefendIntelligence, licence MIT."""
    import httpx

    from .detector import DEFAULT_MODEL

    if DEFAULT_MODEL.exists():
        console.print(f"déjà là : {DEFAULT_MODEL}")
        return
    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp, digest = DEFAULT_MODEL.with_suffix(".part"), hashlib.sha256()
    with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes():
                digest.update(chunk)
                f.write(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        tmp.unlink()
        console.print("[red]empreinte du modèle inattendue, fichier supprimé[/]")
        raise typer.Exit(1)
    tmp.rename(DEFAULT_MODEL)
    console.print(f"modèle enregistré : {DEFAULT_MODEL}")


@app.command()
def scenes(bbox: str = typer.Argument(..., help="ouest,sud,est,nord"), days: int = 60, max_cloud: float = 30):
    """Liste les passes Sentinel-2 disponibles sur une zone."""
    from . import sentinel

    t = Table("passe", "date UTC", "nuages %")
    for s in sentinel.search(_bbox(bbox), days=days, max_cloud=max_cloud, limit=20):
        t.add_row(s.id, s.datetime[:16], f"{s.cloud_cover:.1f}")
    console.print(t)


@app.command()
def scan(bbox: str = typer.Argument(None, help="ouest,sud,est,nord"), preset: str = typer.Option(None),
         scene: str = typer.Option(None, help="identifiant de passe"), confidence: float = 0.20,
         days: int = 30, max_cloud: float = 20, analyse: bool = False):
    """Scanne une zone sur la dernière passe sans nuages."""
    from . import detector, sentinel
    from . import scan as scan_mod

    box, name = _zone(bbox, preset)
    sc = sentinel.get_scene(scene) if scene else None
    meta = scan_mod.run(box, scene=sc, params=detector.DetectParams(confidence=confidence), days=days,
                        max_cloud=max_cloud, name=name, log=lambda m: console.print(f"[dim]{m}[/]"))
    console.print(f"scan [bold]{meta['id']}[/] : {meta['counts']['vessels']} navires")
    if analyse:
        _analyse(meta["id"])


def _analyse(scan_id: str):
    from . import analyst

    if not analyst.available():
        console.print(f"[red]modèle {analyst.MODEL} introuvable sur {analyst.URL}[/] "
                      "(voir VIGIE_LLM_API, VIGIE_LLM_URL, VIGIE_VLM)")
        raise typer.Exit(1)
    res = analyst.analyse(scan_id, progress=lambda d, t, m: console.print(f"[dim]{d}/{t} {m}[/]"))
    console.print(res["report_md"])


@app.command()
def analyse(scan_id: str):
    """Fait regarder chaque navire d'un scan par le VLM, puis écrit le rapport."""
    _analyse(scan_id)


@app.command("list")
def list_cmd():
    """Liste les scans enregistrés."""
    from . import scan as scan_mod

    t = Table("scan", "zone", "passe", "navires", "analysé")
    for m in scan_mod.list_scans():
        t.add_row(m["id"], m.get("name") or json.dumps(m["bbox"]), m["scene"]["datetime"][:10],
                  str(m["counts"]["vessels"]), "oui" if m["analysed"] else "")
    console.print(t)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8740):
    """Lance l'interface web. Depuis une autre machine : ssh -L 8740:localhost:8740 <serveur>."""
    import uvicorn

    os.environ["VIGIE_BIND"] = host
    uvicorn.run("vigie.api:app", host=host, port=port, log_level="info")


@app.command()
def demo(port: int = 8740):
    """Ouvre un vrai scan de Skagen, sans GPU ni modèle : navires, AIS et rapport déjà calculés."""
    from . import scan as scan_mod

    scan_id = json.loads((EXAMPLE / "scan.json").read_text())["id"]
    target = scan_mod.scan_dir(scan_id)
    if not target.exists():
        shutil.copytree(EXAMPLE, target)
    console.print(f"ouvre http://127.0.0.1:{port}/?scan={scan_id}")
    serve(port=port)


@app.command()
def watch(bbox: str = typer.Argument(None, help="ouest,sud,est,nord"), preset: str = typer.Option(None),
          ntfy: str = typer.Option(None, envvar="VIGIE_NTFY", help="adresse ntfy, https://ntfy.sh/mon-sujet"),
          days: int = 7, max_cloud: float = 20):
    """Vérifie une zone une fois : scanne la nouvelle passe s'il y en a une, et prévient. À lancer par cron."""
    from . import watch as watch_mod

    box, name = _zone(bbox, preset)
    res = watch_mod.check(box, name=name, ntfy=ntfy, days=days, max_cloud=max_cloud,
                          log=lambda m: console.print(f"[dim]{m}[/]"))
    if res:
        console.print(res["message"])


def _zone(bbox: str | None, preset: str | None):
    from .presets import PRESETS

    if preset:
        p = next((p for p in PRESETS if p["id"] == preset), None)
        if p is None:
            raise typer.BadParameter(f"zones connues : {', '.join(p['id'] for p in PRESETS)}")
        return tuple(p["bbox"]), p["name"]
    if bbox:
        return _bbox(bbox), None
    raise typer.BadParameter("donne une boîte ou --preset")


def _bbox(text: str):
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("format : ouest,sud,est,nord")
    return tuple(parts)


from .cli_sar import register as _register_sar

_register_sar(app)

if __name__ == "__main__":
    app()
