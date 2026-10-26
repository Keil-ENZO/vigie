"""Commandes radar : `vigie scenes-sar` et `vigie scan-sar`.

Branchées sur l'application de `cli.py` par `register(app)`. Utilisables seules
aussi : `python -m vigie.cli_sar scan-sar --preset skagen`.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _bbox(text: str):
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("format : ouest,sud,est,nord")
    return tuple(parts)


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


def scenes_sar(bbox: str = typer.Argument(None, help="ouest,sud,est,nord"), preset: str = typer.Option(None),
               days: int = 30):
    """Liste les passes Sentinel-1 disponibles sur une zone, de jour comme de nuit."""
    from . import sar

    box, _ = _zone(bbox, preset)
    t = Table("passe", "date UTC", "orbite", "couverture")
    for s in sar.search(box, days=days, limit=50):
        t.add_row(s.id, s.datetime[:16], s.orbit_state or "", f"{(s.coverage or 0) * 100:.0f} %")
    console.print(t)


def scan_sar(bbox: str = typer.Argument(None, help="ouest,sud,est,nord"), preset: str = typer.Option(None),
             scene: str = typer.Option(None, help="identifiant de passe Sentinel-1"), days: int = 12,
             k: float = typer.Option(None, help="seuil CFAR en écarts-types"), analyse: bool = False):
    """Scanne une zone sur la dernière passe radar Sentinel-1."""
    from . import cfar, sar
    from . import scan_sar as scan_mod

    box, name = _zone(bbox, preset)
    params = cfar.CfarParams() if k is None else cfar.CfarParams(k=k)
    sc = sar.get_scene(scene) if scene else None
    meta = scan_mod.run(box, scene=sc, params=params, days=days, name=name,
                        log=lambda m: console.print(f"[dim]{m}[/]"))
    console.print(f"scan radar [bold]{meta['id']}[/] : {meta['counts']['vessels']} navires")
    if analyse:
        from . import analyst

        res = analyst.analyse(meta["id"], progress=lambda d, t, m: console.print(f"[dim]{d}/{t} {m}[/]"))
        console.print(res["report_md"])


def register(app: typer.Typer) -> None:
    app.command("scenes-sar")(scenes_sar)
    app.command("scan-sar")(scan_sar)


app = typer.Typer(help="Vigie radar, Sentinel-1.", no_args_is_help=True)
register(app)

if __name__ == "__main__":
    app()
