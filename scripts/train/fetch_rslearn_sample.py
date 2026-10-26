"""Extrait un échantillon du jeu « sentinel2_vessels » d'Ai2 (rslearn_projects).

Le tar fait 52 Go : on le lit en flux et on s'arrête après --max-bytes, en ne
gardant que l'image RGB, les étiquettes (points) et les métadonnées de chaque
fenêtre. Sert uniquement de test hors domaine, mondial, étiquettes ponctuelles.

Usage : uv run python scripts/train/fetch_rslearn_sample.py --max-bytes 2e9
"""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path

URL = "https://storage.googleapis.com/ai2-rslearn-projects-data/sentinel2_vessels/sentinel2_vessels.tar"
KEEP = ("/metadata.json", "/items.json", "/layers/sentinel2/R_G_B/image.png", "/layers/label/data.geojson")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/rslearn_s2_vessels"))
    ap.add_argument("--max-bytes", type=float, default=2e9)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    curl = subprocess.Popen(["curl", "-s", "-r", f"0-{int(args.max_bytes)}", URL], stdout=subprocess.PIPE)
    kept = 0
    try:
        with tarfile.open(fileobj=curl.stdout, mode="r|") as tar:
            for m in tar:
                if not m.isfile() or not m.name.startswith("windows/") or not m.name.endswith(KEEP):
                    continue
                # Les noms viennent d'une archive non vérifiée : on refuse tout chemin bizarre.
                if ".." in Path(m.name).parts or m.name.startswith("/"):
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                dest = args.out / m.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f.read())
                kept += 1
    except (tarfile.ReadError, EOFError):
        pass  # flux coupé volontairement à --max-bytes
    finally:
        curl.kill()
    print(f"{kept} fichiers extraits dans {args.out}")


if __name__ == "__main__":
    main()
