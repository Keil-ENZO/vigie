"""Format de sortie du scan radar : le même dossier qu'un scan optique."""

import json
import time

import numpy as np
import pytest
import rasterio
from test_sar_cfar import add_ship, sea

from vigie import sar, scan, scan_sar

BBOX = (10.50, 57.55, 10.53, 57.57)


@pytest.fixture
def fake_pass(tmp_path, monkeypatch):
    """Une passe synthétique : deux navires sur la mer, une bande de terre à l'ouest."""
    monkeypatch.setattr(scan, "SCANS", tmp_path / "scans")
    vv, vh = sea(300, seed=1), sea(300, seed=2, mean_db=-26)
    vv[:, :40] = 0.3
    add_ship(vv, 150, 150, 18, 30, db=12)
    add_ship(vh, 150, 150, 18, 30, db=-2)
    add_ship(vv, 220, 80, 8, 90, db=8)
    water = np.ones(vv.shape, bool)
    water[:, :45] = False
    raster = sar.SarRaster(vv=vv, vh=vh, transform=rasterio.Affine(10, 0, 530000, 0, -10, 6380000),
                           crs="EPSG:32632", gsd=10.0, water=water)
    scene = sar.SarScene(id="S1C_IW_GRDH_1SDV_TEST_rtc", datetime="2026-10-18T17:01:18.137522Z",
                         platform="sentinel-1c", collection=sar.RTC, vv_href="vv", vh_href="vh",
                         orbit_state="ascending", relative_orbit=44, epsg=32632, thumbnail=None, coverage=1.0)
    monkeypatch.setattr(sar, "load", lambda *a, **k: raster)
    monkeypatch.setattr(sar, "search", lambda *a, **k: [scene])
    monkeypatch.setattr(scan_sar.infra, "fetch", lambda bbox: [])
    return scene


def test_scan_writes_same_layout_as_optical(fake_pass):
    meta = scan_sar.run(BBOX, name="test", log=lambda m: None)
    d = scan.scan_dir(meta["id"])
    for f in ("scan.json", "detections.geojson", "image.jpg"):
        assert (d / f).exists()
    m = json.loads((d / "scan.json").read_text())
    assert m["sensor"] == "sentinel-1"
    # Les clés que lisent l'interface, l'analyste et ais.crosscheck.
    for key in ("id", "name", "created", "bbox", "scene", "params", "model", "size_px", "gsd_m", "crs", "corners",
                "water_mask", "counts", "timing_s"):
        assert key in m
    for key in ("id", "datetime", "cloud_cover", "platform"):
        assert key in m["scene"]
    assert set(m["counts"]) == {"detections", "vessels", "fixed"}
    assert set(m["timing_s"]) >= {"read", "detect", "total"}
    assert len(m["corners"]) == 4

    det = json.loads((d / "detections.geojson").read_text())
    assert det["type"] == "FeatureCollection"
    assert len(det["features"]) == 2
    optical_keys = {"id", "confidence", "label", "lat", "lon", "length_m", "width_m", "heading_deg", "course_deg",
                    "wake_m", "underway", "measured", "box_px", "water_fraction"}
    for f in det["features"]:
        assert optical_keys <= set(f["properties"])
        assert (d / "crops" / f"{f['properties']['id']}.png").exists()
        assert f["geometry"]["type"] == "Polygon"
    big = det["features"][0]["properties"]
    assert big["id"] == "V001" and abs(big["length_m"] - 170) <= 40

    # La lecture commune marche telle quelle.
    meta2, det2 = scan.load_scan(meta["id"])
    assert meta2["id"] == meta["id"] and len(det2["features"]) == 2
    assert any(x["id"] == meta["id"] for x in scan.list_scans())


def test_api_route_submits_a_job(fake_pass, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vigie import jobs, sar_api

    app = FastAPI()
    app.include_router(sar_api.router)
    client = TestClient(app)
    r = client.post("/api/scans-sar", json={"bbox": list(BBOX), "name": "test"})
    assert r.status_code == 202
    job_id = r.json()["id"]
    for _ in range(100):
        job = jobs.get(job_id)
        if job.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert job.status == "done", job.error
    assert (scan.scan_dir(job.result["scan_id"]) / "scan.json").exists()
    assert client.post("/api/scans-sar", json={"bbox": [10, 57, 11, 58]}).status_code == 422


def test_search_finds_danish_pass():
    from datetime import date

    try:
        scenes = sar.search(BBOX, days=2, until=date(2026, 10, 18))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"réseau indisponible : {exc}")
    assert any(s.id.startswith("S1C_IW_GRDH_1SDV_20261018T1701") for s in scenes)
    assert all(s.vh_href and s.collection == sar.RTC for s in scenes)
