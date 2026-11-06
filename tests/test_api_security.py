import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vigie import api, cli

client = TestClient(api.app, base_url="http://localhost:8740")


def test_foreign_host_is_refused():
    assert client.get("/api/presets").status_code == 200
    assert client.get("/api/presets", headers={"host": "evil.example:8740"}).status_code == 403


def test_extra_hosts_and_wildcard(monkeypatch):
    monkeypatch.setattr(api, "HOSTS", api.HOSTS | {"vigie-box"})
    assert client.get("/api/presets", headers={"host": "vigie-box:8740"}).status_code == 200
    monkeypatch.setattr(api, "HOSTS", {"*"})
    assert client.get("/api/presets", headers={"host": "anything"}).status_code == 200


def test_cross_origin_write_is_refused():
    assert client.delete("/api/scans/x-1", headers={"origin": "https://evil.example"}).status_code == 403
    assert client.delete("/api/scans/x-1", headers={"origin": "http://localhost:8740"}).status_code == 404


@pytest.mark.parametrize("scan_id", ["..%2F..%2Fetc", "a.b", "a%00b"])
def test_scan_ids_cannot_leave_the_scan_folder(scan_id):
    assert client.get(f"/api/scans/{scan_id}").status_code in (400, 404)


def test_scene_id_cannot_walk_the_catalog():
    body = {"bbox": [5.86, 43.06, 5.98, 43.13], "scene_id": "../../search"}
    assert client.post("/api/scans", json=body).status_code == 422


def test_oversized_upload_is_refused_before_decoding(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_PX", 10)
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, "PNG")
    r = client.post("/api/detect", files={"image": ("a.png", buf.getvalue(), "image/png")})
    assert r.status_code == 413


def test_the_demo_scan_ships_with_the_repo():
    assert (cli.EXAMPLE / "scan.json").exists() and (cli.EXAMPLE / "ais.json").exists()
