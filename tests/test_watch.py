from types import SimpleNamespace

import pytest

from vigie import ais, scan, sentinel, watch

BOX = (10.5, 57.55, 10.8, 57.72)
META = {"id": "s1", "scene": {"datetime": "2026-10-18T10:45:00Z"}, "counts": {"vessels": 2}}
DET = {"features": [
    {"properties": {"id": "V001", "length_m": 120.0, "lat": 57.6, "lon": 10.6, "ais": {"status": "dark"}}},
    {"properties": {"id": "V002", "length_m": 90.0, "lat": 57.7, "lon": 10.7, "ais": {"status": "matched"}}},
]}


@pytest.fixture
def world(tmp_path, monkeypatch):
    calls = SimpleNamespace(scans=[], sent=[])
    monkeypatch.setattr(watch, "STATE", tmp_path / "watch.json")
    monkeypatch.setattr(sentinel, "search", lambda *a, **k: [
        SimpleNamespace(id="old", datetime="2026-10-10T10:45:00Z", coverage=1.0),
        SimpleNamespace(id="new", datetime="2026-10-18T10:45:00Z", coverage=0.95),
        SimpleNamespace(id="partial", datetime="2026-10-20T10:45:00Z", coverage=0.2)])
    monkeypatch.setattr(scan, "run", lambda bbox, scene, name, log: calls.scans.append(scene.id) or {"id": "s1"})
    monkeypatch.setattr(scan, "load_scan", lambda _: (META, DET))
    monkeypatch.setattr(ais, "crosscheck", lambda *a, **k: {})
    monkeypatch.setattr(watch, "notify", lambda url, title, message: calls.sent.append((url, message)))
    return calls


def test_a_new_pass_is_scanned_and_dark_vessels_notified_once(world):
    res = watch.check(BOX, name="Skagen", ntfy="http://n/t", log=lambda m: None)
    assert world.scans == ["new"] and res["dark"] == 1
    assert "1 sans AIS sur 2" in world.sent[0][1] and "V001" in world.sent[0][1] and "V002" not in world.sent[0][1]
    assert watch.check(BOX, ntfy="http://n/t", log=lambda m: None) is None
    assert len(world.scans) == 1 and len(world.sent) == 1


def test_missing_ais_keeps_the_pass_pending_without_a_new_scan(world, monkeypatch):
    def not_yet(*a, **k):
        raise LookupError("pas encore publié")

    monkeypatch.setattr(ais, "crosscheck", not_yet)
    assert watch.check(BOX, ntfy="http://n/t", log=lambda m: None) is None
    assert world.sent == []
    monkeypatch.setattr(ais, "crosscheck", lambda *a, **k: {})
    assert watch.check(BOX, ntfy="http://n/t", log=lambda m: None)["dark"] == 1
    assert world.scans == ["new"] and len(world.sent) == 1


def test_no_dark_vessel_stays_silent(world, monkeypatch):
    clean = {"features": [DET["features"][1]]}
    monkeypatch.setattr(scan, "load_scan", lambda _: (META, clean))
    assert watch.check(BOX, ntfy="http://n/t", log=lambda m: None)["dark"] == 0
    assert world.sent == []


def test_outside_ais_coverage_reports_the_count(world):
    res = watch.check((5.86, 43.06, 5.98, 43.13), ntfy="http://n/t", log=lambda m: None)
    assert res["dark"] is None and "2 navires" in world.sent[0][1]
