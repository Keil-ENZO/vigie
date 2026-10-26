import time

from vigie.aislive import Collector, ship_category


def test_ship_category():
    assert ship_category(70) == "cargo" and ship_category(84) == "tanker"
    assert ship_category(30) == "fishing" and ship_category(52) == "service"
    assert ship_category(35) == "military" and ship_category(60) == "passenger" and ship_category(None) == "unknown"


def test_ingest_position_then_static():
    c = Collector()
    c._ingest({"MessageType": "PositionReport",
               "MetaData": {"MMSI": 219000001, "ShipName": "TEST  ", "latitude": 57.6, "longitude": 10.6},
               "Message": {"PositionReport": {"Sog": 11.2, "Cog": 45.0, "TrueHeading": 511, "NavigationalStatus": 0}}})
    c._ingest({"MessageType": "ShipStaticData", "MetaData": {"MMSI": 219000001},
               "Message": {"ShipStaticData": {"Type": 80, "Dimension": {"A": 150, "B": 30, "C": 15, "D": 15},
                                              "Destination": "SKAGEN", "Name": "TEST"}}})
    v = c.snapshot((10.5, 57.5, 10.7, 57.7))
    assert len(v) == 1 and v[0]["category"] == "tanker" and v[0]["length_m"] == 180
    assert v[0]["heading_deg"] is None and v[0]["sog_kn"] == 11.2 and v[0]["name"] == "TEST"
    assert c.snapshot((0, 0, 1, 1)) == []


def test_old_vessels_expire():
    c = Collector()
    c.vessels["1"] = {"mmsi": "1", "lat": 1, "lon": 1, "seen": time.time() - 4000}
    assert c.snapshot((0, 0, 2, 2)) == [] and not c.vessels


def test_watch_keeps_few_boxes():
    c = Collector()
    c._ensure_running = lambda: None
    for i in range(6):
        c.watch((i, 0, i + 1, 1))
    assert len(c.boxes) == 4
