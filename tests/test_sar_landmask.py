import numpy as np
import pytest
import rasterio

from vigie import landmask


def test_water_from_classes_erodes_coast_and_keeps_open_sea():
    lc = np.full((40, 40), 80, np.uint8)
    lc[:, :10] = 50  # bâti
    lc[30:, 30:] = 0  # hors tuile WorldCover : le large
    w = landmask.water_from_classes(lc, margin=3)
    assert not w[:, :13].any()
    assert w[:, 14:].all()


def test_structures_mask_buffers_a_breakwater():
    transform = rasterio.Affine(10, 0, 500000, 0, -10, 6400000)
    # Digue est-ouest de 400 m, au milieu d'une grille de 100 x 100 px en UTM 32N.
    from pyproj import Transformer

    to_wgs = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    line = [to_wgs.transform(500300, 6399500), to_wgs.transform(500700, 6399500)]
    m = landmask.structures_mask([{"coords": line}], transform, "EPSG:32632", (100, 100), buffer_m=30)
    assert m[50, 30:70].all()
    assert m[47:53, 50].all()
    assert not m[40, 50] and not m[50, 20]
    bridge = landmask.structures_mask([{"coords": line, "bridge": True}], transform, "EPSG:32632", (100, 100),
                                      bridge_m=100)
    assert bridge[42, 50] and bridge[58, 50] and not bridge[35, 50] and not bridge[65, 50]


def test_worldcover_skagen_has_land_and_sea():
    from vigie import sar

    bbox = (10.58, 57.70, 10.62, 57.72)
    crs = sar.utm_crs(bbox)
    transform, w, h = sar.grid(bbox, crs)
    try:
        lc = landmask.classes(bbox, transform, crs, (h, w))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"réseau indisponible : {exc}")
    assert (lc == 80).mean() > 0.2
    assert ((lc != 80) & (lc != 0)).mean() > 0.05
