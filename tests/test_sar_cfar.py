import numpy as np
import pytest

from vigie import cfar


def sea(size: int = 200, seed: int = 0, mean_db: float = -18.0) -> np.ndarray:
    """Fouillis de mer : speckle gamma à 4 vues, en intensité linéaire."""
    rng = np.random.default_rng(seed)
    return (10 ** (mean_db / 10) * rng.gamma(4.0, 1 / 4.0, (size, size))).astype(np.float32)


def add_ship(img: np.ndarray, x: int, y: int, length_px: int, angle_deg: float, db: float = 10.0) -> None:
    """Coque fine d'intensité `db`, angle depuis le nord, sens horaire."""
    t = np.radians(angle_deg)
    dx, dy = np.sin(t), -np.cos(t)
    for s in np.linspace(-length_px / 2, length_px / 2, length_px * 4):
        for o in (-0.5, 0, 0.5):
            img[round(y + s * dy - o * dx), round(x + s * dx + o * dy)] = 10 ** (db / 10)


def test_empty_sea_has_no_false_alarm():
    res = cfar.detect(sea(seed=1), None, None, gsd=10)
    assert res.targets == []


@pytest.mark.parametrize("angle", [0, 45, 90, 135])
def test_finds_ship_and_measures_it(angle):
    img = sea(seed=2)
    add_ship(img, 100, 100, 20, angle)
    res = cfar.detect(img, None, None, gsd=10)
    assert len(res.targets) == 1
    t = res.targets[0]
    assert abs(t.lon_px - 100.5) < 2 and abs(t.lat_px - 100.5) < 2
    assert abs(t.length_m - 200) <= 30
    assert abs((t.heading_deg - angle + 90) % 180 - 90) <= 8
    assert t.contrast_db > 20


def test_threshold_adapts_to_rough_sea():
    # Même navire, mer 8 dB plus claire : toujours vu, et rien d'autre.
    img = sea(seed=3, mean_db=-10.0)
    add_ship(img, 60, 60, 15, 30, db=12.0)
    res = cfar.detect(img, None, None, gsd=10)
    assert len(res.targets) == 1


def test_land_mask_silences_land():
    img = sea(seed=4)
    rng = np.random.default_rng(5)
    img[:, :100] = 10 ** (-8 / 10) * rng.gamma(1.0, 1.0, (200, 100))  # terre
    ys, xs = rng.integers(0, 200, 60), rng.integers(0, 95, 60)
    for y, x in zip(ys, xs):
        img[y : y + 2, x : x + 3] = 10 ** (25 / 10)  # bâtiments, pylônes : des réflecteurs forts
    water = np.zeros(img.shape, bool)
    water[:, 100:] = True
    assert cfar.detect(img, None, water, gsd=10).targets == []
    assert len(cfar.detect(img, None, None, gsd=10).targets) > 0


def test_vh_finds_ship_hidden_in_vv():
    vv = sea(seed=6, mean_db=-8.0)
    vh = sea(seed=7, mean_db=-26.0)
    add_ship(vh, 100, 100, 10, 0, db=-5.0)
    assert cfar.detect(vv, None, None, gsd=10).targets == []
    res = cfar.detect(vv, vh, None, gsd=10)
    assert len(res.targets) == 1


def test_sidelobe_cross_does_not_stretch_hull():
    img = sea(seed=8)
    add_ship(img, 100, 100, 15, 45, db=25.0)
    # Croix de lobes secondaires le long des axes de l'image, 25 dB sous le pic.
    img[100, 40:160] = np.maximum(img[100, 40:160], 10 ** 0.0)
    img[40:160, 100] = np.maximum(img[40:160, 100], 10 ** 0.0)
    res = cfar.detect(img, None, None, gsd=10)
    assert len(res.targets) == 1
    assert res.targets[0].length_m <= 200


def test_nan_outside_swath_is_ignored():
    img = sea(seed=9)
    img[:, :50] = np.nan
    add_ship(img, 120, 100, 12, 90)
    res = cfar.detect(img, None, None, gsd=10)
    assert len(res.targets) == 1


def test_confidence_is_monotonic():
    assert cfar.confidence(13) == pytest.approx(0.5)
    assert cfar.confidence(10) < cfar.confidence(20) < cfar.confidence(35) < 1
