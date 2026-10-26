import numpy as np
import pytest

from vigie.morphology import measure
from vigie.tiling import Box


def ship_image(length_px: int, angle_deg: float, size: int = 64):
    img = np.full((size, size, 3), 20, np.uint8)
    c = size / 2
    t = np.radians(angle_deg)
    # angle mesuré depuis le nord, sens horaire
    dx, dy = np.sin(t), -np.cos(t)
    for s in np.linspace(-length_px / 2, length_px / 2, length_px * 4):
        for o in (-1, 0, 1):
            x, y = c + s * dx + o * dy, c + s * dy - o * dx
            img[int(round(y)), int(round(x))] = 220
    return img


@pytest.mark.parametrize("angle", [0, 30, 45, 90, 135])
def test_measure_recovers_length_and_heading(angle):
    img = ship_image(20, angle)
    shape = measure(img, Box(10, 10, 54, 54, 0.9), gsd=10)
    assert shape is not None
    assert abs(shape.length_m - 200) <= 30
    diff = abs((shape.heading_deg - angle + 90) % 180 - 90)
    assert diff <= 8


def test_measure_empty_returns_none():
    img = np.full((40, 40, 3), 20, np.uint8)
    assert measure(img, Box(5, 5, 35, 35, 0.5), gsd=10) is None


def test_wake_gives_course_and_short_hull():
    img = np.full((64, 64, 3), 20, np.uint8)
    # coque brillante au nord, sillage plus pâle vers le sud : cap au nord
    for y in range(16, 22):
        img[y, 31:33] = 230
    for y in range(22, 50):
        img[y, 31:33] = 90
    shape = measure(img, Box(20, 10, 44, 54, 0.9), gsd=10)
    assert shape.length_m <= 90
    assert shape.wake_m and shape.wake_m >= 200
    assert shape.course_deg is not None and min(shape.course_deg, 360 - shape.course_deg) <= 10
