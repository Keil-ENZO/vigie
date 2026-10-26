from vigie.tiling import Box, make_tiles, merge, tile_starts


def test_tile_starts_covers_edges():
    assert tile_starts(500, 640, 96) == [0]
    starts = tile_starts(1500, 640, 96)
    assert starts[0] == 0 and starts[-1] == 1500 - 640
    assert all(b - a <= 640 - 96 for a, b in zip(starts, starts[1:]))


def test_make_tiles_cover_whole_image():
    tiles = make_tiles(1500, 900, 640, 96)
    def inside(x, y):
        return any(t.x <= x < t.x + t.width and t.y <= y < t.y + t.height for t in tiles)

    assert all(inside(x, y) for x in range(0, 1500, 7) for y in range(0, 900, 7))
    assert inside(1499, 899)


def test_merge_drops_duplicates_and_split_boxes():
    a = Box(0, 0, 10, 10, 0.9)
    dup = Box(1, 1, 11, 11, 0.5)
    inside = Box(2, 2, 6, 6, 0.4)  # moitié de navire vue par la tuile voisine
    other = Box(50, 50, 60, 60, 0.3)
    kept = merge([dup, inside, a, other])
    assert kept == [a, other]


def test_merge_keeps_other_classes():
    a, b = Box(0, 0, 10, 10, 0.9, class_id=0), Box(0, 0, 10, 10, 0.8, class_id=1)
    assert len(merge([a, b])) == 2
