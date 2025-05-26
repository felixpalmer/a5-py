import pytest
from a5.core.utils import PentagonShape, Contour, Pentagon
from a5.core.coordinate_systems import Degrees, LonLat

# Create a simple pentagon for testing
@pytest.fixture
def pentagon() -> PentagonShape:
    return PentagonShape([
        (0, 2),     # top
        (2, 1),     # upper right
        (1, -2),    # lower right
        (-1, -2),   # lower left
        (-2, 1),    # upper left
    ])


# ---------- contains_point tests ----------
@pytest.mark.parametrize("point", [
    (0, 0),        # center
    (0, 1.5),      # upper triangle
    (1, 0),        # right triangle
    (-1, 0),       # left triangle
])
def test_contains_point_inside(pentagon: PentagonShape, point: LonLat):
    assert pentagon.contains_point(point) is True


@pytest.mark.parametrize("point", [
    (0, 3),        # above
    (3, 0),        # right
    (0, -3),       # below
    (-3, 0),       # left
    (0, 2.1),      # near top
    (2.1, 1),      # near right
])
def test_contains_point_outside(pentagon: PentagonShape, point: LonLat):
    assert pentagon.contains_point(point) is False


@pytest.mark.parametrize("point", [
    (0, 2),               # on top vertex
    (1.9999, 0.9999),     # near upper-right vertex
    (1, 1.49999),         # right edge
    (-1, 1.49999),        # left edge
])
def test_contains_point_on_edge_or_vertex(pentagon: PentagonShape, point: LonLat):
    assert pentagon.contains_point(point) is True


# ---------- normalize_longitudes tests ----------

def test_normalize_longitudes_no_wrap():
    contour: Contour = [
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10),
        (0, 0),
    ]
    normalized = PentagonShape.normalize_longitudes(contour)
    assert normalized == contour


@pytest.mark.skip(reason="Pending normalization for wrap-around case")
def test_normalize_longitudes_wrap_positive():
    contour: Contour = [
        (170, 0),
        (175, 0),
        (180, 0),
        (-175, 0),  # should become 185
        (-170, 0),  # should become 190
    ]
    normalized = PentagonShape.normalize_longitudes(contour)
    assert normalized[3][0] == pytest.approx(185)
    assert normalized[4][0] == pytest.approx(190)


def test_normalize_longitudes_wrap_negative():
    contour: Contour = [
        (-170, 0),
        (-175, 0),
        (-180, 0),
        (175, 0),   # should become -185
        (170, 0),   # should become -190
    ]
    normalized = PentagonShape.normalize_longitudes(contour)
    assert normalized[3][0] == pytest.approx(-185)
    assert normalized[4][0] == pytest.approx(-190)
