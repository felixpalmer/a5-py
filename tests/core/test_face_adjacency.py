import math

from a5.core.constants import TWO_PI_OVER_5
from a5.core.face_adjacency import FACE_ADJACENCY, seam_transform, seam_triple
from a5.core.tiling import get_pentagon_center
from a5.lattice import Triple, triple_flavor, triple_in_bounds
from a5.projections.dodecahedron import DodecahedronProjection

dodecahedron = DodecahedronProjection()

# Equivalent to vitest's toBeCloseTo(x, 10)
TOLERANCE = 5e-11


def apply(m, p):
    x, y = p[0], p[1]
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


class TestFaceSeam:
    def test_seam_transform_agrees_with_projection(self):
        """seam_transform agrees with the projection across every base edge."""
        for o in range(12):
            for q in range(5):
                adjacent_id, adjacent_quintant = FACE_ADJACENCY[o][q]
                m = seam_transform(o, q)
                # Points of the neighbor's quintant near the shared edge, projected into this face
                for r, angle in ((0.5, 0.0), (0.55, -0.4), (0.55, 0.4)):
                    gamma = adjacent_quintant * TWO_PI_OVER_5 + angle
                    point = (r * math.cos(gamma), r * math.sin(gamma))
                    landed = dodecahedron.forward(dodecahedron.inverse(point, adjacent_id), o)
                    mapped = apply(m, landed)
                    assert abs(mapped[0] - point[0]) < TOLERANCE
                    assert abs(mapped[1] - point[1]) < TOLERANCE

    def test_seam_triple_is_seam_transform_on_cells(self):
        """seam_triple is seam_transform on cells."""
        for hilbert_res in (1, 2, 5):
            max_row = (1 << hilbert_res) - 1
            for q in range(5):
                m = seam_transform(0, q)
                adjacent_quintant = FACE_ADJACENCY[0][q][1]
                # The cells of the two rows along the base edge
                for y in range(max(0, max_row - 1), max_row + 1):
                    for x in range(-y - 1, 1):
                        for parity in (0, 1):
                            triple = Triple(x, y, parity - x - y)
                            if not triple_in_bounds(triple, max_row):
                                continue
                            flavor = triple_flavor(triple, max_row)
                            image = seam_triple(triple, max_row)
                            assert triple_flavor(image, max_row) == flavor ^ 1
                            center = apply(m, get_pentagon_center(hilbert_res, q, triple, flavor))
                            image_center = get_pentagon_center(hilbert_res, adjacent_quintant, image, flavor ^ 1)
                            assert abs(center[0] - image_center[0]) < TOLERANCE
                            assert abs(center[1] - image_center[1]) < TOLERANCE
