import json
import pytest

from a5.core.dodecahedron import project_dodecahedron, unproject_dodecahedron
from a5.core.origin import origins 
from pathlib import Path

TEST_COORDS_PATH = Path(__file__).parent / "test-polar-coordinates.json"
with open(TEST_COORDS_PATH) as f:
    TEST_COORDS = json.load(f)

def test_dodecahedron_round_trip():
    for origin in origins:
        for i, coord in enumerate(TEST_COORDS):
            polar = [coord["rho"], coord["beta"]]
            spherical = project_dodecahedron(polar, origin.quat, origin.angle)
            result = unproject_dodecahedron(spherical, origin.quat, origin.angle)
            assert result[0] == pytest.approx(polar[0], abs=1e-6), f"rho mismatch at index {i}"
            assert result[1] == pytest.approx(polar[1], abs=1e-6), f"beta mismatch at index {i}"
