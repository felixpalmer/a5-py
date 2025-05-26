import json
import numpy as np
import pytest

from a5.core.dodecahedron import project_dodecahedron, unproject_dodecahedron
from a5.core.origin import origins 
from a5.core.coordinate_systems import Polar
from pathlib import Path

TEST_COORDS_PATH = Path(__file__).parent / "test-polar-coordinates.json"
with open(TEST_COORDS_PATH) as f:
    TEST_COORDS = json.load(f)

@pytest.mark.parametrize("origin", origins)
def test_dodecahedron_round_trip(origin):
    for i, coord in enumerate(TEST_COORDS):
        polar = np.array([coord["rho"], coord["beta"]])
        spherical = project_dodecahedron(polar, origin.quat, origin.angle)
        result = unproject_dodecahedron(spherical, origin.quat, origin.angle)
        assert np.isclose(result[0], polar[0], atol=1e-6)
        assert np.isclose(result[1], polar[1], atol=1e-6)
