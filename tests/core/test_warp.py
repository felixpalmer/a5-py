import json
import pytest
from a5.core.warp import normalize_gamma, warp_polar, unwarp_polar, warp_beta, unwarp_beta
from a5.core.constants import PI_OVER_5, PI_OVER_10, TWO_PI_OVER_5, distance_to_edge
from pathlib import Path

# Load test coordinates from JSON file
TEST_COORDS_PATH = Path(__file__).parent / "test-polar-coordinates.json"
with open(TEST_COORDS_PATH) as f:
    TEST_COORDS = json.load(f)

@pytest.mark.parametrize("gamma, expected", [
    (0.1, 0.1),
    (0.2, 0.2),
    (-0.2, -0.2),
    (1.2, 1.2 - TWO_PI_OVER_5)
])
def test_normalize_gamma(gamma, expected):
    assert normalize_gamma(gamma) == pytest.approx(expected,abs=1e-4)

def test_normalize_gamma_periodicity():
    gamma1 = PI_OVER_5
    gamma2 = gamma1 + 2 * PI_OVER_5
    assert normalize_gamma(gamma1) == pytest.approx(normalize_gamma(gamma2),abs=1e-4)

@pytest.mark.parametrize("input_val, expected", [
    ([0, 0], [0, 0]),
    ([1, 0], [1.2988, 0]),
    ([1, PI_OVER_5], [1.1723, PI_OVER_5]),
    ([1, -PI_OVER_5], [1.1723, -PI_OVER_5]),
    ([0.2, 0.0321], [0.1787, 0.03097]),
    ([0.789, -0.555], [0.8128, -0.55057]),
])
def test_warp_polar(input_val, expected):
    result = warp_polar(input_val)
    assert result[0] == pytest.approx(expected[0], abs=1e-4)
    assert result[1] == pytest.approx(expected[1], abs=1e-4)

def test_warp_polar_edge_distance():
    result = warp_polar((distance_to_edge, 0))
    assert result[0] == pytest.approx(distance_to_edge, abs=1e-4)

@pytest.mark.parametrize("input_val, warped", [
    ([0, 0], [0, 0]),
    ([1, 0], [1.2988, 0]),
    ([1, PI_OVER_5], [1.1723, PI_OVER_5]),
    ([1, -PI_OVER_5], [1.1723, -PI_OVER_5]),
    ([0.2, 0.0321], [0.1787, 0.03097]),
    ([0.789, -0.555], [0.8128, -0.55057]),
])
def test_unwarp_polar(input_val, warped):
    result = unwarp_polar(warped)
    assert result[0] == pytest.approx(input_val[0], abs=1e-4)
    assert result[1] == pytest.approx(input_val[1], abs=1e-4)

def test_warp_unwarp_round_trip_polar():
    original = [1, PI_OVER_5]
    warped = warp_polar(original)
    unwarped = unwarp_polar(warped)
    assert unwarped[0] == pytest.approx(original[0], abs=1e-5)
    assert unwarped[1] == pytest.approx(original[1], abs=1e-5)

@pytest.mark.parametrize("input_val, expected", [
    (0, 0),
    (0.1, 0.09657),
    (-0.2, -0.19366),
    (PI_OVER_10, 0.30579),
    (PI_OVER_5, PI_OVER_5),
])
def test_warp_beta(input_val, expected):
    assert warp_beta(input_val) == pytest.approx(expected, abs=1e-4)

def test_warp_beta_symmetric():
    beta = PI_OVER_5
    assert warp_beta(beta) == pytest.approx(-warp_beta(-beta), abs=1e-4)

def test_warp_beta_preserves_zero():
    assert warp_beta(0) == 0

@pytest.mark.parametrize("input_val, expected", [
    (0, 0),
    (0.1, 0.09657),
    (-0.2, -0.19366),
    (PI_OVER_10, 0.30579),
    (PI_OVER_5, PI_OVER_5),
])
def test_unwarp_beta(input_val, expected):
    assert unwarp_beta(expected) == pytest.approx(input_val, abs=1e-4)

def test_unwarp_beta_round_trip():
    beta = 0.3
    warped = warp_beta(beta)
    unwarped = unwarp_beta(warped)
    assert unwarped == pytest.approx(beta, abs=1e-4)

def test_unwarp_beta_symmetric():
    beta = 0.2
    assert unwarp_beta(beta) == pytest.approx(-unwarp_beta(-beta), abs=1e-4)

def test_unwarp_beta_preserves_zero():
    assert unwarp_beta(0) == 0

def test_polar_coordinates_round_trip():
    for coord in TEST_COORDS:
        polar = [coord["rho"], coord["beta"]]
        warped = warp_polar(polar)
        unwarped = unwarp_polar(warped)
        assert unwarped[0] == pytest.approx(polar[0], abs=1e-4)
        assert unwarped[1] == pytest.approx(polar[1], abs=1e-4)
