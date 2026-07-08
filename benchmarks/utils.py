# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

"""Shared benchmark utilities.

Deterministic PRNG (mulberry32) so every run benchmarks identical inputs.
Plain Python ints with & 0xFFFFFFFF masking emulate uint32; the exact numeric
values need not match the TypeScript suite, only be deterministic within Python.
"""

import json
import math
import os
from typing import Callable, List, Tuple

from a5 import lonlat_to_cell

_FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'tests', 'regions', 'fixtures', 'polygon.json',
)


def load_countries() -> list:
    """Load the country polygon fixtures used by compact/polygon benchmarks."""
    with open(_FIXTURE_PATH) as f:
        return json.load(f)['country']


def country_polygon(name: str) -> list:
    """Return the list-of-rings polygon for a country by name."""
    for c in load_countries():
        if c['name'] == name:
            return c['polygon']
    raise KeyError(f'country {name!r} not found in fixture')


def _imul(a: int, b: int) -> int:
    """Emulate JS Math.imul: low 32 bits of a 32-bit integer multiply."""
    return (a * b) & 0xFFFFFFFF


def create_random(seed: int = 42) -> Callable[[], float]:
    """Deterministic PRNG (mulberry32) returning floats in [0, 1)."""
    a = seed & 0xFFFFFFFF

    def rng() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t ^ (t + _imul(t ^ (t >> 7), t | 61))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rng


def sample_points(n: int, seed: int = 42) -> List[Tuple[float, float]]:
    """Points distributed uniformly over the sphere (area-uniform in latitude)."""
    random = create_random(seed)
    points: List[Tuple[float, float]] = []
    for _ in range(n):
        lon = 360 * random() - 180
        lat = math.asin(2 * random() - 1) * 180 / math.pi
        points.append((lon, lat))
    return points


def sample_cells(resolution: int, n: int, seed: int = 42) -> List[int]:
    """Cell IDs of uniformly distributed points at the given resolution."""
    points = sample_points(n, seed)
    return [lonlat_to_cell(points[i], resolution) for i in range(n)]
