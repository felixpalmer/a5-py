# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import json
from pathlib import Path

from a5.lattice import Triple
from a5.lattice.lsystem import (
    CurveNode, curve_child, s_to_cell, s_to_triple, triple_to_curve_node, triple_to_s_lattice,
)
from a5.lattice.curve import round_to_triple


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "lsystem.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


class TestLSystem:
    """The non-self-intersecting L-system curve (planned FUTURE canonical curve)."""

    def test_s_to_cell(self):
        for f in load_fixtures()["sToCell"]:
            cell = s_to_cell(f["s"], f["resolution"], f["orientation"])
            assert cell.triple.x == f["x"]
            assert cell.triple.y == f["y"]
            assert cell.triple.z == f["z"]
            assert cell.flavor == f["flavor"]

    def test_s_to_triple(self):
        for f in load_fixtures()["sToCell"]:
            triple = s_to_triple(f["s"], f["resolution"], f["orientation"])
            assert triple == Triple(f["x"], f["y"], f["z"])

    def test_triple_to_s_lattice(self):
        for f in load_fixtures()["sToCell"]:
            triple = Triple(f["x"], f["y"], f["z"])
            s = triple_to_s_lattice(triple, f["resolution"], f["orientation"])
            assert s == f["s"]

    def test_point_to_s(self):
        for f in load_fixtures()["pointToS"]:
            s = triple_to_s_lattice(round_to_triple((f["i"], f["j"]), f["resolution"]), f["resolution"], f["orientation"])
            assert s == f["s"]


class TestCurveChild:
    """curve_child / triple_to_curve_node: the hierarchy in curve order, one digit per step."""

    ORIENTATIONS = ['uv', 'vu', 'uw', 'wu', 'vw', 'wv']

    @staticmethod
    def _root(orientation):
        return triple_to_curve_node(Triple(0, 0, 0), 0, orientation).node

    @staticmethod
    def _step(node, s, digit, resolution, orientation):
        """Step to the child with `digit`, checking it against s_to_cell, and the
        descent state against triple_to_curve_node's from the child's triple."""
        below = CurveNode()
        triple, flavor = curve_child(node, digit, resolution, orientation, below)
        child_s = s * 4 + digit
        assert (triple, flavor) == tuple(s_to_cell(child_s, resolution, orientation))
        assert tuple(triple_to_curve_node(triple, resolution, orientation)) == (child_s, flavor, below)
        return below, child_s

    def test_steps_agree_with_s_to_cell_and_triple_to_curve_node(self):
        for orientation in self.ORIENTATIONS:
            # Every cell through level 3, then one deep path to level 29 (A5 resolution 30)
            stack = [(self._root(orientation), 0, 0)]
            while stack:
                node, s, resolution = stack.pop()
                if resolution == 3:
                    continue
                for digit in range(4):
                    child_node, child_s = self._step(node, s, digit, resolution + 1, orientation)
                    stack.append((child_node, child_s, resolution + 1))
            node, s = self._root(orientation), 0
            for resolution in range(1, 30):
                digit = (resolution * 7 + ord(orientation[0])) % 4
                node, s = self._step(node, s, digit, resolution, orientation)
