# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# Spherical polygon (with holes) prepared for repeated point-containment
# tests: bounding-cap prefilter, then a trig-free crossing-number test with
# the winding-number test as a robust fallback.

import math
from dataclasses import dataclass
from typing import List, Optional
from ..core.coordinate_systems import Cartesian
from .spherical_polygon import point_in_spherical_polygon, ring_segment_normals


def _point_in_polygon_rings(point: Cartesian, ring_vecs_list: List[List[Cartesian]]) -> bool:
    """
    Point-in-polygon for a polygon with holes: inside the outer ring and
    outside every hole ring. Winding-number test -- robust but O(atan2) per
    edge; used as the fallback for the crossing-number fast path below.
    """
    if not point_in_spherical_polygon(point, ring_vecs_list[0]):
        return False
    for ring_vecs in ring_vecs_list[1:]:
        if point_in_spherical_polygon(point, ring_vecs):
            return False
    return True


@dataclass
class BoundingCap:
    center: Cartesian
    min_dot: float


def _bounding_cap(ring_vecs_list: List[List[Cartesian]]) -> BoundingCap:
    """
    Bounding cap of the polygon: every polygon point is within the cap.
    The winding-number PIP is blind at the polygon's ANTIPODE (the angle sum is
    +/-2pi there too), so distant probes MUST be rejected by the cap first. The cap
    angle is bounded by the farthest ring vertex plus half the longest edge (any
    point of an edge arc is within half the edge length of an endpoint).
    """
    cx = cy = cz = 0.0
    for v in ring_vecs_list[0]:
        cx += v[0]
        cy += v[1]
        cz += v[2]
    length = math.sqrt(cx * cx + cy * cy + cz * cz)
    if length < 1e-12:
        return BoundingCap(center=(0.0, 0.0, 1.0), min_dot=-1.0)
    cx /= length
    cy /= length
    cz /= length
    center = (cx, cy, cz)

    max_angle = 0.0
    max_edge = 0.0
    for ring_vecs in ring_vecs_list:
        n = len(ring_vecs)
        for i in range(n):
            v = ring_vecs[i]
            w = ring_vecs[(i + 1) % n]
            dot_cv = cx * v[0] + cy * v[1] + cz * v[2]
            max_angle = max(max_angle, math.acos(min(1.0, max(-1.0, dot_cv))))
            dot_vw = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
            max_edge = max(max_edge, math.acos(min(1.0, max(-1.0, dot_vw))))
    cap_angle = min(math.pi, max_angle + max_edge / 2)
    return BoundingCap(center=center, min_dot=math.cos(cap_angle))


@dataclass
class PreparedPolygon:
    """
    Polygon prepared for repeated containment tests: rings, per-edge great-circle
    normals, bounding cap, and a reference point for the crossing-number test.

    The reference point sits just OUTSIDE the cap (angle capAngle + 0.2 from its
    center) rather than at the antipode: probes come from inside the cap, so
    the probe->ref arc plane stays well conditioned (|p x ref| >= sin 0.2). The
    fast path is disabled for very large polygons (cap over ~79 deg), where that
    construction can't keep the arc short -- those fall back to the winding test.
    """
    ring_vecs_list: List[List[Cartesian]]
    ring_normals: List[List[Cartesian]]
    cap: BoundingCap
    ref: Cartesian
    use_fast: bool


def prepare_polygon(ring_vecs_list: List[List[Cartesian]]) -> PreparedPolygon:
    cap = _bounding_cap(ring_vecs_list)
    ring_normals = [ring_segment_normals(ring) for ring in ring_vecs_list]
    cap_angle = math.acos(min(1.0, max(-1.0, cap.min_dot)))
    use_fast = cap.min_dot > -1.0 and cap_angle < 1.37
    c = cap.center

    # perp = c x (Z_AXIS or X_AXIS), unit vector perpendicular to the cap center
    axis = (0.0, 0.0, 1.0) if abs(c[2]) < 0.9 else (1.0, 0.0, 0.0)
    perp = (
        c[1] * axis[2] - c[2] * axis[1],
        c[2] * axis[0] - c[0] * axis[2],
        c[0] * axis[1] - c[1] * axis[0],
    )
    d_len = math.sqrt(perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]) or 1.0
    theta = cap_angle + 0.2
    cos_t = math.cos(theta)
    sin_t = math.sin(theta) / d_len
    ref = (
        c[0] * cos_t + perp[0] * sin_t,
        c[1] * cos_t + perp[1] * sin_t,
        c[2] * cos_t + perp[2] * sin_t,
    )
    return PreparedPolygon(
        ring_vecs_list=ring_vecs_list,
        ring_normals=ring_normals,
        cap=cap,
        ref=ref,
        use_fast=use_fast,
    )


_CROSSING_EPS = 1e-14


def _crossing_parity(p: Cartesian, prep: PreparedPolygon) -> Optional[bool]:
    """
    Crossing-number containment: count proper crossings of the arc probe->ref
    with every ring edge (just sign tests -- no trig); odd parity = inside
    (`ref` is outside the polygon, and the even-odd rule handles holes for
    free). Returns None on any near-degenerate sign (probe or a vertex on
    an arc plane) -- the caller falls back to the winding test, which also keeps
    on-edge tie-breaking identical to the previous implementation.
    """
    r = prep.ref
    # normal of the probe->ref arc plane
    abx = p[1] * r[2] - p[2] * r[1]
    aby = p[2] * r[0] - p[0] * r[2]
    abz = p[0] * r[1] - p[1] * r[0]
    crossings = 0
    for ri in range(len(prep.ring_vecs_list)):
        verts = prep.ring_vecs_list[ri]
        norms = prep.ring_normals[ri]
        n = len(verts)
        s_first = abx * verts[0][0] + aby * verts[0][1] + abz * verts[0][2]
        if abs(s_first) < _CROSSING_EPS:
            return None
        s_prev = s_first
        for i in range(n):
            if i + 1 == n:
                s_next = s_first
            else:
                v = verts[i + 1]
                s_next = abx * v[0] + aby * v[1] + abz * v[2]
                if abs(s_next) < _CROSSING_EPS:
                    return None
            if s_prev * s_next < 0:
                # edge endpoints straddle the probe arc's plane: test whether the
                # probe arc straddles the edge's plane on the matching side
                cd = norms[i]
                cbd = -(cd[0] * r[0] + cd[1] * r[1] + cd[2] * r[2])
                dac = cd[0] * p[0] + cd[1] * p[1] + cd[2] * p[2]
                if abs(cbd) < _CROSSING_EPS or abs(dac) < _CROSSING_EPS:
                    return None
                acb = -s_prev
                if acb * cbd > 0 and acb * dac > 0:
                    crossings += 1
            s_prev = s_next
    return (crossings & 1) == 1


def point_in_prepared_polygon(p: Cartesian, prep: PreparedPolygon) -> bool:
    """Full containment test of a point: cap prefilter, then crossing test with winding fallback."""
    cap = prep.cap
    if p[0] * cap.center[0] + p[1] * cap.center[1] + p[2] * cap.center[2] < cap.min_dot:
        return False
    if prep.use_fast:
        result = _crossing_parity(p, prep)
        if result is not None:
            return result
    return _point_in_polygon_rings(p, prep.ring_vecs_list)
