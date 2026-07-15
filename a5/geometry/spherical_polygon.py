# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

from typing import List, Tuple, Union, cast
import math
from ..core.coordinate_systems import Cartesian
from ..math import vec3

# Type aliases for clarity
SphericalPolygon = List[Cartesian]

_winding_centroid = vec3.create()
_center = vec3.create()

# Use Cartesian system for all calculations for greater accuracy
# Using [x, y, z] gives equal precision in all directions, unlike spherical coordinates
UP = (0.0, 0.0, 1.0)


def spherical_triangle_area(v1: Cartesian, v2: Cartesian, v3: Cartesian) -> float:
    """
    Signed area (spherical excess) of the spherical triangle (v1, v2, v3) on the
    unit sphere, in radians.

    Uses the Van Oosterom–Strackee formula.
    atan2 keeps full precision for tiny triangles (numerator -> area/2) and
    does not fold areas above pi back into [-pi, pi].
    Free-function form avoids the class allocation of
    `SphericalTriangleShape([…]).get_area()` on the lon_lat_to_cell hot path.
    """
    norm = (
        1
        + (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
        + (v2[0] * v3[0] + v2[1] * v3[1] + v2[2] * v3[2])
        + (v3[0] * v1[0] + v3[1] * v1[1] + v3[2] * v1[2])
    )
    return 2 * math.atan2(vec3.tripleProduct(v1, v2, v3), norm)


def point_in_spherical_polygon(point: Cartesian, vertices: List[Cartesian]) -> bool:
    """
    Spherical point-in-polygon via signed-angle summation. Works for concave
    polygons (unlike `SphericalPolygonShape.contains_point`, which assumes
    convex "necessary strike"). The math is fully inlined as it's called
    per-cell in polygon-fill hot paths.
    """
    angle_sum = 0.0
    n = len(vertices)
    for i in range(n):
        av = vertices[i]
        bv = vertices[(i + 1) % n]
        dot_pa = point[0] * av[0] + point[1] * av[1] + point[2] * av[2]
        dot_pb = point[0] * bv[0] + point[1] * bv[1] + point[2] * bv[2]
        apx = av[0] - dot_pa * point[0]
        apy = av[1] - dot_pa * point[1]
        apz = av[2] - dot_pa * point[2]
        bpx = bv[0] - dot_pb * point[0]
        bpy = bv[1] - dot_pb * point[1]
        bpz = bv[2] - dot_pb * point[2]
        cx = apy * bpz - apz * bpy
        cy = apz * bpx - apx * bpz
        cz = apx * bpy - apy * bpx
        angle_sum += math.atan2(
            cx * point[0] + cy * point[1] + cz * point[2],
            apx * bpx + apy * bpy + apz * bpz,
        )
    return abs(angle_sum) > math.pi


def ring_winding_sign(ring_vecs: List[Cartesian]) -> int:
    """
    Ring winding direction: +1 for CCW (interior to the left of edge direction),
    -1 for CW. Sums (v_i x v_{i+1}) . centroid across the ring.
    """
    vec3.set(_winding_centroid, 0.0, 0.0, 0.0)
    for v in ring_vecs:
        vec3.add(_winding_centroid, _winding_centroid, v)
    vec3.normalize(_winding_centroid, _winding_centroid)

    n = len(ring_vecs)
    s = 0.0
    for i in range(n):
        s += vec3.tripleProduct(_winding_centroid, ring_vecs[i], ring_vecs[(i + 1) % n])
    return 1 if s > 0 else -1


def ring_segment_normals(ring_vecs: List[Cartesian]) -> List[Cartesian]:
    """Great-circle plane normals for every segment of the ring."""
    n = len(ring_vecs)
    normals: List[Cartesian] = []
    for i in range(n):
        out = vec3.create()
        vec3.cross(out, ring_vecs[i], ring_vecs[(i + 1) % n])
        normals.append(cast(Cartesian, (out[0], out[1], out[2])))
    return normals


class SphericalPolygonShape:
    def __init__(self, vertices: SphericalPolygon):
        # Store vertices as tuples for immutability and consistency
        self.vertices = tuple(tuple(v) if not isinstance(v, tuple) else v for v in vertices)
        # self._is_winding_correct()  # Debug check only, don't correct

    def get_boundary(self, n_segments: int = 1, closed_ring: bool = True) -> SphericalPolygon:
        """
        Returns a closed boundary of the polygon, with n_segments points per edge
        
        Args:
            n_segments: Number of points per edge
            closed_ring: Whether to close the ring by adding the first point at the end
            
        Returns:
            List of Cartesian coordinates forming the boundary
        """
        points: SphericalPolygon = []
        N = len(self.vertices)
        
        for s in range(N * n_segments):
            t = s / n_segments
            points.append(self.slerp(t))
            
        if closed_ring:
            points.append(points[0])
        
        return points

    def slerp(self, t: float) -> Cartesian:
        """
        Interpolates along boundary of polygon. Pass t = 1.5 to get the midpoint between 2nd and 3rd vertices
        
        Args:
            t: Parameter along the boundary (can be fractional)
            
        Returns:
            Cartesian coordinate
        """
        N = len(self.vertices)
        f = t % 1.0
        i = int(math.floor(t % N))
        j = (i + 1) % N
        
        out = vec3.create()
        return vec3.slerp(out, self.vertices[i], self.vertices[j], f)

    def get_transformed_vertices(self, t: float) -> Tuple[Cartesian, Cartesian, Cartesian]:
        """
        Returns the vertex given by index t, along with the vectors:
        - VA: Vector from vertex to point A
        - VB: Vector from vertex to point B
        
        Args:
            t: Parameter along the boundary
            
        Returns:
            Tuple of (V, VA, VB) where V is the vertex and VA, VB are vectors
        """
        N = len(self.vertices)
        i = int(math.floor(t % N))
        j = (i + 1) % N
        k = (i + N - 1) % N

        # Points A & B (vertex before and after)
        V = vec3.clone(self.vertices[i])
        VA = vec3.clone(self.vertices[j])
        VB = vec3.clone(self.vertices[k])
        vec3.sub(VA, VA, V)
        vec3.sub(VB, VB, V)
        
        return V, VA, VB

    def contains_point(self, point: Cartesian) -> float:
        """
        Determines if a point is inside the spherical polygon.
        Adaptation of algorithm from:
        'Locating a point on a spherical surface relative to a spherical polygon'
        Using only the condition of 'necessary strike'
        
        Args:
            point: Cartesian coordinate to test
            
        Returns:
            Positive value if inside, 0 if on edge, negative if outside
        """
        N = len(self.vertices)
        theta_delta_min = float('inf')

        for i in range(N):
            # Transform point and neighboring vertices into coordinate system centered on vertex
            V, VA, VB = self.get_transformed_vertices(i)
            VP = vec3.create()
            vec3.sub(VP, point, V)

            # Normalize to obtain unit direction vectors
            vec3.normalize(VP, VP)
            vec3.normalize(VA, VA)
            vec3.normalize(VB, VB)

            # Cross products will point away from the center of the sphere when
            # point P is within arc formed by VA and VB
            cross_ap = vec3.create()
            vec3.cross(cross_ap, VA, VP)
            cross_pb = vec3.create()
            vec3.cross(cross_pb, VP, VB)

            # Dot product will be positive when point P is within arc formed by VA and VB
            # The magnitude of the dot product is the sine of the angle between the two vectors
            # which is the same as the angle for small angles.
            sin_ap = vec3.dot(V, cross_ap)
            sin_pb = vec3.dot(V, cross_pb)

            # By returning the minimum value we find the arc where the point is closest to being outside
            theta_delta_min = min(theta_delta_min, sin_ap, sin_pb)

        return theta_delta_min

    def get_area(self) -> float:
        """
        Calculate the area of the spherical polygon by decomposing it into a fan of triangles
        
        Returns:
            The area of the spherical polygon in radians
        """
        if not hasattr(self, '_area') or self._area is None:
            self._area = self._get_area()
        return self._area

    def _get_area(self) -> float:
        """
        Internal method to calculate the area of the spherical polygon
        
        Returns:
            The area of the spherical polygon in radians
        """
        if len(self.vertices) < 3:
            return 0.0

        if len(self.vertices) == 3:
            self._area = spherical_triangle_area(self.vertices[0], self.vertices[1], self.vertices[2])
            return self._area

        # Calculate center of polygon
        vec3.set(_center, 0, 0, 0)
        for vertex in self.vertices:
            vec3.add(_center, _center, vertex)
        vec3.normalize(_center, _center)

        # Sum fan of triangles around center
        area = 0.0
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]
            tri_area = spherical_triangle_area(_center, v1, v2)
            if not math.isnan(tri_area):
                area += tri_area

        self._area = area
        return self._area

    def _is_winding_correct(self) -> bool:
        """Check if the polygon vertices are in the correct winding order"""
        if len(self.vertices) < 3:
            return True
        area = self.get_area()
        return area > 0 