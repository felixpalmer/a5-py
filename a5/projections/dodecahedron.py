# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple, Union, cast, Literal
from ..core.coordinate_systems import Radians, Spherical, Cartesian, Polar, Face, FaceTriangle, SphericalTriangle
from ..core.coordinate_transforms import to_cartesian, to_spherical, to_face, to_polar
from ..core.constants import distance_to_edge, interhedral_angle, PI_OVER_5, TWO_PI_OVER_5
from ..core.origin import origins
from ..core.tiling import get_quintant_vertices
from .gnomonic import GnomonicProjection
from .polyhedral import PolyhedralProjection
from .crs import CRS
from ..math import vec2, vec3

# Type definitions
FaceTriangleIndex = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
OriginId = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Create global CRS instance
crs = CRS()


class DodecahedronProjection:
    """
    Dodecahedron projection for mapping between spherical and face coordinates
    """
    
    def __init__(self):
        self.face_triangles: List[FaceTriangle] = []
        self.spherical_triangles: List[SphericalTriangle] = []
        self.polyhedral = PolyhedralProjection()
        self.gnomonic = GnomonicProjection()

    def forward(self, spherical: Spherical, origin_id: OriginId) -> Face:
        """
        Projects spherical coordinates to face coordinates using dodecahedron projection
        
        Args:
            spherical: Spherical coordinates [theta, phi]
            origin_id: Origin ID (0-11)
            
        Returns:
            Face coordinates [x, y]
        """
        origin = origins[origin_id]

        # Transform back to origin space
        unprojected = to_cartesian(spherical)
        out = vec3.create()
        out = vec3.transformQuat(out, unprojected, origin.inverse_quat)

        # Unproject gnomonically to polar coordinates in origin space
        projected_spherical = to_spherical(out)
        polar = self.gnomonic.forward(projected_spherical)

        # Rotate around face axis to remove origin rotation  
        rho, gamma = polar
        polar = cast(Polar, (rho, gamma - origin.angle))

        face_triangle_index = self.get_face_triangle_index(polar)
        reflect = self.should_reflect(polar)
        face_triangle = self.get_face_triangle(face_triangle_index, reflect, False)
        spherical_triangle = self.get_spherical_triangle(face_triangle_index, origin_id, reflect)

        return self.polyhedral.forward(unprojected, spherical_triangle, face_triangle)

    def inverse(self, face: Face, origin_id: OriginId) -> Spherical:
        """
        Unprojects face coordinates to spherical coordinates using dodecahedron projection
        
        Args:
            face: Face coordinates [x, y]
            origin_id: Origin ID (0-11)
            
        Returns:
            Spherical coordinates [theta, phi]
        """
        polar = to_polar(face)
        face_triangle_index = self.get_face_triangle_index(polar)

        reflect = self.should_reflect(polar)
        face_triangle = self.get_face_triangle(face_triangle_index, reflect, False)
        spherical_triangle = self.get_spherical_triangle(face_triangle_index, origin_id, reflect)
        
        unprojected = self.polyhedral.inverse(face, face_triangle, spherical_triangle)
        return to_spherical(unprojected)

    def should_reflect(self, polar: Polar) -> bool:
        """
        Detects when point is beyond the edge of the dodecahedron face
        In the standard case (reflect = false), the face and spherical triangle can be
        used directly.
        In the reflected case (reflect = true), the point is beyond the edge of the dodecahedron face,
        and so the face triangle is squashed to unproject correctly onto the neighboring dodecahedron face.
        
        Args:
            polar: Polar coordinates
            
        Returns:
            True if point is beyond the edge of the dodecahedron face
        """
        rho, gamma = polar
        D = to_face((rho, self.normalize_gamma(gamma)))[0]
        return D > distance_to_edge

    def get_face_triangle_index(self, polar: Polar) -> FaceTriangleIndex:
        """
        Given a polar coordinate, returns the index of the face triangle it belongs to
        
        Args:
            polar: Polar coordinates
            
        Returns:
            Face triangle index, value from 0 to 9
        """
        _, gamma = polar
        return cast(FaceTriangleIndex, (math.floor(gamma / PI_OVER_5) + 10) % 10)

    def get_face_triangle(self, face_triangle_index: FaceTriangleIndex, reflected: bool = False, squashed: bool = False) -> FaceTriangle:
        """
        Gets the face triangle for a given polar coordinate
        
        Args:
            face_triangle_index: Face triangle index, value from 0 to 9
            reflected: Whether to get reflected triangle
            squashed: Whether to get squashed triangle
            
        Returns:
            FaceTriangle: 3 vertices in counter-clockwise order
        """
        index = face_triangle_index
        if reflected:
            index += 20 if squashed else 10

        # Extend array if needed
        while len(self.face_triangles) <= index:
            self.face_triangles.append(None)

        if self.face_triangles[index] is not None:
            return self.face_triangles[index]

        if reflected:
            self.face_triangles[index] = self._get_reflected_face_triangle(face_triangle_index, squashed)
        else:
            self.face_triangles[index] = self._get_face_triangle(face_triangle_index)
            
        return self.face_triangles[index]

    def _get_face_triangle(self, face_triangle_index: FaceTriangleIndex) -> FaceTriangle:
        """Get the basic (non-reflected) face triangle"""
        quintant = math.floor((face_triangle_index + 1) / 2) % 5

        vertices = get_quintant_vertices(quintant).get_vertices()
        v_center, v_corner1, v_corner2 = vertices[0], vertices[1], vertices[2]
        
        # Calculate edge midpoint using vec2.lerp like TypeScript
        v_edge_midpoint = vec2.create()
        vec2.lerp(v_edge_midpoint, v_corner1, v_corner2, 0.5)
        v_edge_midpoint = cast(Face, (v_edge_midpoint[0], v_edge_midpoint[1]))

        # Sign of gamma determines which triangle we want to use, and thus vertex order
        even = face_triangle_index % 2 == 0

        # Note: center & midpoint compared to DGGAL implementation are swapped
        # as we are using a dodecahedron, rather than an icosahedron.
        return [v_center, v_edge_midpoint, v_corner1] if even else [v_center, v_corner2, v_edge_midpoint]

    def _get_reflected_face_triangle(self, face_triangle_index: FaceTriangleIndex, squashed: bool = False) -> FaceTriangle:
        """Get the reflected face triangle"""
        # First obtain ordinary unreflected triangle
        face_triangle = self._get_face_triangle(face_triangle_index)
        A = vec2.clone(face_triangle[0])
        B = vec2.clone(face_triangle[1])
        C = vec2.clone(face_triangle[2])

        # Reflect dodecahedron center (A) across edge (BC)
        even = face_triangle_index % 2 == 0
        vec2.negate(A, A)
        midpoint = B if even else C

        # Squashing is important. A squashed triangle when unprojected will yield the correct spherical triangle.
        scale_factor = (1 + 1 / math.cos(interhedral_angle)) if squashed else 2
        # Manual scaleAndAdd: A = A + midpoint * scale_factor
        A[0] += midpoint[0] * scale_factor
        A[1] += midpoint[1] * scale_factor

        # Swap midpoint and corner to maintain correct vertex order
        return [cast(Face, (A[0], A[1])), cast(Face, (C[0], C[1])), cast(Face, (B[0], B[1]))]

    def get_spherical_triangle(self, face_triangle_index: FaceTriangleIndex, origin_id: OriginId, reflected: bool = False) -> SphericalTriangle:
        """
        Gets the spherical triangle for a given face triangle index and origin
        
        Args:
            face_triangle_index: Face triangle index
            origin_id: Origin ID
            reflected: Whether to get reflected triangle
            
        Returns:
            Spherical triangle
        """
        index = 10 * origin_id + face_triangle_index  # 0-119
        if reflected:
            index += 120

        # Extend array if needed
        while len(self.spherical_triangles) <= index:
            self.spherical_triangles.append(None)

        if self.spherical_triangles[index] is not None:
            return self.spherical_triangles[index]

        self.spherical_triangles[index] = self._get_spherical_triangle(face_triangle_index, origin_id, reflected)
        return self.spherical_triangles[index]

    def _get_spherical_triangle(self, face_triangle_index: FaceTriangleIndex, origin_id: OriginId, reflected: bool = False) -> SphericalTriangle:
        """Compute the spherical triangle for given parameters"""
        origin = origins[origin_id]
        face_triangle = self.get_face_triangle(face_triangle_index, reflected, True)
        
        spherical_triangle = []
        for face in face_triangle:
            rho, gamma = to_polar(face)
            rotated_polar = cast(Polar, (rho, gamma + origin.angle))
            rotated = to_cartesian(self.gnomonic.inverse(rotated_polar))
            # Transform using vec3.transformQuat like TypeScript
            transformed = vec3.create()
            vec3.transformQuat(transformed, rotated, origin.quat)
            vertex = crs.get_vertex(transformed)
            spherical_triangle.append(vertex)

        return cast(SphericalTriangle, tuple(spherical_triangle))

    def normalize_gamma(self, gamma: Radians) -> Radians:
        """
        Normalizes gamma to the range [-PI_OVER_5, PI_OVER_5]
        
        Args:
            gamma: The gamma value to normalize
            
        Returns:
            Normalized gamma value
        """
        segment = gamma / TWO_PI_OVER_5
        s_center = round(segment)
        s_offset = segment - s_center

        # Azimuthal angle from triangle bisector
        beta = s_offset * TWO_PI_OVER_5
        return cast(Radians, beta) 