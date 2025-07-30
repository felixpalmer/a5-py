# A5
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import math
from typing import List, Tuple, Union, cast, Literal
from ..core.coordinate_systems import Radians, Spherical, Cartesian, Polar, Face, FaceTriangle, SphericalTriangle
from ..core.coordinate_transforms import to_cartesian, to_spherical, to_face, to_polar
from ..core.quat import conjugate, transform_quat
from ..core.constants import distance_to_edge, interhedral_angle, PI_OVER_5, TWO_PI_OVER_5
from ..core.origin import origins
from ..core.tiling import get_quintant_vertices
from .gnomonic import GnomonicProjection
from .polyhedral import PolyhedralProjection
from .crs import CRS
from ..math import vec2, vec3, quat as quat_glm

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
        self._face_triangles: List[Tuple[Face, Face, Face]] = [None] * 30  # 10 regular + 10 reflected + 10 squashed
        self._spherical_triangles: List[SphericalTriangle] = [None] * 240  # 120 regular + 120 reflected  
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
        out = transform_quat(unprojected, origin.inverse_quat)

        # Unproject gnomonically to polar coordinates in origin space
        projected_spherical = to_spherical(out)
        polar = self.gnomonic.forward(projected_spherical)

        # Rotate around face axis to remove origin rotation  
        rho, gamma = polar
        polar = cast(Polar, (rho, gamma - origin.angle))

        face_triangle_index = self._get_face_triangle_index(polar)
        face_triangle = self._get_face_triangle(face_triangle_index)
        spherical_triangle = self._get_spherical_triangle(face_triangle_index, origin_id, False)

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
        face_triangle_index = self._get_face_triangle_index(polar)

        # Detect when point is beyond the edge of the dodecahedron face
        rho, gamma = polar
        D = to_face((rho, self._normalize_gamma(gamma)))[0]
        reflect = D > distance_to_edge

        # In the standard case, both these are the same as we can unproject directly
        # In the reflected case, both are off the edge of the dodecahedron face,
        # with the squashed triangle unprojecting correctly onto the neighboring dodecahedron face.
        face_triangle = self._get_face_triangle(face_triangle_index, reflect, False)
        spherical_triangle = self._get_spherical_triangle(face_triangle_index, origin_id, reflect)
        
        unprojected = self.polyhedral.inverse(face, face_triangle, spherical_triangle)
        return to_spherical(unprojected)

    def _get_face_triangle_index(self, polar: Polar) -> FaceTriangleIndex:
        """
        Given a polar coordinate, returns the index of the face triangle it belongs to
        
        Args:
            polar: Polar coordinates
            
        Returns:
            Face triangle index, value from 0 to 9
        """
        _, gamma = polar
        return cast(FaceTriangleIndex, (math.floor(gamma / PI_OVER_5) + 10) % 10)

    def _get_face_triangle(self, face_triangle_index: FaceTriangleIndex, reflected: bool = False, squashed: bool = False) -> FaceTriangle:
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

        if self._face_triangles[index] is not None:
            return self._face_triangles[index]

        if reflected:
            self._face_triangles[index] = self._get_reflected_face_triangle(face_triangle_index, squashed)
        else:
            self._face_triangles[index] = self._get_basic_face_triangle(face_triangle_index)
            
        return self._face_triangles[index]

    def _get_basic_face_triangle(self, face_triangle_index: FaceTriangleIndex) -> FaceTriangle:
        """Get the basic (non-reflected) face triangle"""
        quintant = math.floor((face_triangle_index + 1) / 2) % 5

        vertices = get_quintant_vertices(quintant).get_vertices()
        v_center, v_corner1, v_corner2 = vertices[0], vertices[1], vertices[2]
        
        # Calculate edge midpoint using gl-matrix style
        edge_midpoint = vec2.create()
        vec2.add(edge_midpoint, v_corner1, v_corner2)
        vec2.scale(edge_midpoint, edge_midpoint, 0.5)
        v_edge_midpoint = cast(Face, (edge_midpoint[0], edge_midpoint[1]))

        # Sign of gamma determines which triangle we want to use, and thus vertex order
        even = face_triangle_index % 2 == 0

        # Note: center & midpoint compared to DGGAL implementation are swapped
        # as we are using a dodecahedron, rather than an icosahedron.
        if even:
            return cast(FaceTriangle, (v_center, v_edge_midpoint, v_corner1))
        else:
            return cast(FaceTriangle, (v_center, v_corner2, v_edge_midpoint))

    def _get_reflected_face_triangle(self, face_triangle_index: FaceTriangleIndex, squashed: bool = False) -> FaceTriangle:
        """Get the reflected face triangle"""
        # First obtain ordinary unreflected triangle
        A, B, C = self._get_basic_face_triangle(face_triangle_index)

        # Reflect dodecahedron center (A) across edge (BC)
        even = face_triangle_index % 2 == 0
        A_reflected = (-A[0], -A[1])
        midpoint = B if even else C

        # Squashing is important. A squashed triangle when unprojected will yield the correct spherical triangle.
        scale_factor = (1 + 1 / math.cos(interhedral_angle)) if squashed else 2
        A_final = (
            A_reflected[0] + midpoint[0] * scale_factor,
            A_reflected[1] + midpoint[1] * scale_factor
        )

        # Swap midpoint and corner to maintain correct vertex order
        return cast(FaceTriangle, (cast(Face, A_final), cast(Face, C), cast(Face, B)))

    def _get_spherical_triangle(self, face_triangle_index: FaceTriangleIndex, origin_id: OriginId, reflected: bool = False) -> SphericalTriangle:
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

        if self._spherical_triangles[index] is not None:
            return self._spherical_triangles[index]

        self._spherical_triangles[index] = self._compute_spherical_triangle(face_triangle_index, origin_id, reflected)
        return self._spherical_triangles[index]

    def _compute_spherical_triangle(self, face_triangle_index: FaceTriangleIndex, origin_id: OriginId, reflected: bool = False) -> SphericalTriangle:
        """Compute the spherical triangle for given parameters"""
        origin = origins[origin_id]
        face_triangle = self._get_face_triangle(face_triangle_index, reflected, True)

        spherical_triangle = []
        for face in face_triangle:
            rho, gamma = to_polar(face)
            rotated_polar = cast(Polar, (rho, gamma + origin.angle))
            rotated = to_cartesian(self.gnomonic.inverse(rotated_polar))
            transformed_vec = vec3.create()
            vec3.transformQuat(transformed_vec, rotated, origin.quat)
            transformed = (transformed_vec[0], transformed_vec[1], transformed_vec[2])
            vertex = crs.get_vertex(transformed)
            spherical_triangle.append(vertex)

        return cast(SphericalTriangle, tuple(spherical_triangle))

    def _normalize_gamma(self, gamma: Radians) -> Radians:
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