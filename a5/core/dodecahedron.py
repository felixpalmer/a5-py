# dodecahedron.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

# NumPy removed
from typing import Tuple
from .quat import transform_quat, conjugate
from .coordinate_transforms import Radians, Spherical, Cartesian, Polar, to_cartesian, to_spherical
from .warp import warp_polar, unwarp_polar
from ..projections.gnomonic import GnomonicProjection

gnomonic = GnomonicProjection()

def project_dodecahedron(unwarped: Polar, origin_transform: Tuple[float, float, float, float], origin_rotation: Radians) -> Spherical:
    # Warp in polar space to minimize area variation across sphere
    rho, gamma = warp_polar(unwarped)

    # Rotate around face axis to match origin rotation
    polar = (rho, gamma + origin_rotation)

    # Project gnomonically onto sphere and obtain cartesian coordinates
    projected_spherical = gnomonic.forward(polar)
    projected = to_cartesian(projected_spherical)  # [x, y, z]

    # Rotate to correct orientation on globe and return spherical coordinates
    rotated = transform_quat(projected, origin_transform)
    return to_spherical(rotated)


def unproject_dodecahedron(spherical: Spherical, origin_transform: Tuple[float, float, float, float], origin_rotation: Radians) -> Polar:
    # Transform back to origin space
    cartesian = to_cartesian(spherical)
    inverse_quat = conjugate(origin_transform)
    rotated = transform_quat(cartesian, inverse_quat)

    # Unproject gnomonically to polar coordinates in origin space
    projected_spherical = to_spherical(rotated)
    polar = gnomonic.inverse(projected_spherical)

    # Rotate around face axis to remove origin rotation
    rho, gamma = polar
    gamma -= origin_rotation

    # Unwarp the polar coordinates to obtain points in lattice space
    return unwarp_polar((rho, gamma))
