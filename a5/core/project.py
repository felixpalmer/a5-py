import numpy as np
from .utils import PentagonShape, Origin
from .origin import move_point_to_face, find_nearest_origin, is_nearest_origin
from .dodecahedron import project_dodecahedron
from .coordinate_transforms import to_lonlat, to_polar
from .constants import PI_OVER_5

# Reusable matrix to avoid recreation
_rotation = None

def project_point(vertex, origin):
    global _rotation
    unwarped = to_polar(vertex)
    point = project_dodecahedron(unwarped, origin.quat, origin.angle)
    closest = origin if is_nearest_origin(point, origin) else find_nearest_origin(point)

    if closest.id != origin.id:
        # Move point to be relative to new origin
        if _rotation is None:
            _rotation = np.zeros((2, 2))
        angle = origin.angle
        c, s = np.cos(angle), np.sin(angle)
        _rotation[0, 0], _rotation[0, 1] = c, -s
        _rotation[1, 0], _rotation[1, 1] = s, c
        dodec_point2 = np.dot(_rotation, vertex)
        offset_dodec, interface_quat = move_point_to_face(dodec_point2, origin, closest)

        angle2 = 0.0
        if origin.angle != closest.angle and closest.angle != 0:
            angle2 = -PI_OVER_5

        polar2 = np.array(to_polar(offset_dodec))  # Convert tuple to numpy array
        polar2[1] = polar2[1] - angle2

        # Project back to sphere
        point2 = project_dodecahedron(tuple(polar2), interface_quat.flatten(), angle2)  # Ensure quaternion is flattened
        point = np.array(point)  # Convert point to numpy array for modification
        point[0] = point2[0]
        point[1] = point2[1]

    return to_lonlat(tuple(point))  # Convert back to tuple for return

def project_pentagon(pentagon: PentagonShape, origin: Origin):
    vertices = pentagon.get_vertices()
    rotated_vertices = [project_point(vertex, origin) for vertex in vertices]
    # Normalize longitudes to handle antimeridian crossing
    normalized_vertices = PentagonShape.normalize_longitudes(rotated_vertices)
    return normalized_vertices 