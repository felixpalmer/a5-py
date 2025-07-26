import math
from .utils import Origin
from ..geometry.pentagon import PentagonShape
from .origin import move_point_to_face, find_nearest_origin, is_nearest_origin
from .dodecahedron import project_dodecahedron
from .coordinate_transforms import to_lonlat, to_polar
from .constants import PI_OVER_5

def project_point(vertex, origin):
    unwarped = to_polar(vertex)
    point = project_dodecahedron(unwarped, origin.quat, origin.angle)
    closest = origin if is_nearest_origin(point, origin) else find_nearest_origin(point)

    if closest.id != origin.id:
        # Move point to be relative to new origin
        angle = origin.angle
        c, s = math.cos(angle), math.sin(angle)
        
        # Manual 2x2 matrix multiplication
        dodec_point2 = (
            c * vertex[0] - s * vertex[1],
            s * vertex[0] + c * vertex[1]
        )
        
        offset_dodec, interface_quat = move_point_to_face(dodec_point2, origin, closest)

        angle2 = 0.0
        if origin.angle != closest.angle and closest.angle != 0:
            angle2 = -PI_OVER_5

        polar2 = list(to_polar(offset_dodec))  # Convert to list for modification
        polar2[1] = polar2[1] - angle2

        # Project back to sphere
        point2 = project_dodecahedron(tuple(polar2), interface_quat, angle2)
        point = list(point)  # Convert to list for modification
        point[0] = point2[0]
        point[1] = point2[1]
        point = tuple(point)

    return to_lonlat(point)

def project_pentagon(pentagon: PentagonShape, origin: Origin):
    vertices = pentagon.get_vertices()
    rotated_vertices = [project_point(vertex, origin) for vertex in vertices]
    # Normalize longitudes to handle antimeridian crossing
    normalized_vertices = PentagonShape.normalize_longitudes(rotated_vertices)
    return normalized_vertices