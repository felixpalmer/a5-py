"""
Quaternion helper functions for A5.
"""

import numpy as np

def conjugate(q: np.ndarray) -> np.ndarray:
    """
    Calculate the conjugate of a quaternion.
    
    Args:
        q: The quaternion [x, y, z, w]
        
    Returns:
        The conjugate quaternion [-x, -y, -z, w]
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])

def transform_quat(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Transform a vector by a quaternion.
    
    Args:
        a: The vector to transform [x, y, z]
        q: The quaternion [x, y, z, w]
        
    Returns:
        The transformed vector [x, y, z]
    """
    qx, qy, qz, qw = q
    x, y, z = a

    # Calculate cross product q × a
    uvx = qy * z - qz * y
    uvy = qz * x - qx * z
    uvz = qx * y - qy * x

    # Calculate cross product q × (q × a)
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx

    # Scale uv by 2 * w
    w2 = qw * 2
    uvx *= w2
    uvy *= w2
    uvz *= w2

    # Scale uuv by 2
    uuvx *= 2
    uuvy *= 2
    uuvz *= 2

    # Add all components
    return np.array([
        x + uvx + uuvx,
        y + uvy + uuvy,
        z + uvz + uuvz
    ])

def set_axis_angle(axis: np.ndarray, rad: float) -> np.ndarray:
    """
    Sets a quaternion from the given angle and rotation axis.
    
    Args:
        axis: The axis around which to rotate [x, y, z]
        rad: The angle in radians
        
    Returns:
        The quaternion [x, y, z, w]
    """
    rad = rad * 0.5
    s = np.sin(rad)
    return np.array([
        s * axis[0],
        s * axis[1],
        s * axis[2],
        np.cos(rad)
    ])

def rotation_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Sets a quaternion to represent the shortest rotation from one vector to another.
    Both vectors are assumed to be unit length.
    
    Args:
        a: The initial vector [x, y, z]
        b: The destination vector [x, y, z]
        
    Returns:
        The quaternion [x, y, z, w]
    """
    dot = np.dot(a, b)
    
    if dot < -0.999999:
        # Vectors are nearly opposite, use x-axis as reference
        tmpvec = np.cross(np.array([1, 0, 0]), a)
        if np.linalg.norm(tmpvec) < 0.000001:
            # If x-axis is parallel to a, use y-axis
            tmpvec = np.cross(np.array([0, 1, 0]), a)
        tmpvec = tmpvec / np.linalg.norm(tmpvec)
        return set_axis_angle(tmpvec, np.pi)
    elif dot > 0.999999:
        # Vectors are nearly parallel, return identity quaternion
        return np.array([0, 0, 0, 1])
    else:
        # Normal case
        tmpvec = np.cross(a, b)
        out = np.array([tmpvec[0], tmpvec[1], tmpvec[2], 1 + dot])
        # Normalize
        return out / np.linalg.norm(out) 