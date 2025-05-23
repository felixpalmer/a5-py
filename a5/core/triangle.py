# SPDX-License-Identifier: Apache-2.0
# Copyright (c) A5 contributors

import numpy as np

class Triangle:
    """
    Triangle class for fast repeated point-in-triangle testing.
    """

    def __init__(self, a, b, c):
        """
        Initialize the triangle with vertices a, b, c (2D points as numpy arrays or lists).
        """
        # Change to coordinate space where `a` is at origin
        self.origin = np.array(a, dtype=np.float64)
        self.edge1 = np.subtract(b, self.origin)
        self.edge2 = np.subtract(c, self.origin)
        self.test_point = np.zeros(2, dtype=np.float64)

        # Pre-calculate constant dot products
        self.dot11 = np.dot(self.edge1, self.edge1)
        self.dot12 = np.dot(self.edge1, self.edge2)
        self.dot22 = np.dot(self.edge2, self.edge2)
        inv_denom = 1.0 / (self.dot11 * self.dot22 - self.dot12 * self.dot12)
        self.dot11 *= inv_denom
        self.dot12 *= inv_denom
        self.dot22 *= inv_denom

    def contains_point(self, p):
        """
        Test if point p is inside the triangle.
        :param p: 2D point as a numpy array or list
        :return: True if point is in triangle, False otherwise
        """
        # Move test point to same coordinate space as triangle
        self.test_point[:] = np.subtract(p, self.origin)

        # Project onto edges
        dotp1 = np.dot(self.test_point, self.edge1)
        dotp2 = np.dot(self.test_point, self.edge2)

        # Check if point is in triangle
        u = self.dot22 * dotp1 - self.dot12 * dotp2
        if u < 0:
            return False
        v = self.dot11 * dotp2 - self.dot12 * dotp1
        return (v >= 0) and (u + v <= 1)