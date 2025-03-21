"""This file contains helper functions for the four.py file."""

import numpy as np
from shapely.geometry import LineString


def interpolate_line_coords(coord1, coord2, num_points=10):
    """

    Generate intermediate coordinates between nodes using lin interpolation.

    Args:
        coord1 (tuple): The starting coordinate (longitude, latitude)
        coord2 (tuple): The ending coordinate (longitude, latitude)
        num_points (int): The number of points to generate

    Returns:
        list: A list of coordinates
    """
    line = LineString([coord1, coord2])
    distances = np.linspace(0, line.length, num_points)
    return [line.interpolate(distance).coords[0] for distance in distances]
