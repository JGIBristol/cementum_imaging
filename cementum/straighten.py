"""
Functions and helpers for straightening the images

"""
import numpy as np
from PIL import ImageFilter, Image


def find_edges(image: np.ndarray) -> np.ndarray:
    """
    Find edges in the image

    :param image: image to find edges in

    :returns: image with edges indicated

    """
    edges = np.asarray(Image.fromarray(image).filter(ImageFilter.FIND_EDGES)).copy()

    # If the edge is on the outside of the image, remove it
    edges[:, 0] = 0
    edges[:, -1] = 0
    edges[0] = 0
    edges[-1] = 0

    return edges


def fit_edges(edges: np.ndarray) -> np.ndarray:
    """
    Given an array representing edge locations

    """
    assert np.unique(edges).size == 2, "Edges must be binary"

    # Find the x and y co-ordinates of the two edges

    # Correct for any large jumps in the x-coordinate

    # Fit polynomials to the edges

    # Return these polynomials
