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


def _identify_edges(edges: np.ndarray) -> np.ndarray:
    """
    Given an array representing edge locations,
    return a tuple of arrays representing both edgse

    """
    y_coords, x_coords = np.where(edges)
    unique_y_coords = np.unique(y_coords)

    # Find the co-ordinates of the two edges
    first_edge_x = []
    last_edge_x = []

    for y in unique_y_coords:
        x_at_y = x_coords[y_coords == y]

        # Check that there's only two edges:
        unique_diffs = np.unique(np.diff(x_at_y))
        assert (
            len(unique_diffs) == 1 or len(unique_diffs) == 2 and 1 in unique_diffs
        ), f"There should only be two edges: found {len(x_at_y)} at {y=}: {x_at_y} {np.diff(x_at_y)}"

        first_edge_x.append(np.min(x_at_y))
        last_edge_x.append(np.max(x_at_y))

    first_edge_x = np.array(first_edge_x)
    last_edge_x = np.array(last_edge_x)

    return np.stack((unique_y_coords, first_edge_x), axis=-1), np.stack(
        (unique_y_coords, last_edge_x), axis=-1
    )


def fit_edges(edges: np.ndarray) -> np.ndarray:
    """
    Given an array representing edge locations,

    """
    assert np.unique(edges).size == 2, "Edges must be binary"

    # Find the x and y co-ordinates of both edges
    first_edge, last_edge = _identify_edges(edges)

    # Correct for any large jumps in the x-coordinate

    # Fit polynomials to the edges

    # Return these polynomials
