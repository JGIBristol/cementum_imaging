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
    y_coords, x_coords = edges.nonzero()
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

    return np.stack((first_edge_x, unique_y_coords), axis=-1), np.stack(
        (last_edge_x, unique_y_coords), axis=-1
    )


def _correct_jumps(edge: np.ndarray, threshold: int) -> np.ndarray:
    """
    Given arrays representing edge locations, find where the x-coordinate jumps by a large amount
    and smooth it

    """
    smoothed_x = edge[:, 0].copy()
    # Find the locations where the x-coordinate jumps by a large amount
    for location in (np.diff(smoothed_x) > threshold).nonzero()[0]:
        smoothed_x[location + 1] = smoothed_x[location]

    return np.stack((smoothed_x, edge[:, 1]), axis=-1)


def _fit_polynomial(edge: np.ndarray, degree: int) -> np.ndarray:
    """
    Fit a polynomial to the provided edge
    This will be parameterised as x = f(y), since the edge is vertical

    """
    return np.polynomial.Polynomial.fit(
        edge[:, 1], edge[:, 0], degree, domain=(0, len(edge))
    )


def fit_edges(
    edges: np.ndarray, *, jump_threshold: int = 10, fit_degree=6
) -> np.ndarray:
    """
    Given an array representing edge locations,

    """
    assert np.unique(edges).size == 2, "Edges must be binary"

    # Find the x and y co-ordinates of both edges
    first_edge, last_edge = _identify_edges(edges)

    # Correct for any large jumps in the x-coordinate
    smoothed_first_edge = _correct_jumps(first_edge, jump_threshold)
    smoothed_last_edge = _correct_jumps(last_edge, jump_threshold)

    # Fit polynomials to the edges
    first_coefs = _fit_polynomial(smoothed_first_edge, fit_degree)
    last_coefs = _fit_polynomial(smoothed_last_edge, fit_degree)

    # Return these polynomials
    return first_coefs, last_coefs
