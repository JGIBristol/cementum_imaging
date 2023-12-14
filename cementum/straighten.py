"""
Functions and helpers for straightening the images

"""
import numpy as np
import cv2
import shapely
from scipy.spatial import distance
from scipy.interpolate import interp1d
from PIL import ImageFilter, Image
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA


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


def _xy_clean(x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """
    Clean and smooth by interpolating a set of x and y coordinates

    :param x: The x coordinates to be cleaned.
    :param y: The y coordinates to be cleaned.

    :returns: The cleaned coordinates.
    """
    # Combine the x and y coordinates into a single array
    coordinates = np.concatenate(
        (x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)), axis=1
    )

    # Perform PCA on the coordinates
    pca = PCA(2)
    pca.fit(coordinates)
    coordinates_pca = pca.transform(coordinates)

    # Sort the coordinates by the x values
    sorted_indices = np.argsort(coordinates_pca[:, 0])
    sorted_coordinates = coordinates_pca[sorted_indices]

    # Interpolate more points
    interpolation = interp1d(
        sorted_coordinates[:, 0], sorted_coordinates[:, 1], kind="linear"
    )
    new_x = np.linspace(
        np.min(sorted_coordinates[:, 0]), np.max(sorted_coordinates[:, 0]), 100
    )
    new_y = interpolation(new_x)

    # Transform back to the original coordinate space
    cleaned_coordinates = pca.inverse_transform(
        np.concatenate((new_x.reshape(-1, 1), new_y.reshape(-1, 1)), axis=1)
    )

    # Return the cleaned coordinates as an integer array
    return np.hstack(
        (
            cleaned_coordinates[:, 0].reshape(-1, 1),
            cleaned_coordinates[:, 1].reshape(-1, 1),
        )
    ).astype(int)


def contour2skeleton(contour: np.ndarray) -> np.ndarray:
    """
    Convert a contour to a skeleton

    :param contour: contour to convert
    :returns: skeleton of the contour

    """
    raise NotImplementedError("I don't really know what this is meant to do")

    # Calculate the bounding rectangle of the contour
    x, y, width, height = cv2.boundingRect(contour)

    # Translate the contour to the origin
    contour_translated = contour - [x, y]

    # Create a blank binary image of the same size as the bounding rectangle
    binary_image = np.zeros((height, width))

    # Draw the translated contour onto the blank image and normalize it
    binary_image = (
        cv2.drawContours(
            binary_image, [contour_translated], -1, color=255, thickness=cv2.FILLED
        )
        // 255
    )

    # Reduce the shape in the image to a single-pixel wide skeleton
    skeleton = skeletonize(binary_image > 0)

    # Find the coordinates of the non-zero pixels in the skeleton image
    skeleton_coordinates = np.argwhere(skeleton > 0)

    # Flip the coordinates to (x, y) order
    skeleton_coordinates = np.flip(skeleton_coordinates, axis=None)

    # Separate the x and y coordinates into two separate arrays
    x_coords, y_coords = skeleton_coordinates[:, 0], skeleton_coordinates[:, 1]

    # Clean the x and y coordinates
    cleaned_skeleton = _xy_clean(x_coords, y_coords)

    # Translate the skeleton back to its original position
    cleaned_skeleton = cleaned_skeleton + [x, y]

    return cleaned_skeleton


def simplify(curve: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Simplify a curve - remove points that don't add much useful information about the shape of the curve

    :param curve: curve to simplify, (N, 2) array of points
    :param tolerance: the simplified line will be no further than this distance from the original

    :returns: simplified curve as an (M, 2) array of points

    """
    x, y = shapely.LineString(curve).simplify(tolerance=tolerance).xy

    return np.column_stack((x, y))


def extendline(
    line: tuple[tuple[float, float], tuple[float, float]], extra_length: float
) -> tuple[float, float]:
    """
    Extend a line segment by the given length in the direction from start to end

    :param line: two points; each being two co-ordinates (x, y)
    :param extra_length: the length to extend the line by

    :returns: the new end point of the line
    """
    start, end = line
    line_length = distance.euclidean(start, end)

    new_end_x = end[0] + ((end[0] - start[0]) / line_length * extra_length)
    new_end_y = end[1] + ((end[1] - end[1]) / line_length * extra_length)

    return (new_end_x, new_end_y)
