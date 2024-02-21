"""
Functions and helpers for straightening the images

"""

import warnings

import numpy as np
import cv2
import shapely
from scipy.spatial import distance
from scipy.interpolate import interp1d
from scipy.ndimage import label as scipy_label
from PIL import ImageFilter, Image
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from skimage.transform import warp, PiecewiseAffineTransform

from . import util


class MultipleEdgesError(Exception):
    pass


class Not3RegionsError(MultipleEdgesError):
    pass


class NotContiguousError(MultipleEdgesError):
    pass


def check_mask(mask: np.ndarray) -> None:
    """
    Check whether a mask is valid

    It should have three unique regions (background, cementum, dentin) that are all contiguous

    :param mask: mask labelling background, cementum, dentin

    """
    n_unique = len(np.unique(mask.flat))

    # Check that we have exactly three values in our mask
    if n_unique != 3:
        raise Not3RegionsError("Mask should have exactly three unique values")

    # Check that the three regions are all contiguous
    labelled_mask, _ = scipy_label(mask)
    counts = np.bincount(labelled_mask.flat)
    for i, count in enumerate(counts, start=1):
        if count != 1:
            raise NotContiguousError(f"Region {i} is not contiguous")


def find_edges(image: np.ndarray) -> np.ndarray:
    """
    Find edges in the image

    :param image: image to find edges in

    :returns: image with edges indicated
    :raises: all sorts of stuff

    """
    check_mask(image)
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
        if not (len(unique_diffs) == 1 or len(unique_diffs) == 2 and 1 in unique_diffs):
            raise MultipleEdgesError(
                f"There should only be two edges: found {len(x_at_y)} at {y=}: {x_at_y} {np.diff(x_at_y)}"
            )

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
    new_end_y = end[1] + ((end[1] - start[1]) / line_length * extra_length)

    return (new_end_x, new_end_y)


def extend_curve(curve: np.ndarray, extra_length: float) -> np.ndarray:
    """
    Given an (N, 2) array of points, extend at both ends by the given length to give
    an (N + 2, 2) array of points.

    :param curve: curve to extend as an (N, 2) array of points
    :param extra_length: the length to extend the curve by

    """
    new_start = np.array(extendline((curve[1], curve[0]), extra_length))
    new_end = np.array(extendline((curve[-2], curve[-1]), extra_length))

    return np.vstack((new_start, curve, new_end))


def straighten_curve(curve: np.ndarray) -> np.ndarray:
    """
    Given an (N, 2) array of points (x, y), return a 1d length-N array of coordinates along the curve

    The first co-ordinate is the start of the curve (not 0)

    :param curve: curve to straighten as an (N, 2) array of points
    :returns: straightened curve as a length-N array of coordinates

    """
    x_diffs = curve[1:, 0] - curve[:-1, 0]
    y_diffs = curve[1:, 1] - curve[:-1, 1]
    lengths = np.sqrt(x_diffs**2 + y_diffs**2)

    # Prepend with 0
    lengths = np.concatenate(([0], lengths))

    # Cumulative sum
    lengths = np.cumsum(lengths)

    return curve[0, 1] + lengths


def partial_transform_matrix(
    start_line: np.ndarray, end_line: np.ndarray
) -> np.ndarray:
    """
    Calculate the affine transformation matrix to transform the start line to the end line

    :param start_line: start line to transform, as an (N, 2) array of points
    :param end_line: end line to transform to, as an (N, 2) array of points
    :param partial: whether to return a partial transformation matrix (only translation, rotation and uniform scaling)

    :returns: 2x3 transformation matrix

    """

    matrix, inliers = cv2.estimateAffinePartial2D(start_line, end_line)

    # Check there are no outliers
    # for inlier, start, end in zip(inliers, start_line, end_line):
    #     if not inlier:
    #         raise ValueError(f"Outlier found at: {start=}->{end=}")

    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Transform the given points by the given matrix

    :param points: points to transform, as an (N, 2) array of points
    :param matrix: transformation matrix, as a 2x3 array

    :returns: transformed points as an (N, 2) array of points

    """
    return cv2.transform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)


def _length(curve: np.ndarray) -> float:
    return np.sum(np.sqrt(np.diff(curve[:, 0]) ** 2 + np.diff(curve[:, 1]) ** 2))


def straight_mesh(
    mask: np.ndarray,
    n_y: int,
    n_x: tuple[int, int, int],
    *,
    poly_degree: int = 6,
) -> np.ndarray:
    """
    From a mask, create a corresponding straightened mesh

    Here the cementum region is centred on the same place as in the original mask,
    and the width is such that the overall area is the same.
    This means that it will be slightly longer than the original mask

    """

    def area(curve1: np.ndarray, curve2: np.ndarray) -> float:
        # Create a shapely polygon
        # Reverse the second curve so that the polygon is closed and the points describe
        # a path around it in the right sense
        return shapely.Polygon(np.vstack([curve1, curve2[::-1]])).area

    n_left, n_inside, n_right = n_x

    # Find the edges of the mask
    first_edge, last_edge = _identify_edges(find_edges(mask))

    # Find their average length
    avg_length = (_length(first_edge) + _length(last_edge)) / 2

    # y values are linearly spaced, plus an extra point at the bottom
    y_vals = np.linspace(0, avg_length, n_y, endpoint=True)
    y_vals = np.concatenate((y_vals, [avg_length + y_vals[-1] - y_vals[-2]]))

    # Find the area enclosed
    total_area = area(first_edge, last_edge)

    # Find the width
    width = total_area / avg_length

    # Find the average x-coordinate of the edges
    # This will be the centre of the straight region in the new mask
    middle = (np.mean(first_edge[:, 0]) + np.mean(last_edge[:, 0])) / 2
    left = middle - width / 2
    right = middle + width / 2

    pts = []
    for y_val in y_vals:
        pts.append(
            np.column_stack(
                [
                    np.linspace(0, left, n_left - 1, endpoint=False),
                    np.full(n_left - 1, y_val),
                ]
            )
        )

        # Find points in the middle of the curve
        pts.append(
            np.column_stack(
                [
                    np.linspace(left, right, n_inside - 1, endpoint=False),
                    np.full(n_inside - 1, y_val),
                ]
            )
        )

        # Find points on the right of the curve
        pts.append(
            np.column_stack(
                [np.linspace(right, mask.shape[0], n_right), np.full(n_right, y_val)]
            )
        )
    pts = np.concatenate(pts, axis=0)

    # Order the points by y-value, then x
    return pts[np.lexsort((pts[:, 0], pts[:, 1]))]


def mask_mesh(
    mask: np.ndarray,
    n_y: int,
    n_x: tuple[int, int, int],
    *,
    poly_degree: int = 6,
) -> np.ndarray:
    """
    Given a mask, return a mesh of points that are evenly spaced within the masked regions

    The mask should run from the top to the bottom of the image

    """
    n_left, n_inside, n_right = n_x

    # Find the edges of the mask
    first_edge, last_edge = _identify_edges(find_edges(mask))

    # Fit polynomials to them
    first_poly = _fit_polynomial(first_edge, poly_degree)
    last_poly = _fit_polynomial(last_edge, poly_degree)

    # Create an array of y-values
    y_vals = np.linspace(0, mask.shape[1], n_y, endpoint=True)

    # Append another y value to the end
    avg_length = (_length(first_edge) + _length(last_edge)) / 2
    y_vals = np.concatenate((y_vals, [avg_length + y_vals[-1] - y_vals[-2]]))

    pts = []
    for y_val, first, last in zip(y_vals, first_poly(y_vals), last_poly(y_vals)):
        # Find points on the left of the curve
        pts.append(
            np.column_stack(
                [
                    np.linspace(0, first, n_left - 1, endpoint=False),
                    np.full(n_left - 1, y_val),
                ]
            )
        )

        # Find points in the middle of the curve
        pts.append(
            np.column_stack(
                [
                    np.linspace(first, last, n_inside - 1, endpoint=False),
                    np.full(n_inside - 1, y_val),
                ]
            )
        )

        # Find points on the right of the curve
        pts.append(
            np.column_stack(
                [np.linspace(last, mask.shape[0], n_right), np.full(n_right, y_val)]
            )
        )
    pts = np.concatenate(pts, axis=0)

    # Order the points by y-value, then x
    return pts[np.lexsort((pts[:, 0], pts[:, 1]))]


def apply_transformation(
    image: np.ndarray, curve_mesh: np.ndarray, straight_mesh: np.ndarray, **warp_kw
) -> np.ndarray:
    """
    Apply the transformation to the image

    """
    # Estimate the transformation
    transform = PiecewiseAffineTransform()
    result = transform.estimate(curve_mesh, straight_mesh)
    assert result

    # If the image doesn't extend as far as the mesh, pad it with zeros
    y_extent = straight_mesh[:, 1].max()
    if image.shape[1] < y_extent:
        # Find how many pixels to pad by
        extra_pixels = int(straight_mesh[:, 1].max() - image.shape[1])

        # Pad the image
        image = np.pad(image, ((0, extra_pixels), (0, 0)), mode="constant")

    # This is a bad solution
    kw = {"clip": False, "cval": 255, "order": 0}
    if "clip" in warp_kw:
        kw["clip"] = warp_kw["clip"]
    if "cval" in warp_kw:
        kw["cval"] = warp_kw["cval"]
    if "order" in warp_kw:
        kw["order"] = warp_kw["order"]

    # Apply the transformation to the image
    transformed_image = warp(image, transform.inverse, **kw, preserve_range=True)

    return transformed_image


def remove_white_cols(
    straight_image: np.ndarray, straight_mask: np.ndarray
) -> np.ndarray:
    """
    After the straightening, sometimes the image is padded with white columns on the right hand side.
    We should remove these columns, from both the image and the mask.
    This isn't a long term solution, but it will do for now.

    :param straight_image: image after straigtening
    :param mask: mask after straigtening

    """
    assert straight_image.shape == straight_mask.shape

    # Find which columns are all white
    zero_rows = np.all(straight_image == np.max(straight_image), axis=0)

    last_non_zero_col = np.max(np.where(zero_rows == False))
    n_to_remove = straight_image.shape[0] - last_non_zero_col
    warnings.warn(
        util.coloured(
            f"Temporary solution: removing {n_to_remove} saturated columns from right",
            util.bcolours.WARNING,
        )
    )

    # Find the last non-zero column from the right
    return (
        straight_image[:, : last_non_zero_col + 1],
        straight_mask[:, : last_non_zero_col + 1],
    )


def remove_padding(image: np.ndarray) -> np.ndarray:
    """
    Remove padding of 0s at the bottom of an image

    """
    # Find which rows are all 0
    zero_rows = np.all(image == 0, axis=1)

    # Find the last non-zero row from the bottom
    last_non_zero_row = np.max(np.where(zero_rows == False))

    n_to_remove = last_non_zero_row - image.shape[1]
    warnings.warn(
        util.coloured(
            f"Temporary solution: removing {n_to_remove} zero rows from bottom",
            util.bcolours.WARNING,
        )
    )

    # Remove zero rows from the bottom
    return image[: last_non_zero_row + 1]
