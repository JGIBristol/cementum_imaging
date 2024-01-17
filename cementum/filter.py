"""
Steerable Gaussian filter for cementum preprocessing

"""
import numpy as np
from scipy.stats import norm
from scipy.ndimage.filters import convolve1d


def steerable_filter(img: np.ndarray, theta: float, sigma: float):
    """
    Applies a steerable Gaussian filter to an image.

    :param img: image to apply the filter to
    :param theta: angle of the filter, in degrees
    :param sigma: standard deviation of the filter

    :return: the filtered image

    """
    # Sigma needs to be at least 0.4 for the width to be at least 1
    assert sigma >= 0.4

    # Convert to radians
    theta = -theta * (np.pi / 180)

    # Generate some x values spanning the filter; we need at least 1
    width = int(np.floor((5 / 2) * sigma))
    x = np.arange(-width, width + 1)

    # Calculate the Gaussian and its derivative
    gaussian = norm.pdf(x, loc=0, scale=sigma)
    derivative = -(x / sigma) * gaussian

    # Calculate image gradients along axes
    i1 = convolve1d(np.asarray(np.rot90(img)), derivative, axis=0, mode="constant")
    Ix = np.rot90(convolve1d(i1, gaussian, mode="constant"), 3)

    i2 = convolve1d(np.asarray(np.rot90(img)), gaussian, axis=0, mode="constant")
    Iy = np.rot90(convolve1d(i2, derivative, mode="constant"), 3)

    # Evaluate oriented filter response
    return np.cos(theta) * Ix + np.sin(theta) * Iy


def apply_weighted_filters(
    image: np.ndarray,
    widths: tuple,
    *,
    weights: tuple = None,
    filterLvl: float = 2,
    scale_factor: float = 100,
) -> np.ndarray:
    """
    Apply a vertical Gaussian filter to an image, using a combination of width-1 and width-2 filters.

    :param image: the image to apply the filter to
    :param widths: the widths of the filters to use
    :param weights: the relative weighting of filters. If not specified, uses equally weighted filters
    :param filterLvl: the strength of the filter.
        np.inf returns the original image; 0 sets all pixels to NaN or inf.
    :param scale_factor: scale factor applied to the filters

    :raises ValueError: if any weight is not between 0 and 1

    :returns: the filtered image as an 8-bit array

    """
    n_filters = len(widths)
    if weights is None:
        weights = [1 / n_filters] * n_filters

    for weight in weights:
        if not 0 <= weight <= 1:
            raise ValueError("Weights must all be between 0 and 1")

    # Make filters with different widths
    filters = [steerable_filter(image, 0, width) for width in widths]

    # Take a weighted average of the filters
    mask = image.copy().astype(np.float64)
    for weight, filter in zip(weights, filters):
        mask += weight * filter * scale_factor

    # Apply the combined filters to the image, scaling by filterlvl
    filtered = image + (mask / filterLvl)

    # Scale the image to 16-bit
    min_, max_ = np.min(filtered), np.max(filtered)
    eight_bit_img = (2**16 * (((filtered - min_) / (max_ - min_)))).astype(np.uint16)

    return eight_bit_img
