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
    # Convert to radians
    theta = -theta * (np.pi / 180)

    # Find the width of the filter: must be at least 1
    width = int(np.floor((5 / 2) * sigma))
    assert width >= 1

    # Generate some x values spanning the filter
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
    image: np.ndarray, filterLvl: float, *, weight: float = 0.5
) -> np.ndarray:
    """
    Apply a vertical Gaussian filter to an image, using a combination of width-1 and width-2 filters.

    :param image: the image to apply the filter to
    :param filterLvl: the strength of the filter.
        np.inf returns the original image; 0 sets all pixels to NaN or inf.
        2 might be a reasonable value to start with
    :param weight: the weight to give to the width-1 filter. Must be

    :raises ValueError: if the weight is not between 0 and 1

    :returns: the filtered image as an 8-bit array

    """
    if not 0 <= weight <= 1:
        raise ValueError("Weight must be between 0 and 1")

    # Make two filters with different widths
    J = steerable_filter(image, 0, 1)
    K = steerable_filter(image, 0, 2)

    # Take a weighted average of the two filters
    mask = image + ((weight * J + (1 - weight) * K) * 10)

    filtered = image + (mask / filterLvl)

    # Scale the image to 16-bit
    min_, max_ = np.min(filtered), np.max(filtered)
    eight_bit_img = (2**16 * (((filtered - min_) / (max_ - min_)))).astype(np.uint16)

    return eight_bit_img
