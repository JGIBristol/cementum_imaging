"""
Steerable Gaussian filter for cementum preprocessing

"""
import numpy as np
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
    x = np.linspace(-width, width, (width * 2) + 1)

    # Calculate the Gaussian and its derivative
    gaussian = np.exp(-(x**2) / (2 * sigma**2))
    derivative = (x / sigma) * np.exp(-(x**2) / (2 * sigma**2))

    # Calculate image gradients along axes
    i1 = convolve1d(np.asarray(np.rot90(img)), derivative, axis=0, mode="constant")
    Ix = np.rot90(convolve1d(i1, gaussian, mode="constant"), 3)

    i2 = convolve1d(np.asarray(np.rot90(img)), gaussian, axis=0, mode="constant")
    Iy = np.rot90(convolve1d(i2, derivative, mode="constant"), 3)

    # Evaluate oriented filter response
    return np.cos(theta) * Ix + np.sin(theta) * Iy


def steerGaussWrapper(cem, filterLvl):
    """
    Perform steerable gaussian filter - must have "steerGauss.m" open.
    """
    J = steerable_filter(cem, 90, 1)
    K = steerable_filter(cem, 90, 2)

    """
    then create mask
    """

    mask = cem + (J * 10)
    maskToSave = cem + (K * 10)

    """
    make array of both matrices
    """
    maskArray = np.uint8(mask)
    cemArray = np.uint8(cem)

    """
    And filter
    """
    filteredCem = cem + (mask / filterLvl)

    filterVec = np.mean(filteredCem, 0)
    filterVec = filterVec[filterVec != 0]

    dynamicRange = max(filterVec) - min(filterVec)

    eightBitTest = filteredCem * (
        256 * ((filteredCem - np.min(filterVec)) / np.max(filteredCem))
    )
    eightBitTest = np.uint8(eightBitTest)

    sixteenBitTest = filteredCem * (
        65536 * ((filteredCem - np.min(filterVec)) / np.max(filteredCem))
    )
    sixteenBitTest = np.uint16(sixteenBitTest)

    return filteredCem, eightBitTest
