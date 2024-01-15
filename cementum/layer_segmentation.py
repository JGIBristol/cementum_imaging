"""
Classical segementation of the cementum layers

"""
import numpy as np
from scipy.stats import norm


def fit_fcn(
    x,
    a: float,
    b: float,
    h1: float,
    d1: float,
    s1: float,
    h2: float,
    d2: float,
    s2: float,
    h3: float,
    d3: float,
    s3: float,
    k: float,
):
    """
    Fit function to find the cementum region

    parameterised such that d1, d2 > 0

    :param x: x value
    :param a: constant, below which f(x) = 0
    :param b: height of step
    :param h: height of a Gaussian
    :param d: delta to the middle of the Gaussian
    :param s: sigma of the Gaussian
    :param k: gradient of linear term

    """
    gauss1 = h1 * norm(loc=a + d1, scale=s1).pdf(x)
    gauss2 = h2 * norm(loc=a + d1 + d2, scale=s2).pdf(x)
    gauss3 = h3 * norm(loc=a + d1 + d2 + d3, scale=s3).pdf(x)

    retval = b + gauss1 + gauss2 + gauss3 + k * x

    # Set values below a to 0
    retval[x < a] = 0

    return retval


def find_cementum_edges(straightened_img: np.ndarray) -> tuple[float, float]:
    """
    Find the edges of the cementum layers in a straightened image; the cementum must
    run vertically in the image, with the background on the left.

    We expect the cementum layers to run vertically, so we take the mean of each column
    in the image and then fit an empirical distribution to it.
    We expect to see:
        - A dark region (0 intensity) for the background
        - A sharp step up to the first layer, with a peak marking the start of the layer
        - A Gaussian intensity variation within the cementum
        - A peak marking the end of the cementum
        - A linear increase throughout the dentine

    :param straightened_img: the straightened image

    :return: x-locations of the cementum edges

    """
