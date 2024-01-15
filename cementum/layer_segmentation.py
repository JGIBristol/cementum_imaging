"""
Classical segementation of the cementum layers

"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit


def fit_fcn_extra_peak(
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


def _fit_curve(x: np.ndarray, values: np.ndarray, initial_values: list):
    """
    Fit the curve

    """
    bounds = (
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -np.inf],
        [
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ],
    )

    popt, pcov = curve_fit(
        fit_fcn_extra_peak,
        x,
        values,
        p0=initial_values,
        bounds=bounds,
    )

    return popt


def find_cementum_edges(
    straightened_img: np.ndarray,
    initial_guess: list = None,
    *,
    return_params: bool = False
) -> tuple:
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
    :param initial_guess: initial guess for the parameters of the fit function. Must be length 12
    :param return_params: if True, return the parameters of the fit function

    :return: x-locations of the cementum edges as a tuple.
        Additionally return the parameters of the fit function if specified

    """
    # Choose initial values if not specified
    if not initial_guess:
        initial_guess = [75, 18, 50, 3, 2, 600, 22, 10, 85, 27, 2, 0.62]

    assert len(initial_guess) == 12, "Initial guess must be length 12"

    # independent variable is just the pixel number
    x = np.arange(straightened_img.shape[0])

    # Find the mean of each column
    y = np.mean(straightened_img, axis=0)

    # Fit
    params = _fit_curve(x, y, initial_guess)

    # Find the x-locations of the cementum edges
    a, d1, d2, d3 = [params[i] for i in [0, 3, 6, 9]]
    start, end = a + d1, a + d1 + d2 + d3

    if return_params:
        return (start, end), params

    return start, end
