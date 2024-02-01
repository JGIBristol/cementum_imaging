"""
Classical segementation of the cementum layers

"""
import warnings

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from . import util


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

    retval = b + gauss1 + gauss2 + k * x

    # Set values below a to 0
    retval[x < a] = 0

    return retval


def _fit_curve(x: np.ndarray, values: np.ndarray, initial_values: list):
    """
    Fit the curve

    """
    bounds = (
        [0, 0, 0, 0, 0, 0, 0, 0, -np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    )

    popt, pcov = curve_fit(
        fit_fcn,
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
    return_params: bool = False,
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
    :param initial_guess: initial guess for the parameters of the fit function. Must be length 9
    :param return_params: if True, return the parameters of the fit function

    :return: x-locations of the cementum edges as a tuple.
        Additionally return the parameters of the fit function if specified

    """
    # Choose initial values if not specified
    if not initial_guess:
        initial_guess = [150, 18, 600, 22, 10, 85, 27, 2, 0.62]

    assert len(initial_guess) == 9, "Initial guess must be length 9"

    # independent variable is just the pixel number
    x = np.arange(straightened_img.shape[1])

    # Find the mean of each column
    y = np.mean(straightened_img, axis=0)

    # Fit
    params = _fit_curve(x, y, initial_guess)

    # Find the x-locations of the cementum edges
    a, d1, d2 = [params[i] for i in [0, 3, 6]]
    start, end = a + d1, a + d1 + d2

    if return_params:
        return (start, end), params

    return start, end


def _line(x, a, b):
    return a * x + b


def _reduced_chi2(y, y_fit, *, n_params):
    residuals = y - y_fit
    return np.sum(residuals**2) / (len(y) - n_params)


def fit_line(n_pixels: int, intensity: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Fit a line to the rightmost n_pixels in an intensity profile.

    :param n_pixels: number of pixels to fit
    :param intensity: intensity profile

    :return: fit params: (gradient, intercept) of the line
    :return: reduced chi-squared of the fit
    return: x-values of the fit

    """
    # Create an array of x-values
    x_vals = np.arange(len(intensity))[-n_pixels:]

    # Slice the y_values
    y_vals = intensity[-n_pixels:]

    # Fit a line
    params, _ = curve_fit(_line, x_vals, y_vals)

    # Find the reduced chi-squared
    chi2 = _reduced_chi2(y_vals, _line(x_vals, *params), n_params=2)

    # Return gradient + reduced chi2
    return params, chi2, x_vals


def line_with_bump(x, a, b, offset, h, s):
    """
    Line with a Gaussian bump at x=0

    """
    return _line(x, a, b) + h * norm(loc=offset + x[0], scale=s).pdf(x)


def fit_line_with_bump(
    n_pixels: int, intensity: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """
    Fit a line with a Gaussian bump at x=0 to the rightmost n_pixels in an intensity profile.

    """
    # Create an array of x-values
    x_vals = np.arange(len(intensity))[-n_pixels:]

    # Slice the y_values
    y_vals = intensity[-n_pixels:]

    # Fit a line
    params, _ = curve_fit(
        line_with_bump,
        x_vals,
        y_vals,
        p0=[0, 0, 10, 200, 1],
        bounds=[[-np.inf, -np.inf, 10, 150, 0], [np.inf, np.inf, 20, 500, 4]],
    )

    # Find the reduced chi-squared
    chi2 = _reduced_chi2(y_vals, line_with_bump(x_vals, *params), n_params=4)

    # Return gradient + reduced chi2
    return params, chi2, x_vals


def fit_line_restricted_domain(
    offset: int,
    intensity: np.ndarray,
    *,
    n_pixels: int = 50,
) -> tuple[np.ndarray, float, float]:
    """
    Fit a line to n_pixels of an image, offset from the right edge by offset_pixels

    :param n_pixels: number of pixels to fit
    :param intensity: intensity profile

    :return: fit params: (gradient, intercept) of the line
    :return: reduced chi-squared of the fit
    return: x-values of the fit

    """
    # The domain to use for the fit
    keep = slice(-(n_pixels + offset), -offset if offset else None)

    # Create an array of x-values
    x_vals = np.arange(len(intensity))[keep]

    # Slice the y_values
    y_vals = intensity[keep]

    # Fit a line
    params, _ = curve_fit(_line, x_vals, y_vals)

    # Find the reduced chi-squared
    chi2 = _reduced_chi2(y_vals, _line(x_vals, *params), n_params=2)

    # Return gradient + reduced chi2
    return params, chi2, x_vals


def fit_line_with_bump_restricted_domain(
    offset: int,
    intensity: np.ndarray,
    *,
    n_pixels: int = 50,
) -> tuple[np.ndarray, float, float]:
    """
    Fit a line to n_pixels of an image, offset from the right edge by offset_pixels

    :param n_pixels: number of pixels to fit
    :param intensity: intensity profile

    :return: fit params: (gradient, intercept) of the line
    :return: reduced chi-squared of the fit
    return: x-values of the fit

    """
    # The domain to use for the fit
    keep = slice(-(n_pixels + offset), -offset if offset else None)

    # Create an array of x-values
    x_vals = np.arange(len(intensity))[keep]

    # Slice the y_values
    y_vals = intensity[keep]

    # Fit a line
    try:
        params, _ = curve_fit(
            line_with_bump,
            x_vals,
            y_vals,
            p0=[0, 0, 10, 50, 1],
            bounds=[[-np.inf, -np.inf, -10, 0, 0], [np.inf, np.inf, 20, 500, 10]],
        )
    except RuntimeError as e:
        warnings.warn(util.coloured(e, util.bcolours.WARNING))
        return np.zeros(5), np.inf, x_vals

    # Find the reduced chi-squared
    chi2 = _reduced_chi2(y_vals, line_with_bump(x_vals, *params), n_params=4)

    # Return gradient + reduced chi2
    return params, chi2, x_vals


def find_cementum(
    left_boundaries: list[int],
    delta_chi2: list[float],
    *,
    tolerance: float = 5.0,
    rel_height: float = 0.95,
) -> int:
    """
    Find the approximate location of the cementum-dentine boundary

    Detects the last peak in deltachi2 and returns the appropriate pixel value

    :param left_boundaries: leftmost boundary of the fit region for each fit
    :param delta_chi2: difference in chi2 value for each fit
    :param tolerance: promience below which it isnt considered a peak
    :param rel_height: relative height of the peak as a fraction of its prominence

    :return: approximate pixel value of the cementum.

    """
    # Find the peaks
    peak_indices, _ = find_peaks(delta_chi2, height=tolerance)

    # Find the peak intersection points
    _, _, intersection, _ = peak_widths(delta_chi2, peak_indices, rel_height=rel_height)

    return left_boundaries[int(np.round(intersection[0]))]


def find_boundary(
    intensity: np.ndarray,
    *,
    domain_length: int,
    tolerance: float = 5.0,
    rel_height: float = 0.95,
    step_size: int = 3,
) -> int:
    """
    Given an array of average pixel intensities, find the location of the cementum-dentin boundary

    :param intensity: array of average pixel intensities
    :param domain_length: number of pixels in the fit domain
    :param tolerance: prominence below which it isnt considered a peak
    :param rel_height: relative height of the peak

    :returns: approximate pixel value of the cementum-dentin boundary

    """
    # Define an array of points to use as the starting x value in the sliding window
    fit_starts = np.arange(0, len(intensity) - 2 * domain_length, step_size)[::-1]

    # Define arrays for storing the chi2s
    line_chi2s = np.zeros_like(fit_starts)
    bump_chi2s = np.zeros_like(fit_starts)

    # Perform the fits
    for i, start in enumerate(fit_starts):
        _, line_chi2, _ = fit_line_restricted_domain(
            offset=start,
            intensity=intensity,
            n_pixels=domain_length,
        )
        _, bump_chi2, _ = fit_line_with_bump_restricted_domain(
            offset=start,
            intensity=intensity,
            n_pixels=domain_length,
        )

        line_chi2s[i] = line_chi2
        bump_chi2s[i] = bump_chi2

    # Find delta chi2
    delta_chi2 = line_chi2s - bump_chi2s

    # Find the location of the deltachi2 peak
    return find_cementum(
        fit_starts, delta_chi2, tolerance=tolerance, rel_height=rel_height
    )
