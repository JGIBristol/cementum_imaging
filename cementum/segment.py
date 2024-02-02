"""
Layer segmentation tools

"""
import numpy as np

from scipy.signal import find_peaks
from skimage.filters import farid_v, sato


def filter(image: np.ndarray, *, ridge_threshold: float = 3.0) -> np.ndarray:
    """
    Apply a Sato filter, threshold and a farid_v filter to locate ridges in an image

    :param image: image of straightened, cropped cementum to find layers in
    :param ridge_threshold: threshold for the Sato filter

    :returns: an image with ridges highlighted as peaks and troughs in intensity

    """
    # Apply a Sato filter and threshold
    filtered = sato(image, sigmas=range(1, 5, 2))
    is_ridge = filtered > ridge_threshold
    filtered[is_ridge] = 0
    filtered[~is_ridge] = 255

    # Apply a Farid filter
    return farid_v(filtered)


def layer_locations(
    filtered_image: np.ndarray, *, height: float = 35.0
) -> list[np.ndarray]:
    """
    Find the pixel locations of layers in each row of an image

    :param image: image of straightened, cropped cementum with ridges highlighted
    :paeam height: threshold height of peaks

    :returns: list of pixel locations of layers in each row of the image

    """
    # Detect peaks in each row
    peaks = []
    for row in filtered_image:
        peaks.append(find_peaks(row, height=height)[0])

    return peaks
