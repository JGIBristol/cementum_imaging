"""
Classical segementation of the cementum layers

"""
import numpy as np


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
