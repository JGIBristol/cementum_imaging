"""
Functions and helpers for straightening the images

"""
import numpy as np
from PIL import ImageFilter, Image


def find_edges(image: np.ndarray) -> np.ndarray:
    """
    Find edges in the image

    :param image: image to find edges in

    :returns: image with edges indicated

    """
    return Image.fromarray(image).filter(ImageFilter.FIND_EDGES)
