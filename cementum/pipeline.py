"""
Putting things together

"""
from dataclasses import dataclass

import matplotlib.pyplot as plt

from . import straighten


@dataclass
class Params:
    """
    Parameters used in the cementum straightening pipeline

    """

    n_y: int
    n_x: tuple[int, int, int]
    straighten_order: int


def count_layers(image_path: str, mask_path: str, params: Params) -> list[int]:
    """
    Count the number of layers in each row in an image

    """
    # Load the image and mask
    raw_image = plt.imread(image_path)
    raw_mask = plt.imread(mask_path)

    # Straighten the image
    curved_mesh = straighten.mask_mesh(raw_mask, params.n_y, params.n_x)
    straight_mesh = straighten.straighten_mesh(raw_mask, params.n_y, params.n_x)
    straight_image = straighten.apply_transformation(
        raw_image, curved_mesh, straight_mesh, order=params.straighten_order
    )

    # Cut off white columns and black rows
    # Crop out cementum
    # Contrast normalisation
    # Ridge detection
    # Count layers
