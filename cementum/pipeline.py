"""
Putting things together

"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
from skimage.filters import gaussian

from . import straighten, fit_cementum, filter


@dataclass
class Params:
    """
    Parameters used in the cementum straightening pipeline

    """

    # Straightening
    n_y: int
    n_x: tuple[int, int, int]
    straighten_order: int

    # Cementum detection
    domain_length: int

    # Contrast adjustment
    filter_angle: float = 0.0
    filter_sigma: float
    blur_sigma: float


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
    straight_image, _ = straighten.remove_white_cols(straight_image, straight_image)
    straight_image = straighten.remove_padding(straight_image)

    # Crop out cementum
    col_intensity = np.mean(straight_image, axis=0)
    dentin = fit_cementum.find_boundary(col_intensity, params.domain_length)
    background = fit_cementum.find_background(col_intensity)
    cropped_image = straight_image[:, background:dentin]

    # Contrast normalisation
    blurred_image = gaussian(
        filter.steerable_filter(
            cropped_image, theta=params.filter_angle, sigma=params.filter_sigma
        ),
        sigma=params.blur_sigma,
    )

    # Ridge detection
    # Count layers
