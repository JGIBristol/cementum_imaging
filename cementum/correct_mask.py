"""
Validation + correction for the cementum mask

It should  be three contiguous regions (background, cementum, dentin) - sometimes this isn't the case

"""

import logging

import numpy as np
from scipy.ndimage import label as scipy_label, binary_dilation


class InvalidMaskError(Exception):
    """
    Base class for mask validation errors

    """

    pass


class Not3RegionsError(InvalidMaskError):
    """
    More than 3 unique values in the mask

    """

    def __init__(self, n_regions: int):
        super().__init__(
            f"Mask should have exactly three unique values, not {n_regions}"
        )


class NotContiguousError(InvalidMaskError):
    """
    Some regions in the mask are not contiguous

    """

    def __init__(self, message="All regions in the mask should be contiguous"):
        super().__init__(message)


def n_contiguous_regions(mask: np.ndarray) -> int:
    """
    Count the number of contiguous regions in a mask

    """
    unique_values = np.unique(mask)
    total_regions = 0
    for value in unique_values:
        _, num_labels = scipy_label(mask == value)
        total_regions += num_labels
    return total_regions


def check_mask(mask: np.ndarray) -> None:
    """
    Check whether a mask is valid

    It should have three unique regions (background, cementum, dentin) that are all contiguous

    :param mask: mask labelling background, cementum, dentin

    """
    # Check that we have exactly three values in our mask
    n_unique = len(np.unique(mask.flat))
    if n_unique != 3:
        raise Not3RegionsError(n_unique)

    # Check that the three regions are all contiguous
    if n_contiguous_regions(mask) != 3:
        raise NotContiguousError


def dilate_mask(mask: np.ndarray, *, kernel_size: int = 5) -> np.ndarray:
    """
    Dilate a mask to ensure that the regions are contiguous.

    :param mask: The mask to dilate
    :param kernel: The kernel to use for dilation. Defaults to 5x5

    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Find non-background regions
    cementum_dilated = binary_dilation(mask == 2, structure=kernel)
    dentin_dilated = binary_dilation(mask == 3, structure=kernel)

    # Dilate the regions + replace the original ones
    retval = mask.copy()
    retval = np.where(cementum_dilated, 2, retval)
    retval = np.where(dentin_dilated, 3, retval)

    return retval


def bkg_on_right(mask: np.ndarray) -> bool:
    """
    Check if there are any background (1) pixels in the rightmost column of a mask

    """
    return 1 in mask[:, -1]


def fill_right_bkg(mask: np.ndarray) -> np.ndarray:
    """
    Fill background (1) pixels that are in a region on the right edge of the image with dentin (3)

    """
    # Label connected background components
    labeled_bkg, _ = scipy_label(mask == 1)

    # Find the highest pixel x-value in the actual background region
    max_bkg = np.max(np.where(labeled_bkg == 1)[1])

    # Find which pixels are to the right of this
    right_of_bkg = np.indices(mask.shape)[1] > max_bkg

    # Find which pixels are in a contiguous region that touches the right edge
    right_regions = np.unique(labeled_bkg[:, -1])

    # Fill these pixels with dentin (3)
    copy = mask.copy()
    copy[right_of_bkg & (mask == 1) & np.isin(labeled_bkg, right_regions)] = 3

    return copy


def correct_mask(
    mask: np.ndarray, *, kernel_size: int = 5, verbose: bool = False
) -> np.ndarray:
    """
    Validate + correct the mask

    :param mask: The mask to validate and possibly correct

    """
    try:
        check_mask(mask)

        if verbose:
            logging.info("Mask is valid")

    except InvalidMaskError:
        # Binary dilation
        mask = dilate_mask(mask, kernel_size=kernel_size)

        # Fill any region on the right
        if bkg_on_right(mask):
            mask = fill_right_bkg(mask)

    try:
        check_mask(mask)
    except InvalidMaskError as e:
        logging.fatal(f"Mask is still invalid after correction: {e}")
        raise e

    return mask
