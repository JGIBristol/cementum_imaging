"""
Validation + correction for the cementum mask

"""

def n_contiguous_regions(mask: np.ndarray) -> int:
    """
    Count the number of contiguous regions in a mask

    """


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


def fill_right_bkg():
    """
    Fill background (1) pixels that are in a region on the right edge of the image with dentin (3)

    """
