"""
Test the mask correction code

"""

import numpy as np
import pytest

from cementum import correct_mask


def test_n_contiguous_regions():
    # Mainline case
    mask = np.array(
        [
            [1, 2, 3, 3],
            [1, 2, 3, 3],
            [1, 2, 3, 3],
        ]
    )
    assert correct_mask.n_contiguous_regions(mask) == 3

    # Sometimes there's a small region of background that's not contiguous
    mask = np.array(
        [
            [1, 2, 3, 1],
            [1, 2, 3, 1],
            [1, 2, 3, 1],
        ]
    )
    assert correct_mask.n_contiguous_regions(mask) == 4

    # Check something weird
    mask = np.array(
        [
            [1, 2, 3, 1],
            [1, 1, 1, 1],
            [1, 2, 3, 1],
        ]
    )
    assert correct_mask.n_contiguous_regions(mask) == 5


def test_too_many_vals():
    """
    Check the right error gets raised if there are too many unique values in the mask

    """
    mask = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    )
    with pytest.raises(correct_mask.Not3RegionsError):
        correct_mask.check_mask(mask)


def test_noncontiguous():
    """
    Check that the right error gets raised if there are three noncontiguous regions

    """
    mask = np.array(
        [
            [1, 2, 3, 1],
            [1, 2, 3, 1],
            [1, 2, 3, 1],
        ]
    )
    with pytest.raises(correct_mask.NotContiguousError):
        correct_mask.check_mask(mask)


def test_fill_right_bkg():
    """
    Mainline case

    """
    mask = np.array(
        [
            [1, 2, 3, 1],
            [1, 2, 3, 1],
            [1, 2, 3, 1],
        ]
    )

    expected = np.array(
        [
            [1, 2, 3, 3],
            [1, 2, 3, 3],
            [1, 2, 3, 3],
        ]
    )

    assert np.all(correct_mask.fill_right_bkg(mask) == expected)


def test_fill_right_bkg_noop():
    """
    Check that the function returns the same thing if there is no background on the right

    """
    mask = np.array(
        [
            [1, 2, 3, 3],
            [1, 2, 3, 3],
            [1, 2, 3, 3],
        ]
    )

    assert np.all(correct_mask.correct_mask(mask) == mask)


def test_correct_mask_no_op():
    """
    Check that a valid mask doesn't change

    """
    mask = np.array(
        [
            [1, 2, 3, 3],
            [1, 2, 3, 3],
            [1, 2, 3, 3],
        ]
    )

    assert np.all(correct_mask.fill_right_bkg(mask) == mask)


def test_mask_dilation():
    """
    Check that a mask with a small background region between the cementum and dentin is dilated correctly

    """
    widths = [100, 50, 250]
    height = sum(widths)
    mask = np.array(
        [
            [
                *[1] * widths[0],
                *[2] * widths[1],
                *[3] * widths[2],
            ]
            for _ in range(height)
        ]
    )

    # Add a thin region between cementum and dentin
    mask[:, widths[0] + widths[1]] = 1

    # Ensure that the right error gets raised for the mask as-is
    with pytest.raises(correct_mask.NotContiguousError):
        correct_mask.check_mask(mask)

    # Correct it
    corrected = correct_mask.correct_mask(mask)
    correct_mask.check_mask(corrected)


def test_mask_right_bkg():
    """
    Check that a mask with a background region on the right is correctly replaced with dentin

    """
    widths = [100, 50, 250]
    height = sum(widths)
    mask = np.array(
        [
            [
                *[1] * widths[0],
                *[2] * widths[1],
                *[3] * widths[2],
            ]
            for _ in range(height)
        ]
    )

    # Add a thin region  on the right
    mask[:, -1] = 1

    # Ensure that the right error gets raised for the mask as-is
    with pytest.raises(correct_mask.NotContiguousError):
        correct_mask.check_mask(mask)

    # Correct it
    corrected = correct_mask.correct_mask(mask)
    correct_mask.check_mask(corrected)


def test_dilated_and_right():
    """
    Check that a mask both a background region on the right and a small region between cementum and dentin is correctly filled

    """
    widths = [100, 50, 250]
    height = sum(widths)
    mask = np.array(
        [
            [
                *[1] * widths[0],
                *[2] * widths[1],
                *[3] * widths[2],
            ]
            for _ in range(height)
        ]
    )

    # Add a thin region  on the right
    mask[:, -1] = 1

    # Add a thin region between cementum and dentin
    mask[:, widths[0] + widths[1]] = 1

    # Ensure that the right error gets raised for the mask as-is
    with pytest.raises(correct_mask.NotContiguousError):
        correct_mask.check_mask(mask)

    # Correct it
    corrected = correct_mask.correct_mask(mask)
    correct_mask.check_mask(corrected)


def test_four_regions():
    """
    If the mask somehow has four regions, check that the right error is raised

    """
    mask = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    )
    with pytest.raises(correct_mask.Not3RegionsError):
        correct_mask.correct_mask(mask)
