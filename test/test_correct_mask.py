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
