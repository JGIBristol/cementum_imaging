"""
Test the mask correction code

"""

import numpy as np
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
