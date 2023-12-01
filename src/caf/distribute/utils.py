# -*- coding: utf-8 -*-
"""
Module for miscellaneous utilities for the package
"""
# Built-Ins

# Third Party
import numpy as np
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #

# # # FUNCTIONS # # #
def infill_cost_matrix(cost_matrix: np.ndarray,
                       diag_factor: float = 0.5,
                       zeros_infill: float = 0.5) -> np.ndarray:
    """
    Infill the cost matrix before starting the gravity model.

    This function infills in two ways; firstly it infills the main diagonal (i.e.
    intrazonal costs) with the minimum value from each respective row multiplied
    by a factor, the logic being that an intrazonal trip is probably 50% (or
    whatever factor chosen) of the distance to the nearest zone. It also infills
    zeros in the matrix with a user defined factor to avoid errors in the seed matrix.

    Parameters
    ----------
    cost_matrix: The cost matrix. This should be a square array
    diag_factor: The factor the rows' minimum values will be multiplied by to
    infill intrazonal costs.
    zeros_infill: The infill value for other (none diagonal) zeros in the matrix

    Returns
    -------
    np.ndarray: The input matrix with values infilled.
    """
    # TODO add multipiple factors by area-type
    mins = cost_matrix.min(axis=1) * diag_factor
    np.fill_diagonal(cost_matrix, mins)
    cost_matrix[cost_matrix == 0] = zeros_infill
    return cost_matrix

