# -*- coding: utf-8 -*-
"""Module for miscellaneous utilities for the package."""
# Built-Ins
import functools
from typing import Literal

# Third Party
import numpy as np
import pandas as pd

# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
def infill_cost_matrix(
    cost_matrix: np.ndarray, diag_factor: float = 0.5, zeros_infill: float = 0.5
) -> np.ndarray:
    """
    Infill the cost matrix before starting the gravity model.

    This function infills in two ways; firstly it infills the main diagonal (i.e.
    intrazonal costs) with the minimum value from each respective row multiplied
    by a factor, the logic being that an intrazonal trip is probably 50% (or
    whatever factor chosen) of the distance to the nearest zone. It also infills
    zeros in the matrix with a user defined value to avoid errors in the seed matrix.

    Parameters
    ----------
    cost_matrix: The cost matrix. This should be a square array
    diag_factor: The factor the rows' minimum values will be multiplied by to
    infill intrazonal costs.
    zeros_infill: The infill value for other (non-diagonal) zeros in the matrix

    Returns
    -------
    np.ndarray: The input matrix with values infilled.
    """
    # TODO(IS) allow infilling diagonals only where zero
    min_row = np.min(np.ma.masked_where(cost_matrix <= 0, cost_matrix), axis=1) * diag_factor

    np.fill_diagonal(cost_matrix, min_row)
    # Needed due to a bug in np.fill_diagonal
    cost_matrix[cost_matrix > 1e10] = zeros_infill
    cost_matrix[cost_matrix == 0] = zeros_infill
    return cost_matrix


def validate_zones(
    trip_ends: pd.DataFrame,
    costs: pd.DataFrame,
    costs_format: Literal["long", "wide"],
    tld_lookup: pd.DataFrame,
):
    """
    Validate inputs to a multi area gravity model.

    This checks that the zones are identical for each. It is assumed the zones
    form the index of each of these, and the columns of costs if in wide
    format. There is no return from this function if the zones do match, only
    an error raised if they don't.
    """
    # TODO(IS) add tests for this
    if costs_format == "long":
        orig_zones = costs.index.get_level_values[0].values
        dest_zones = costs.index.get_level_values[1].values
    elif costs_format == "wide":
        orig_zones = costs.index.values
        dest_zones = costs.columns.values
    else:
        raise ValueError(
            "costs_format must be either wide, if costs is "
            "given as a wide matrix, or long if costs is given "
            "as a long matrix."
        )
    zones_list = [orig_zones, dest_zones, trip_ends.index.values, tld_lookup.index.values]
    check = functools.reduce(np.array_equal, zones_list)
    if not check:
        raise ValueError(
            "The zones do not match for all of these. It is "
            "assumed the zones are contained in rows/indices,"
            "so if that is not the case that may be why this "
            "error has been raised."
        )
