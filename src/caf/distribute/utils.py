# -*- coding: utf-8 -*-
"""
Module for miscellaneous utilities for the package
"""
# Built-Ins
from typing import Literal
import functools

# Third Party
import numpy as np
import pandas as pd
from caf.toolkit import cost_utils
from caf.distribute.gravity_model import multi_area

# Local Imports
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
    # TODO allow infilling diagonals only where zero
    min_row = np.min(np.ma.masked_where(cost_matrix <= 0, cost_matrix), axis=1) * diag_factor

    np.fill_diagonal(cost_matrix, min_row)
    cost_matrix[cost_matrix > 1e10] = zeros_infill
    cost_matrix[cost_matrix == 0] = zeros_infill
    return cost_matrix


def process_tlds(
    tlds: pd.DataFrame,
    cat_col: str,
    min_col: str,
    max_col: str,
    ave_col: str,
    trips_col: str,
    tld_lookup: pd.DataFrame,
    lookup_cat_col: str,
    lookup_zone_col: str,
    function_params: dict[str, float],
) -> list[multi_area.MultiCostDistribution]:
    """
    Read in a dataframe of distributions by category and a lookup, and return
    a list of distributions ready to be passed to a multi area gravity model.
    """
    tlds = tlds.set_index(cat_col)
    tld_lookup = tld_lookup.sort_values(lookup_zone_col)
    dists = []
    for cat in tlds.index.unique():
        tld = tlds.loc[cat]
        tld = cost_utils.CostDistribution(
            tld, min_col=min_col, max_col=max_col, avg_col=ave_col, trips_col=trips_col
        )
        zones = tld_lookup[tld_lookup[lookup_cat_col] == cat].index.values
        if len(zones) == 0:
            raise ValueError(
                f"{cat} doesn't seem to appear in the given tld "
                "lookup. Check for any typos (e.g. lower/upper case). "
                f"If this is expected, remove {cat} from your "
                "tlds dataframe before inputting."
            )
        distribution = multi_area.MultiCostDistribution(
            name=cat, cost_distribution=tld, zones=zones, function_params=function_params
        )
        dists.append(distribution)

    return dists


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
