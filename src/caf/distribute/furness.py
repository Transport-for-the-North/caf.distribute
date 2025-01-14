# -*- coding: utf-8 -*-
"""Furness functions for distributing vectors to matrices."""
# Built-Ins
import logging
import warnings

# Third Party
import numpy as np
from numpy.testing import assert_approx_equal
import pandas as pd
import operator
from caf.toolkit import pandas_utils as pd_utils

# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #
# TODO(BT): Add 3D Furness from NorMITs Demand


# # # FUNCTIONS # # #
# TODO(BT): Add a pandas wrapper to doubly_constrained_furness()


def doubly_constrained_furness(
    seed_vals: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    tol: float = 1e-9,
    max_iters: int = 5000,
    warning: bool = True,
) -> tuple[np.ndarray, int, float]:
    """
    Perform a doubly constrained furness for max_iters or until tol is met.

    Controls numpy warnings to warn of any overflow errors encountered

    Parameters
    ----------
    seed_vals:
        Initial values for the furness. Must be of shape
        (len(n_rows), len(n_cols)).

    row_targets:
        The target values for the sum of each row.
        i.e np.sum(matrix, axis=1)

    col_targets:
        The target values for the sum of each column
        i.e np.sum(matrix, axis=0)

    tol:
        The maximum difference between the achieved and the target values
        to tolerate before exiting early. R^2 is used to calculate the
        difference.

    max_iters:
        The maximum number of iterations to complete before exiting.

    warning:
        Whether to print a warning or not when the tol cannot be met before
        max_iters.

    Returns
    -------
    furnessed_matrix:
        The final furnessed matrix

    completed_iters:
        The number of completed iterations before exiting

    achieved_rmse:
        The Root Mean Squared Error difference achieved before exiting
    """
    # pylint: disable=too-many-locals
    # TODO(MB) Incorporate Nhan's furnessing optimisations
    # Error check
    if seed_vals.shape != (len(row_targets), len(col_targets)):
        raise ValueError(
            f"The shape of the seed values given does not match the row and "
            f"col targets. Seed_vals are shape {str(seed_vals.shape)}. "
            f"Expected shape ({len(row_targets):d}, {len(col_targets):d})."
        )

    if np.any(np.isnan(row_targets)) or np.any(np.isnan(col_targets)):
        raise ValueError("np.nan found in the targets. Cannot run.")

    # Need to ensure furnessed mat is floating to avoid numpy casting
    # errors in loop
    furnessed_mat = seed_vals.copy()
    if np.issubdtype(furnessed_mat.dtype, np.integer):
        furnessed_mat = furnessed_mat.astype(float)

    # Init loop
    early_exit = False
    cur_rmse = np.inf
    iter_num = 0
    n_vals = len(row_targets)

    # Can return early if all 0 - probably shouldn't happen!
    if row_targets.sum() == 0 or col_targets.sum() == 0:
        warnings.warn("Furness given targets of 0. Returning all 0's")
        return np.zeros_like(seed_vals), iter_num, np.inf

    # Set up numpy overflow errors
    with np.errstate(over="raise"):
        for iter_num in range(max_iters):
            # ## COL CONSTRAIN ## #
            # Calculate difference factor
            col_ach = np.sum(furnessed_mat, axis=0)
            diff_factor = np.divide(
                col_targets,
                col_ach,
                where=col_ach != 0,
                out=np.ones_like(col_targets, dtype=float),
            )

            # adjust cols
            furnessed_mat *= diff_factor

            # ## ROW CONSTRAIN ## #
            # Calculate difference factor
            row_ach = np.sum(furnessed_mat, axis=1)
            diff_factor = np.divide(
                row_targets,
                row_ach,
                where=row_ach != 0,
                out=np.ones_like(row_targets, dtype=float),
            )

            # adjust rows
            furnessed_mat *= np.atleast_2d(diff_factor).T

            # Calculate the diff - leave early if met
            row_diff = (row_targets - np.sum(furnessed_mat, axis=1)) ** 2
            col_diff = (col_targets - np.sum(furnessed_mat, axis=0)) ** 2
            cur_rmse = ((np.sum(row_diff) + np.sum(col_diff)) / n_vals) ** 0.5
            if cur_rmse < tol:
                early_exit = True
                break

            # We got a NaN! Make sure to point out we didn't converge
            if np.isnan(cur_rmse):
                warnings.warn(
                    "np.nan value found in the rmse calculation. It must have "
                    "been introduced during the furness process."
                )
                return np.zeros(furnessed_mat.shape), iter_num, np.inf

    # Warn the user if we exhausted our number of loops
    if not early_exit and warning:
        warnings.warn(
            f"The doubly constrained furness exhausted its max "
            f"number of loops ({max_iters:d}), while achieving an RMSE "
            f"difference of {cur_rmse:f}. The values returned may not be "
            f"accurate."
        )

    return furnessed_mat, iter_num + 1, cur_rmse


def furness_pandas_wrapper(
    seed_values: pd.DataFrame,
    row_targets: pd.DataFrame,
    col_targets: pd.DataFrame,
    max_iters: int = 2000,
    seed_infill: float = 1e-3,
    normalise_seeds: bool = True,
    tol: float = 1e-9,
    idx_col: str = "model_zone_id",
    unique_col: str = "trips",
    round_dp: int = 8,
    unique_zones: list[int] = None,
    unique_zones_join_fn: callable = operator.and_,
) -> tuple[pd.DataFrame, int, float]:
    """
    Wrapper around doubly_constrained_furness() to handle pandas in/out

    Internally checks and converts the pandas inputs into numpy in order to
    run doubly_constrained_furness(). Converts the output back into pandas
    at the end

    Parameters
    ----------
    seed_values:
        The seed values to use for the furness. The index and columns must
        match the idx_col of row_targets and col_targets.

    row_targets:
        The target values for the sum of each row. In production/attraction
        furnessing, this would be the productions. The idx_col must match
        the idx_col of col_targets.

    col_targets:
        The target values for the sum of each column. In production/attraction
        furnessing, this would be the attractions. The idx_col must match
        the idx_col of row_targets.

    max_iters:
        The maximum number of iterations to complete before exiting.

    tol:
        The maximum difference between the achieved and the target values
        to tolerate before exiting early. R^2 is used to calculate the
        difference.

    seed_infill:
        The value to infill any seed values that are 0.

    normalise_seeds:
        Whether to normalise the seeds so they total to one before
        sending them to the furness.

    idx_col:
        Name of the columns in row_targets and col_targets that contain the
        index data that matches seed_values index/columns

    unique_col:
        Name of the columns in row_targets and col_targets that contain the
        values to target during the furness.

    round_dp:
        The number of decimal places to round the output values of the
        furness to. Uses 4 by default.

    unique_zones:
        A list of unique zones to keep in the seed matrix when starting the
        furness. The given productions and attractions will also be limited
        to these zones as well.

    unique_zones_join_fn:
        The function to call on the column and index masks to join them for
        the seed matrices. By default, a bitwise and is used. See pythons
        builtin operator library for more options.

    Returns
    -------
    furnessed_matrix:
        The final furnessed matrix, in the same format as seed_values

    completed_iters:
        The number of completed iterations before exiting

    achieved_rmse:
        The Root Mean Squared Error difference achieved before exiting
    """
    # Init
    row_targets = row_targets.copy()
    col_targets = col_targets.copy()
    seed_values = seed_values.copy()

    row_targets = row_targets.reindex(columns=[idx_col, unique_col])
    col_targets = col_targets.reindex(columns=[idx_col, unique_col])
    row_targets = row_targets.set_index(idx_col)
    col_targets = col_targets.set_index(idx_col)

    # ## VALIDATE INPUTS ## #
    ref_index = row_targets.index
    if len(ref_index.difference(col_targets.index)) > 0:
        raise ValueError("Row and Column target indexes do not match.")

    if len(ref_index.difference(seed_values.index)) > 0:
        raise ValueError("Row and Column target indexes do not match " "seed index.")

    if len(ref_index.difference(seed_values.columns)) > 0:
        raise ValueError("Row and Column target indexes do not match " "seed columns.")

    assert_approx_equal(
        row_targets[unique_col].sum(),
        col_targets[unique_col].sum(),
        err_msg="Row and Column target totals do not match. Cannot Furness.",
    )

    # ## TIDY AND INFILL SEED ## #
    # Infill the 0 zones
    seed_values = seed_values.mask(seed_values <= 0, seed_infill)
    if normalise_seeds:
        seed_values /= seed_values.sum()

    # If we were given certain zones, make sure everything else is 0
    if unique_zones is not None:
        # Get the mask and extract the data
        mask = pd_utils.get_wide_mask(
            df=seed_values,
            select=unique_zones,
            join_fn=unique_zones_join_fn,
        )
        seed_values = seed_values.where(mask, 0)

    # ## CONVERT TO NUMPY AND FURNESS ## #
    row_targets = row_targets.values.flatten()
    col_targets = col_targets.values.flatten()
    seed_values = seed_values.values

    furnessed_mat, n_iters, achieved_rmse = doubly_constrained_furness(
        seed_vals=seed_values,
        row_targets=row_targets,
        col_targets=col_targets,
        tol=tol,
        max_iters=max_iters,
    )

    furnessed_mat = np.round(furnessed_mat, round_dp)

    # ## STICK BACK INTO PANDAS ## #
    furnessed_mat = pd.DataFrame(index=ref_index, columns=ref_index, data=furnessed_mat).round(
        round_dp
    )

    return furnessed_mat, n_iters, achieved_rmse
