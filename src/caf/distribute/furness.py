# -*- coding: utf-8 -*-
"""Furness functions for distributing vectors to matrices."""
# Built-Ins
import logging
import warnings
from dataclasses import dataclass

# Third Party
import numpy as np
import pandas as pd

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


@dataclass
class props_input:
    """
    props: np.ndarray
        This is essentially a cost matrix, but costs are replaced by the percentage
        from the given cost band.
    zones: np.ndarray
        The zones in your zone system this prop matrix applies to
    prop_vals: np.ndarray
        The unique values in props
    """
    props: np.ndarray
    zones: np.ndarray
    prop_vals: np.ndarray


def cost_to_prop(costs: np.ndarray, bands: pd.DataFrame, val_col: str):
    """
    Convert a cost matrix and cost bands into proportions expected.

    Parameters
    ----------
    costs: np.ndarray
        The costs matrix. If this is being run for a multi-area distribution,
        only include costs for the bands applicable.
    bands: pd.DataFrame
        The cost bands for the given costs.
    val_col: str
        The column name of the values in the bands DataFrame.
    """
    bands_sum = bands[val_col].sum()
    bands[val_col] /= bands_sum
    bands_array = bands.values
    band_indices = np.zeros_like(costs, dtype=float)
    for band_start, band_end, prop in bands_array:
        band_mask = (costs >= band_start) & (costs <= band_end)
        band_indices[band_mask] = prop

    band_indices[band_indices == 0] = bands[val_col].min() * 0.5
    return band_indices, bands[val_col].values


def triply_constrained_furness(
    props: list[props_input],
    row_targets,
    col_targets,
    max_iters,
    mat_size: tuple[int, int],
    tol=1e-5,
):
    """
    Furness a seed matrix with triple constraints.

    Furnesses with the matrix tightly constrained to origin and destination
    targets (with rmse checked against tolerance), and loosely constrained
    to cost band shares, i.e. constrained in each furness iteration, but no
    exit criteria for this constraint. This theoretically gives good flexibilty,
    where the furness should achieve good convergence on all three constraints
    where possible, and achieve the first two where the third may not be possible.

    Parameters
    ----------
    props: list[props_input]
        A list of info about cost bins. This is produced by cost_to_props
    row_targets: np.ndarray
        The targets for the rows (origins) in the matrix
    col_targets: np.ndarray
        The targets for the cols (destinations) in the matrix
    max_iters: int
        Max iterations for the furness to run before exiting
    mat_size: tuplr[int, int]
        The size of the matrix being furnessed
    tol: float
        The convergence criteria
    """
    early_exit = False
    cur_rmse = np.inf
    iter_num = 0
    n_vals = len(row_targets)

    # build seed
    furnessed_mat = np.zeros(mat_size)
    for distro in props:
        furnessed_mat[distro.zones] = distro.props

    # one row, col furness loop outside iterations
    row_ach = np.sum(furnessed_mat, axis=1)
    diff_factor = np.divide(
        row_targets, row_ach, where=row_ach != 0, out=np.ones_like(row_targets, dtype=float)
    )

    furnessed_mat = np.multiply(furnessed_mat.T, diff_factor).T
    # Adjust cols
    col_ach = np.sum(furnessed_mat, axis=0)
    diff_factor = np.divide(
        col_targets,
        col_ach,
        where=col_ach != 0,
        out=np.ones_like(col_targets, dtype=float),
    )
    furnessed_mat *= diff_factor
    # Can return early if all 0 - probably shouldn't happen!
    if row_targets.sum() == 0 or col_targets.sum() == 0:
        warnings.warn("Furness given targets of 0. Returning all 0's")
        return np.zeros_like(props), iter_num, np.inf
    for iter_num in range(1, max_iters):
        # first adjust to match cost bands; this is the 'third' constraint but
        # is done first as the other two need to be matched more closely
        for distro in props:
            to_alter = furnessed_mat[distro.zones]
            checker = {}
            for i in distro.prop_vals:
                tot_demand = to_alter[distro.props == i].sum()
                checker[i] = tot_demand
            df = pd.DataFrame.from_dict(checker, orient="index").reset_index()
            df.columns = ["target_prop", "demand"]
            df["act_prop"] = df["demand"] / df["demand"].sum()
            df["adj"] = df["target_prop"] / df["act_prop"]
            df.fillna(0, inplace=True)
            df.set_index("target_prop", inplace=True)
            for i in df.index:
                to_alter[distro.props == i] *= df.loc[i, "adj"]
            furnessed_mat[distro.zones] = to_alter
        # Adjust rows
        row_ach = np.sum(furnessed_mat, axis=1)
        diff_factor = np.divide(
            row_targets,
            row_ach,
            where=row_ach != 0,
            out=np.ones_like(row_targets, dtype=float),
        )

        furnessed_mat = np.multiply(furnessed_mat.T, diff_factor).T
        # Adjust cols
        col_ach = np.sum(furnessed_mat, axis=0)
        diff_factor = np.divide(
            col_targets,
            col_ach,
            where=col_ach != 0,
            out=np.ones_like(col_targets, dtype=float),
        )
        furnessed_mat *= diff_factor

        row_diff = (row_targets - np.sum(furnessed_mat, axis=1)) ** 2
        col_diff = (col_targets - np.sum(furnessed_mat, axis=0)) ** 2
        cur_rmse = ((np.sum(row_diff) + np.sum(col_diff)) / n_vals) ** 0.5
        if cur_rmse < tol:
            early_exit = True
            break
        if early_exit is not True:
            warnings.warn(
                f"The doubly constrained furness exhausted its max "
                f"number of loops ({max_iters:d}), while achieving an RMSE "
                f"difference of {cur_rmse:f}. The values returned may not be "
                f"accurate."
            )

    return furnessed_mat
