# -*- coding: utf-8 -*-
"""Furness functions for distributing vectors to matrices."""
# Built-Ins
import logging
import warnings
from typing import Optional
from dataclasses import dataclass

# Third Party
import numpy as np
import pandas as pd
from caf.toolkit import translation

# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
# TODO(BT): Add 3D Furness from NorMITs Demand
@dataclass
class FurnessInputs:
    """
    Parameters
    ----------
    seed_vals: np.ndarray
        Initial values for the furness. This must be at the lower (less
        aggregate) zone system.
    row_targets: np.ndarray
        See doubly constrained furness.
    col_targets: np.ndarray
        See doubly constrained furness.
    tol:
        See doubly constrained furness
    max_iters:
        The max number of iterations for the outer process (i.e. furness at
        both levels and check convergence)
    warning:
        See doubly constrained furness
    """

    seed_vals: np.ndarray
    row_targets: np.ndarray
    col_targets: np.ndarray
    tol: float = 1e-9
    max_iters: int = 5000
    warning: bool = True


@dataclass
class SectoralConstraintInputs:
    """
    Input additional arguments for sectorial constraint.

    These are inputs not needed for a doubly constrained furness, as those will
    always be needed.

    Parameters
    ----------
    trans_vector: pd.DataFrame
        Vector used for translating between zones and sectors. This should be
        one way, from zone up to sector, and should be one to one (all factors
        equal to 1).
    from_col: str
        The name of the column in the translation vector containing zone
        ids for the original (i.e. less aggregate) zone system.
    to_col: str
        The name of the column in the translation vector containing zone
        ids for the sectoral (i.e. more aggregate) zone system.
    factor_col: str
        The name of the column in the translation vector containing factors
        for translation. This column should contain all ones, but is left
        to the user to provide as an extra check on inputs.
    target_mat: pd.DataFrame
        The matrix at sectoral level which should be adjusted to. Zone names
        here should match those in the translation vector.
    zonal_zones: Optional[collections.Collection] = None
        Zone names of the lower zone system. These must be in the correct order,
        and must match translation vector(s). If None is provided, this will
        to numbers from 1 to the length of the matrix
    outer_max_iters:
        Passed as max_iters when doubly_constrained_furness is called.
    furness_input: FurnessInputs
        Inputs for a doubly constrained furness.
    """

    trans_vector: pd.DataFrame
    from_col: str
    to_col: str
    factor_col: str
    target_mat: pd.DataFrame
    outer_max_iters: int = 10
    zonal_zones: Optional[np.ndarray] = None
    furness_inputs: Optional[FurnessInputs] = None


# # # FUNCTIONS # # #
def calc_rmse(col_targets, furnessed_mat, row_targets, n_vals: Optional[int] = None):
    """Calculate the RMSE for a matrix, compared to row and column targets."""
    if n_vals is None:
        n_vals = len(row_targets)
    row_diff = (row_targets - np.sum(furnessed_mat, axis=1)) ** 2
    col_diff = (col_targets - np.sum(furnessed_mat, axis=0)) ** 2
    rmse = ((np.sum(row_diff) + np.sum(col_diff)) / n_vals) ** 0.5
    return rmse


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
            cur_rmse = calc_rmse(col_targets, furnessed_mat, row_targets, n_vals)
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


def sectoral_constraint(inputs: SectoralConstraintInputs):
    """
    Furness with an extra constraint to match a matrix at a more aggregate zoning.

    Parameters
    ----------
    inputs: SectoralConstraintInputs
        See docstring of input class.
    Returns
    -------
    furnessed_matrix:
        The final furnessed matrix. This matrix will match 'sectoral_targets'
        precisely.

    completed_iters:
        The number of completed outer iterations - each iteration is a 2-d
         furness then an adjustment to the sectoral targets before exiting.

    achieved_rmse:
        The Root Mean Squared Error difference achieved before exiting to
        row and column targets.
    """
    finputs = inputs.furness_inputs
    if finputs is None:
        raise ValueError("Furness inputs must be provided." "")
    iter = 1
    seed_vals_inner = finputs.seed_vals.copy()
    n_vals = len(finputs.row_targets)
    if (inputs.trans_vector[inputs.factor_col] != 1).all():
        raise ValueError(
            "This process is designed to work with zones the nest "
            "perfectly within sectors. The translation vector provided "
            "implies this isn't the case. Either fix the translation  "
            "or reconsider using this function. "
        )
    if inputs.zonal_zones is None:
        inputs.zonal_zones = range(1, len(finputs.seed_vals) + 1)
    while True:
        furnessed, _, _ = doubly_constrained_furness(
            seed_vals_inner,
            finputs.row_targets,
            finputs.col_targets,
            finputs.tol,
            finputs.max_iters,
            finputs.warning,
        )
        trans_mat = pd.DataFrame(
            furnessed, index=inputs.zonal_zones, columns=inputs.zonal_zones
        )
        aggregated = translation.pandas_matrix_zone_translation(
            trans_mat, inputs.trans_vector, inputs.from_col, inputs.to_col, inputs.factor_col
        )
        adjustment_mat = translation.pandas_matrix_zone_translation(
            inputs.target_mat / aggregated,
            inputs.trans_vector,
            inputs.to_col,
            inputs.from_col,
            inputs.factor_col,
            check_totals=False,
        )
        adjusted = furnessed * adjustment_mat.to_numpy()
        rmse = calc_rmse(finputs.col_targets, adjusted, finputs.row_targets, n_vals)
        if rmse < finputs.tol:
            return adjusted, iter, rmse
        seed_vals_inner = adjusted
        iter += 1
        if iter > finputs.max_iters:
            warnings.warn(
                "Process has reached the max number of iterations "
                f"without converging. The RMSE is {rmse}. Returning "
                f"the matrix as it currently is"
            )
            return adjusted, iter, rmse
