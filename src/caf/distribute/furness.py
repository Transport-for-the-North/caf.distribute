# -*- coding: utf-8 -*-
"""Furness functions for distributing vectors to matrices."""
# Built-Ins
import logging
import warnings
from typing import Optional
from dataclasses import dataclass
from copy import deepcopy

# Third Party
import numpy as np
import pandas as pd
from caf.toolkit import translation
import xarray as xr

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
    furness_inputs: FurnessInputs
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


@dataclass
class PropsInput:
    """
    Input to triply constrained furness.
    This is returned from cost_to_prop
    Parameters
    ----------
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
    bands.loc[:, val_col] /= bands_sum
    bands_array = bands.values
    band_indices = np.zeros_like(costs, dtype=float)
    for band_start, band_end, prop in bands_array:
        band_mask = (costs >= band_start) & (costs <= band_end)
        band_indices[band_mask] = prop

    band_indices[band_indices == 0] = bands[val_col].min() * 0.5
    return band_indices, bands[val_col].values


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


def dist_match(props: dict[int, PropsInput], mat: np.ndarray, return_checkers=False):
    checkers = {}
    for area, distro in props.items():
        to_alter = mat[distro.zones]
        checker = {}
        for i in distro.prop_vals:
            tot_demand = to_alter[distro.props == i].sum()
            checker[i] = tot_demand
        # pylint: disable=unsupported-assignment-operation, unsubscriptable-object
        # pylint error
        checker_df = pd.DataFrame.from_dict(checker, orient="index").reset_index()
        checker_df.columns = ["target_prop", "demand"]
        checker_df["act_prop"] = checker_df["demand"] / checker_df["demand"].sum()
        checker_df["adj"] = (checker_df["target_prop"] / checker_df["act_prop"]).replace(0, 1)
        checker_df.fillna(0, inplace=True)
        checker_df.set_index("target_prop", inplace=True)

        for i in checker_df.index:
            to_alter[distro.props == i] *= checker_df.loc[i, "adj"]
        mat[distro.zones] = to_alter
        checkers[area] = checker_df
    if return_checkers:
        return pd.concat(checkers), mat
    return mat


def triply_constrained_furness(
    props: list[PropsInput],
    row_targets,
    col_targets,
    max_iters,
    mat_size: tuple[int, int],
    init_mat: Optional[np.ndarray] = None,
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
    props: list[PropsInput]
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
    checker_df = pd.DataFrame({"adj": np.inf}, index=[0])
    if init_mat is None:
        # build seed
        furnessed_mat = np.zeros(mat_size)
        for distro in props:
            furnessed_mat[distro.zones] = distro.props

        furnessed_mat, _, _ = doubly_constrained_furness(
            furnessed_mat, row_targets, col_targets, max_iters=10, warning=False
        )
    else:
        furnessed_mat = deepcopy(init_mat)
    for iter_num in range(1, max_iters):
        # first adjust to match cost bands; this is the 'third' constraint but
        # is done first as the other two need to be matched more closely
        if (checker_df["adj"] - 1).abs().max() > 5e-1:
            dist_match(props, furnessed_mat)
            # pylint:enable=unsupported-assignment-operation, unsubscriptable-object
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
            print(f"Converged in {iter_num} iterations with rmse of {cur_rmse}")
            early_exit = True
            break
    if (not early_exit) & (iter_num >= max_iters):
        warnings.warn(
            f"The triply constrained furness exhausted its max "
            f"number of loops ({max_iters:d}), while achieving an RMSE "
            f"difference of {cur_rmse:f}. The values returned may not be "
            f"accurate."
        )

    return furnessed_mat


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
        raise ValueError("Furness inputs must be provided.")
    iter = 1
    seed_vals_inner = finputs.seed_vals.copy()
    # seed_vals_inner[np.where(inputs.target_mat == 0)] = 0
    zonal_excl = [i for i in inputs.zonal_zones if i not in inputs.trans_vector[inputs.to_col]]
    sectoral_excl = [
        i for i in inputs.trans_vector[inputs.to_col] if i not in inputs.zonal_zones
    ]
    n_vals = len(finputs.row_targets)
    if (inputs.trans_vector[inputs.factor_col] != 1).all():
        raise ValueError(
            "This process is designed to work with zones that nest "
            "perfectly within sectors. The translation vector provided "
            "implies this isn't the case. Either fix the translation "
            "or reconsider using this function."
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
        # row_diff = aggregated.sum(axis=1) / inputs.target_mat.sum(axis=1) - 1
        # col_diff = aggregated.sum(axis=0) / inputs.target_mat.sum(axis=0) - 1
        # if (np.absolute(row_diff).max() > 0.01) | (np.absolute(col_diff).max() > 0.01):
        #     raise ValueError("The furnessed matrix aggregated up to sectoral level "
        #                      "does not match the target matrix. Check your translation, "
        #                      "zonal trip ends and target matrix.")

        # This is for factors, so everything is multiplied by one to match
        # sectoral factors to zones
        adjustment_mat = translation.pandas_matrix_zone_translation(
            pd.DataFrame(inputs.target_mat, index=aggregated.index, columns=aggregated.columns)
            .div(aggregated)
            .fillna(1)
            .replace(to_replace={np.inf: 1}),
            inputs.trans_vector,
            inputs.to_col,
            inputs.from_col,
            inputs.factor_col,
            check_totals=False,
        )
        adjusted = furnessed * adjustment_mat
        # Check rmse compared to zonal targets after sectoral adjustment
        rmse = calc_rmse(finputs.col_targets, adjusted.values, finputs.row_targets, n_vals)
        if rmse < finputs.tol:
            return adjusted.to_numpy(), iter, rmse
        seed_vals_inner = adjusted.to_numpy()
        iter += 1
        if iter > finputs.max_iters:
            warnings.warn(
                "Process has reached the max number of iterations "
                f"without converging. The RMSE is {rmse}. Returning "
                f"the matrix as it currently is"
            )
            return adjusted.to_numpy(), iter, rmse


@dataclass
class SegInput:
    """
    props: Instance of PropsInput used for the tld aspect of segmentation_furness
    """
    props: PropsInput
    col_targets: np.ndarray
    row_targets: np.ndarray


def seg_inner(targets, results):
    for targ in targets:
        comp_mat = results.sum(set(results.dims).difference(targ.dims))
        adj = (targ / comp_mat).fillna(1)
        results *= adj
    # calc rmse
    diff = 0
    for targ in targets:
        check_mat = results.sum(set(results.dims).difference(targ.dims))
        diff += float(((check_mat - targ) ** 2).sum()) / np.prod(targ.shape)
    rmse = (diff) ** 0.5
    return results, rmse


def segmentation_furness(
    triple_inputs: dict[int, SegInput],
    sum_mat: np.ndarray,
    mat_size: tuple[int, int],
    max_iters: int = 5000,
    outer_max_iters: int = 500,
    tol: float = 1e-5,
):
    """
    Furness to a set of tlds, trip ends at some level of segmentation, and a full 
    matrix at an aggregate level.
    
    The furness initially attempts to match to distributions, trip ends and matrix. If 
    that process stops improving before convergence is met, it will fall back to only 
    furnessing to matrix and trip ends. In practice this process rarely gets to the 
    default level of convergence.
    Args:
        triple_inputs: dict[int, SegInput]
            Dict of segment value to an instance of SegInput
        sum_mat: np.ndarray
            The aggregate level matrix used as a target
        mat_size: tuple[int, int]
            The shape of the matrices.
        max_iters: int=5000: 
            The max number of iterations when resorting to the simpler furness
        outer_max_iters: int=500
            The max number of iterations at full detail
        tol: float = 1e-5
            The RMSE the process aims to get to.

    Returns:
        _type_: _description_
    """
    # Convert inputs to Xarrays for fast furnessing
    rows = xr.DataArray(
        data=[i.row_targets for i in triple_inputs.values()],
        dims=["seg", "o"],
        coords={"seg": list(triple_inputs.keys()), "o": np.arange(mat_size[0])},
    )
    cols = xr.DataArray(
        data=[i.col_targets for i in triple_inputs.values()],
        dims=["seg", "d"],
        coords={"seg": list(triple_inputs.keys()), "d": np.arange(mat_size[1])},
    )
    mat_targ = xr.DataArray(sum_mat, dims=["o", "d"])
    targets = [cols, rows, mat_targ]
    # Initial matrices are just the overall target matrix
    results = xr.DataArray(
        [np.ones(sum_mat.shape)] * len(triple_inputs),
        dims=["seg", "o", "d"],
        coords={
            "seg": list(triple_inputs.keys()),
            "o": np.arange(mat_size[0]),
            "d": np.arange(mat_size[1]),
        },
    )
    prev_rmse = np.inf
    iter_num = 1
    while True:
        for seg, val in triple_inputs.items():
            dist_matched = dist_match(val.props, results.sel(seg=seg).values)
            results.loc[seg] = dist_matched

        results, rmse = seg_inner(targets, results)

        LOG.debug(f"iter:{iter_num}, rmse:{rmse}")
        if rmse < tol:
            LOG.info(
                f"Full furnessing (to distributions, trip ends and matrix sum) has converged in "
                f"{iter_num} iterations, with an rmse of {rmse}."
            )
            break
        # This break condition needs deciding better. Some do seem to improve very slowly for 
        # a very long time.
        if prev_rmse - rmse < rmse / 100:
            LOG.warning(
                f"Full furnessing (to distributions, trip ends and matrix sum) has stopped "
                f"improving after {iter_num} iterations, with an rmse of {rmse}. Furnessing to "
                f"just trip ends and matrix sum to reach convergence."
            )
            for i in range(max_iters):
                results, rmse = seg_inner(targets, results)
                if rmse < tol:
                    LOG.info(
                        f"Partial furnessing to trip ends and matrix sum has converged in {i} "
                        f"iterations with an rmse of {rmse}."
                    )
                    break
                if prev_rmse - rmse < tol:
                    break
                prev_rmse = rmse
            if rmse > tol:
                LOG.warning(
                    f"Partial furnessing to trip ends and matrix sum has failed to converge in {i} "
                    f"iterations with an rmse of {rmse}."
                )
            break
        if iter_num > outer_max_iters:
            LOG.warning(
                f"Full furnessing (to distributions, trip ends and matrix sum) has failed to "
                f"converge after {iter_num} iterations, with an rmse of {rmse}. Furnessing to "
                f"just trip ends and matrix sum to reach convergence."
            )
            for i in range(max_iters):
                results, rmse = seg_inner(targets, results)
                if rmse < tol:
                    LOG.info(
                        f"Partial furnessing to trip ends and matrix sum has converged in {i} "
                        f"iterations with an rmse of {rmse}."
                    )
                    break
            if rmse > tol:
                LOG.warning(
                    f"Partial furnessing to trip ends and matrix sum has failed to converge in {i} "
                    f"iterations with an rmse of {rmse}."
                )
            break
        prev_rmse = rmse
        iter_num += 1
    checkers = {}
    for seg, val in triple_inputs.items():
        checks, alt_mat = dist_match(val.props, results.sel(seg=seg).values, True)
        checkers[seg] = checks
    return (
        results.to_dataframe(name="trips").squeeze().unstack(level="d"),
        rmse,
        pd.concat(checkers),
    )
