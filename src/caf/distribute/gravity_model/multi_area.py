# -*- coding: utf-8 -*-
"""Implementation of a self-calibrating single area gravity model."""
# Built-Ins
import logging
from dataclasses import dataclass
from typing import Any, Optional
import os
from pathlib import Path
import functools

# Third Party
import numpy as np
import pandas as pd
from scipy import optimize
from caf.distribute.gravity_model.core import GravityModelCalibrateResults

# Local Imports
from caf.toolkit import cost_utils, timing, BaseConfig
from caf.distribute.gravity_model import core
from caf.distribute import cost_functions, furness


# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# pylint:disable=duplicate-code
# # # CLASSES # # #
class MultiDistInput(BaseConfig):
    """
    Input to multi cost distribution calibrator.

    Parameters
    ----------
    TLDFile: Path
        Path to a file containing distributions. This should contain 5 columns,
        the names of which must be specified below.
    TldLookupFile: Path
        Path to a lookup from distribtion areas to zones. Should contain 2
        columns which are explained below.
    cat_col: str
        The name of the column containing distribution area/categories in TLDFile.
        E.g. 'City', 'Village', 'Town', if there are different distributions for
        these different are types
    min_col: str
        The name of the column containing lower bounds of cost bands.
    max_col: str
        The name of the column containing upper bounds of cost bands.
    ave_col: str
        The name of the column containing average values of cost bands.
    trips_col: str
        The name of the column containing numbers of trips for a given cost band.
    lookup_cat_col: str
        The name of the column in the lookup containing the categories. The
        names of the values (but not the column name) must match the names in
        the cat_col of the TLD file. There must not be any distributions defined
        in the TLDFile which do not appear in the lookup.
    lookup_zone_col: str
        The column in the lookup containing zone identifiers. The lookup must
        contain all zones in the zone system.
    init_params: dict[str, float]
        A dict containing init_params for the cost function when calibrating.
        If left blank the default value from the cost_function will be used.
    log_path: Path
        Path to where the log file should be saved. Saved as a csv but this can
        also be a path to a txt file.
    furness_tolerance: float
        The tolerance for the furness in the gravity function. In general lower
        tolerance will take longer but may yield better results.
    furness_jac: bool
        Whether to furness within the jacobian function. Not furnessing within
        the jacobian does not represent knock on effects to other areas of
        altering parameters for a given area. If you expect these effects to be
        significant this should be set to True, but otherwise the process runs
        quicker with it set to False.
    """

    TLDFile: Path
    TldLookupFile: Path
    cat_col: str
    min_col: str
    max_col: str
    ave_col: str
    trips_col: str
    lookup_cat_col: str
    lookup_zone_col: str
    init_params: dict[str, float]
    log_path: Path
    furness_tolerance: float = 1e-6
    furness_jac: float = False


@dataclass
class MultiCostDistribution:
    """
    Dataclass for storing needed info for a MultiCostDistribution model.

    Parameters
    ----------
    name: str
        The name of the distribution (this will usually identify the area
        applicable e.g. City, Rural)
    cost_distribution: cost_utils.CostDistribution
        A cost distribution in a CostDistribution class from toolkit. This
        will often be a trip-length distribution but cost can be in any units
        as long as they match the cost matrix
    zones: np.ndarray
        The zones this distribution applies to. This is NOT zone number, or zone
        ID but the indices of the relevant zones in your cost matrix/target_rows
    function_params: dict[str,str]
        Initial parameters for your cost function to start guessing at. There
        is a method included for choosing these which hasn't yet been
        implemented.
    """

    name: str
    cost_distribution: cost_utils.CostDistribution
    zones: np.ndarray
    function_params: dict[str, float]


class MultiAreaGravityModelCalibrator(core.GravityModelBase):
    """
    A self-calibrating multi-area gravity model.

    Parameters
    ----------
    row_targets:
        The targets for each row that the gravity model should be aiming to
        match. This can alternatively be thought of as the rows that wish to
        be distributed.

    col_targets:
        The targets for each column that the gravity model should be
        aiming to match. This can alternatively be thought of as the
        columns that wish to be distributed.

    cost_matrix:
        A matrix detailing the cost between each and every zone. This
        matrix must be the same size as
        `(len(row_targets), len(col_targets))`.

    cost_function:
        The cost function to use when calibrating the gravity model. This
        function is applied to `cost_matrix` before Furnessing during
        calibration.

    target_cost_distributions:
        A list of cost distributions for the model. See documentation for the
        class MultiCostDistribution. All zones in the cost_matrix/targets must
        be accounted for, and should only appear in one distribution each.
    """

    def __init__(
        self,
        row_targets: np.ndarray,
        col_targets: np.ndarray,
        cost_matrix: np.ndarray,
        cost_function: cost_functions.CostFunction,
        params: Optional[MultiDistInput],
    ):
        super().__init__(cost_function=cost_function, cost_matrix=cost_matrix)
        self.row_targets = row_targets
        self.col_targets = col_targets
        if len(row_targets) != cost_matrix.shape[0]:
            raise IndexError("row_targets doesn't match cost_matrix")
        if len(col_targets) != cost_matrix.shape[1]:
            raise IndexError("col_targets doesn't match cost_matrix")
        if params is not None:
            self.tlds = pd.read_csv(params.TLDFile)
            self.tlds.rename(
                columns={
                    params.cat_col: "cat",
                    params.min_col: "min",
                    params.max_col: "max",
                    params.ave_col: "ave",
                    params.trips_col: "trips",
                },
                inplace=True,
            )
            self.lookup = pd.read_csv(params.TldLookupFile)
            self.lookup.rename(
                columns={params.lookup_zone_col: "zone", params.lookup_cat_col: "cat"},
                inplace=True,
            )
            self.tlds.set_index("cat", inplace=True)
            self.lookup.sort_values("zone")
            self.init_params = params.init_params
            self.dists = self.process_tlds()
            self.log_path = params.log_path
            self.furness_tol = params.furness_tolerance
            self.furness_jac = params.furness_jac

    def process_tlds(self):
        """Get distributions in the right format for a multi-area gravity model."""
        dists = []
        for cat in self.tlds.index.unique():
            tld = self.tlds.loc[cat]
            tld = cost_utils.CostDistribution(tld)
            zones = self.lookup[self.lookup["cat"] == cat].index.values
            if len(zones) == 0:
                raise ValueError(
                    f"{cat} doesn't seem to appear in the given tld "
                    "lookup. Check for any typos (e.g. lower/upper case). "
                    f"If this is expected, remove {cat} from your "
                    "tlds dataframe before inputting."
                )

            distribution = MultiCostDistribution(
                name=cat, cost_distribution=tld, zones=zones, function_params=self.init_params
            )
            dists.append(distribution)

        return dists

    def _calculate_perceived_factors(
        self,
        target_cost_distribution: cost_utils.CostDistribution,
        achieved_band_shares: np.ndarray,
    ) -> None:
        raise NotImplementedError("WIP")

    @property
    def achieved_band_share(self) -> np.ndarray:
        """Overload achieved_band _share for multiple bands."""
        if self.achieved_cost_dist is None:
            raise ValueError("Gravity model has not been run. achieved_band_share is not set.")
        shares = []
        for dist in self.achieved_cost_dist:
            shares.append(dist.band_share_vals)
        return np.concatenate(shares)

    def _create_seed_matrix(self, cost_distributions, cost_args, params_len):
        base_mat = np.zeros_like(self.cost_matrix)
        for i, dist in enumerate(cost_distributions):
            init_params = cost_args[i * params_len : i * params_len + params_len]
            init_params_kwargs = self._cost_params_to_kwargs(init_params)
            mat_slice = self.cost_function.calculate(
                self.cost_matrix[dist.zones], **init_params_kwargs
            )
            base_mat[dist.zones] = mat_slice
        return base_mat

    # pylint: disable=too-many-locals
    def _calibrate(
        self,
        diff_step: float = 1e-8,
        ftol: float = 1e-4,
        xtol: float = 1e-4,
        grav_max_iters: int = 100,
        failure_tol: float = 0,
        default_retry: bool = True,
        verbose: int = 0,
        **kwargs,
    ) -> dict[str, GravityModelCalibrateResults]:
        params_len = len(self.dists[0].function_params)
        ordered_init_params = []
        for dist in self.dists:
            params = self._order_cost_params(dist.function_params)
            for val in params:
                ordered_init_params.append(val)

        gravity_kwargs: dict[str, Any] = {
            "running_log_path": self.log_path,
            "cost_distributions": self.dists,
            "diff_step": diff_step,
            "params_len": params_len,
        }
        optimise_cost_params = functools.partial(
            optimize.least_squares,
            fun=self._gravity_function,
            method=self._least_squares_method,
            bounds=(
                self._order_bounds()[0] * len(self.dists),
                self._order_bounds()[1] * len(self.dists),
            ),
            jac=self._jacobian_function,
            verbose=verbose,
            ftol=ftol,
            xtol=xtol,
            max_nfev=grav_max_iters,
            kwargs=gravity_kwargs | kwargs,
        )
        result = optimise_cost_params(x0=ordered_init_params)

        LOG.info(
            "%scalibration process ended with "
            "success=%s, and the following message:\n"
            "%s",
            self.unique_id,
            result.success,
            result.message,
        )

        best_convergence = self.achieved_convergence
        best_params = result.x

        if (not all(self.achieved_convergence) >= failure_tol) and default_retry:
            LOG.info(
                "%sachieved a convergence of %s, "
                "however the failure tolerance is set to %s. Trying again with "
                "default cost parameters.",
                self.unique_id,
                self.achieved_convergence,
                failure_tol,
            )
            self._attempt_id += 1
            ordered_init_params = self._order_cost_params(self.cost_function.default_params)
            result = optimise_cost_params(x0=ordered_init_params)

            # Update the best params only if this was better
            if np.mean(list(self.achieved_convergence.values())) > np.mean(
                list(best_convergence.values())
            ):
                best_params = result.x

        self._attempt_id = -2
        self._gravity_function(
            init_params=best_params,
            **(gravity_kwargs | kwargs),
        )

        assert self.achieved_cost_dist is not None
        results = {}
        for i, dist in enumerate(self.dists):
            result_i = GravityModelCalibrateResults(
                cost_distribution=self.achieved_cost_dist[i],
                cost_convergence=self.achieved_convergence[dist.name],
                value_distribution=self.achieved_distribution[dist.zones],
                target_cost_distribution=dist.cost_distribution,
                cost_function=self.cost_function,
                cost_params=self._cost_params_to_kwargs(
                    best_params[i * params_len : i * params_len + params_len]
                ),
            )

            results[dist.name] = result_i
        return results

    def calibrate(
        self,
        running_log_path: os.PathLike,
        *args,
        **kwargs,
    ) -> GravityModelCalibrateResults:
        """Find the optimal parameters for self.cost_function.

        Optimal parameters are found using `scipy.optimize.least_squares`
        to fit the distributed row/col targets to `target_cost_distribution`.

        Parameters
        ----------
        init_params:
            A dictionary of {parameter_name: parameter_value} to pass
            into the cost function as initial parameters.

        running_log_path:
            Path to output the running log to. This log will detail the
            performance of the run and is written in .csv format.

        target_cost_distribution:
            The cost distribution to calibrate towards during the calibration
            process.

        diff_step:
            Copied from scipy.optimize.least_squares documentation, where it
            is passed to:
            Determines the relative step size for the finite difference
            approximation of the Jacobian. The actual step is computed as
            `x * diff_step`. If None (default), then diff_step is taken to be a
            conventional “optimal” power of machine epsilon for the finite
            difference scheme used

        ftol:
            The tolerance to pass to `scipy.optimize.least_squares`. The search
            will stop once this tolerance has been met. This is the
            tolerance for termination by the change of the cost function

        xtol:
            The tolerance to pass to `scipy.optimize.least_squares`. The search
            will stop once this tolerance has been met. This is the
            tolerance for termination by the change of the independent
            variables.

        grav_max_iters:
            The maximum number of calibration iterations to complete before
            termination if the ftol has not been met.

        failure_tol:
            If, after initial calibration using `init_params`, the achieved
            convergence is less than this value, calibration will be run again with
            the default parameters from `self.cost_function`.

        default_retry:
            If, after running with `init_params`, the achieved convergence
            is less than `failure_tol`, calibration will be run again with the
            default parameters of `self.cost_function`.
            This argument is ignored if the default parameters are given
            as `init_params.

        n_random_tries:
            If, after running with default parameters of `self.cost_function`,
            the achieved convergence is less than `failure_tol`, calibration will
            be run again using random values for the cost parameters this
            number of times.

        verbose:
            Copied from scipy.optimize.least_squares documentation, where it
            is passed to:
            Level of algorithm’s verbosity:
            - 0 (default) : work silently.
            - 1 : display a termination report.
            - 2 : display progress during iterations (not supported by ‘lm’ method).

        kwargs:
            Additional arguments passed to self.gravity_furness.
            Empty by default. The calling signature is:
            `self.gravity_furness(seed_matrix, **kwargs)`

        Returns
        -------
        results:
            An instance of GravityModelCalibrateResults containing the
            results of this run.

        See Also
        --------
        `caf.distribute.furness.doubly_constrained_furness()`
        `scipy.optimize.least_squares()`
        """
        for dist in self.dists:
            self.cost_function.validate_params(dist.function_params)
        self._validate_running_log(running_log_path)
        self._initialise_internal_params()
        return self._calibrate(  # type: ignore
            *args,
            running_log_path=running_log_path,
            **kwargs,
        )

    def _jacobian_function(
        self,
        init_params: list[float],
        cost_distributions: list[MultiCostDistribution],
        diff_step: float,
        running_log_path,
        params_len,
    ):
        del running_log_path
        # Build empty jacobian matrix
        jac_length = sum(len(dist.cost_distribution) for dist in cost_distributions)
        jac_width = len(cost_distributions) * params_len
        jacobian = np.zeros((jac_length, jac_width))
        # Build seed matrix
        base_mat = self._create_seed_matrix(cost_distributions, init_params, params_len)
        # Calculate net effect of furnessing (saves a lot of time on furnessing here)
        furness_factor = np.divide(
            self.achieved_distribution,
            base_mat,
            where=base_mat != 0,
            out=np.zeros_like(base_mat),
        )
        # Allows iteration of cost_distributions within a loop of cost_distributions
        inner_dists = cost_distributions.copy()

        for j, dist in enumerate(cost_distributions):
            distributions = init_params[j * params_len : j * params_len + params_len]
            init_params_kwargs = self._cost_params_to_kwargs(distributions)
            for i, cost_param in enumerate(self.cost_function.kw_order):
                cost_step = init_params_kwargs[cost_param] * diff_step
                adj_cost_kwargs = init_params_kwargs.copy()
                adj_cost_kwargs[cost_param] += cost_step
                adj_mat_slice = self.cost_function.calculate(
                    self.cost_matrix[dist.zones], **adj_cost_kwargs
                )
                adj_mat = base_mat.copy()
                adj_mat[dist.zones] = adj_mat_slice
                adj_dist = adj_mat * furness_factor
                if self.furness_jac:
                    adj_dist, *_ = furness.doubly_constrained_furness(
                        seed_vals=adj_dist,
                        row_targets=self.achieved_distribution.sum(axis=1),
                        col_targets=self.achieved_distribution.sum(axis=0),
                        tol=self.furness_tol / 10,
                        max_iters=20,
                        warning=False,
                    )
                test_res = []
                for inner_dist in inner_dists:
                    adj_cost_dist = cost_utils.CostDistribution.from_data(
                        matrix=adj_dist[inner_dist.zones],
                        cost_matrix=self.cost_matrix[inner_dist.zones],
                        bin_edges=inner_dist.cost_distribution.bin_edges,
                    )
                    act_cost_dist = cost_utils.CostDistribution.from_data(
                        matrix=self.achieved_distribution[inner_dist.zones],
                        cost_matrix=self.cost_matrix[inner_dist.zones],
                        bin_edges=inner_dist.cost_distribution.bin_edges,
                    )
                    test_res.append(
                        act_cost_dist.band_share_vals - adj_cost_dist.band_share_vals
                    )
                test_outer = np.concatenate(test_res)
                jacobian[:, 2 * j + i] = test_outer / cost_step
        return jacobian

    def _gravity_function(
        self, init_params, cost_distributions, running_log_path, params_len, diff_step=0
    ):
        del diff_step

        base_mat = self._create_seed_matrix(cost_distributions, init_params, params_len)
        matrix, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=base_mat,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            tol=self.furness_tol,
        )
        convergences, distributions, residuals = {}, [], []
        for dist in cost_distributions:
            (
                single_cost_distribution,
                single_achieved_residuals,
                single_convergence,
            ) = core.cost_distribution_stats(
                achieved_trip_distribution=matrix[dist.zones],
                cost_matrix=self.cost_matrix[dist.zones],
                target_cost_distribution=dist.cost_distribution,
            )
            convergences[dist.name] = single_convergence
            distributions.append(single_cost_distribution)
            residuals.append(single_achieved_residuals)

        log_costs = {}

        for i, dist in enumerate(cost_distributions):
            j = 0
            for name in dist.function_params.keys():
                log_costs[f"{name}_{i}"] = init_params[params_len * i + j]
                j += 1
            log_costs[f"convergence_{i}"] = convergences[dist.name]

        end_time = timing.current_milli_time()
        self._log_iteration(
            log_path=running_log_path,
            attempt_id=self._attempt_id,
            loop_num=self._loop_num,
            loop_time=(end_time - self._loop_start_time) / 1000,
            cost_kwargs=log_costs,
            furness_iters=iters,
            furness_rmse=rmse,
            convergence=np.mean(list(convergences.values())),
        )

        self._loop_num += 1
        self._loop_start_time = timing.current_milli_time()

        self.achieved_cost_dist: list[cost_utils.CostDistribution] = distributions
        self.achieved_convergence: dict[str, float] = convergences
        self.achieved_distribution = matrix

        achieved_residuals = np.concatenate(residuals)

        return achieved_residuals

    # pylint:enable=too-many-locals
    def run(self):
        """
        Run the gravity_model without calibrating.

        This should be done when you have calibrating previously to find the
        correct parameters for the cost function.
        """
        params_len = len(self.dists[0].function_params)
        cost_args = []
        for dist in self.dists:
            for param in dist.function_params.values():
                cost_args.append(param)

        self._gravity_function(
            init_params=cost_args,
            cost_distributions=self.dists,
            running_log_path=self.log_path,
            params_len=params_len,
        )

        assert self.achieved_cost_dist is not None
        results = {}
        for i, dist in enumerate(self.dists):
            gresult = GravityModelCalibrateResults(
                cost_distribution=self.achieved_cost_dist[i],
                cost_convergence=self.achieved_convergence[dist.name],
                value_distribution=self.achieved_distribution[dist.zones],
                target_cost_distribution=dist.cost_distribution,
                cost_function=self.cost_function,
                cost_params=self._cost_params_to_kwargs(
                    cost_args[i * params_len : i * params_len + params_len]
                ),
            )

            results[dist.name] = gresult
        return results


def gravity_model(
    row_targets: pd.Series,
    col_targets: np.ndarray,
    cost_distributions: list[MultiCostDistribution],
    cost_function: cost_functions.CostFunction,
    cost_mat: pd.DataFrame,
    furness_max_iters: int,
    furness_tol: float,
):
    """
    Run a gravity model and returns the distributed row/col targets.

    Uses the given cost function to generate an initial matrix which is
    used in a double constrained furness to distribute the row and column
    targets across a matrix. The cost_params can be used to achieve different
    results based on the cost function.

    Parameters
    ----------
    row_targets:
        The targets for the rows to sum to. These are usually Productions
        in Trip Ends.

    col_targets:
        The targets for the columns to sum to. These are usually Attractions
        in Trip Ends.

    cost_function:
        A cost function class defining how to calculate the seed matrix based
        on the given cost. cost_params will be passed directly into this
        function.

    costs:
        A matrix of the base costs to use. This will be passed into
        cost_function alongside cost_params. Usually this will need to be
        the same shape as (len(row_targets), len(col_targets)).

    furness_max_iters:
        The maximum number of iterations for the furness to complete before
        giving up and outputting what it has managed to achieve.

    furness_tol:
        The R2 difference to try and achieve between the row/col targets
        and the generated matrix. The smaller the tolerance the closer to the
        targets the return matrix will be.

    cost_params:
        Any additional parameters that should be passed through to the cost
        function.

    Returns
    -------
    distributed_matrix:
        A matrix of the row/col targets distributed into a matrix of shape
        (len(row_targets), len(col_targets))

    completed_iters:
        The number of iterations completed by the doubly constrained furness
        before exiting

    achieved_rmse:
        The Root Mean Squared Error achieved by the doubly constrained furness
        before exiting

    Raises
    ------
    TypeError:
        If some of the cost_params are not valid cost parameters, or not all
        cost parameters have been given.
    """
    seed_slices = []
    for distribution in cost_distributions:
        cost_slice = cost_mat.loc[distribution.zones]
        seed_slice = cost_function.calculate(cost_slice, **distribution.function_params)
        seed_slices.append(seed_slice)
    seed_matrix = pd.concat(seed_slices)

    # Furness trips to trip ends
    matrix, iters, rmse = furness.doubly_constrained_furness(
        seed_vals=seed_matrix.values,
        row_targets=row_targets,
        col_targets=col_targets,
        tol=furness_tol,
        max_iters=furness_max_iters,
    )

    return matrix, iters, rmse


# pylint:enable=duplicate-code

# # # FUNCTIONS # # #
