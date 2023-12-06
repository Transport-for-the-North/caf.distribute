# -*- coding: utf-8 -*-
"""Implementation of a self-calibrating single area gravity model."""
# Built-Ins
import logging
from dataclasses import dataclass
from typing import Any
import os

# Third Party
import numpy as np
import pandas as pd
from scipy import optimize
from caf.distribute.gravity_model.core import GravityModelCalibrateResults
import functools

# Local Imports
from caf.toolkit import cost_utils, timing
from caf.distribute.gravity_model import core
from caf.distribute import cost_functions, furness


# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
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

    # actual cost matrix
    # adjusted cost matrix (optional) instead could be callable

    # Need to update functions in base class to allow for giving multiple cost distributions
    # Ideally calibrate and calibrate_with_perceived factors should be defined flexibly
    # in base class

    def __init__(
        self,
        row_targets: np.ndarray,
        col_targets: np.ndarray,
        cost_matrix: np.ndarray,
        cost_function: cost_functions.CostFunction,
    ):
        super().__init__(cost_function=cost_function, cost_matrix=cost_matrix)
        self.row_targets = row_targets
        self.col_targets = col_targets
        if len(row_targets) != cost_matrix.shape[0]:
            raise IndexError("row_targets doesn't match cost_matrix")
        if len(col_targets) != cost_matrix.shape[1]:
            raise IndexError("col_targets doesn't match cost_matrix")

    def _calculate_perceived_factors(self) -> None:
        # TODO this
        raise NotImplementedError("WIP")

    @property
    def multi_achieved_band_shares(self) -> np.ndarray:
        """Analogous to _achieved_band_shares but for a multi-tld"""
        if self.achieved_cost_dist is None:
            raise ValueError("Gravity model has not been run. achieved_band_share is not set.")
        shares = []
        for dist in self.achieved_cost_dist:
            shares.append(dist.band_share_vals)
        return np.concatenate(shares)

    def _calibrate(
        self,
        running_log_path: os.PathLike,
        init_params: list[MultiCostDistribution],
        diff_step: float = 1e-8,
        ftol: float = 1e-6,
        xtol: float = 1e-4,
        grav_max_iters: int = 100,
        failure_tol: float = 0,
        default_retry: bool = True,
        verbose: int = 0,
        **kwargs,
    ) -> GravityModelCalibrateResults:
        params_len = len(init_params[0].function_params)
        ordered_init_params = []
        for dist in init_params:
            params = self._order_cost_params(dist.function_params)
            for val in params:
                ordered_init_params.append(val)

        gravity_kwargs: dict[str, Any] = {
            "running_log_path": running_log_path,
            "cost_distributions": init_params,
            "diff_step": diff_step,
            "params_len": params_len,
        }
        optimise_cost_params = functools.partial(
            optimize.least_squares,
            fun=self._gravity_function,
            method=self._least_squares_method,
            bounds=(
                self._order_bounds()[0] * len(init_params),
                self._order_bounds()[1] * len(init_params),
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
            if np.mean(self.achieved_convergence) > np.mean(best_convergence):
                best_params = result.x

        self._attempt_id = -2
        self._gravity_function(
            cost_args=best_params,
            **(gravity_kwargs | kwargs),
        )

        assert self.achieved_cost_dist is not None
        results = {}
        for i, dist in enumerate(init_params):
            gresult = GravityModelCalibrateResults(
                cost_distribution=self.achieved_cost_dist[i],
                cost_convergence=self.achieved_convergence[dist.name],
                value_distribution=self.achieved_distribution[dist.zones],
                target_cost_distribution=dist.cost_distribution,
                cost_function=self.cost_function,
                cost_params=self._cost_params_to_kwargs(
                    best_params[i * params_len : i * params_len + params_len]
                ),
            )

            results[dist.name] = gresult
        return results

    def calibrate(
        self,
        init_params: list[MultiCostDistribution],
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
        for dist in init_params:
            self.cost_function.validate_params(dist.function_params)
        self._validate_running_log(running_log_path)
        self._initialise_internal_params()
        return self._calibrate(  # type: ignore
            *args,
            init_params=init_params,
            running_log_path=running_log_path,
            **kwargs,
        )

    def _jacobian_function(
        self,
        cost_args: list[float],
        cost_distributions: list[MultiCostDistribution],
        diff_step: float,
        running_log_path,
        params_len,
    ):
        del running_log_path
        # Build empty jacobian matrix
        jac_length = sum([len(dist.cost_distribution) for dist in cost_distributions])
        jac_width = len(cost_distributions) * params_len
        jacobian = np.zeros((jac_length, jac_width))
        # Build seed matrix
        base_mat = np.zeros(self.achieved_distribution.shape)
        for i, dist in enumerate(cost_distributions):
            init_params_kwargs = cost_args[i * params_len : i * params_len + params_len]
            init_params_kwargs = self._cost_params_to_kwargs(init_params_kwargs)
            mat_slice = self.cost_function.calculate(
                self.cost_matrix[dist.zones], **init_params_kwargs
            )
            base_mat[dist.zones] = mat_slice
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
            distributions = cost_args[j * params_len : j * params_len + params_len]
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
                adj_final, *_ = furness.doubly_constrained_furness(
                    seed_vals=adj_dist,
                    row_targets=self.achieved_distribution.sum(axis=1),
                    col_targets=self.achieved_distribution.sum(axis=0),
                    tol=1e-7,
                    max_iters=20,
                    warning=False,
                )
                test_res = []
                for inner_dist in inner_dists:
                    adj_cost_dist = cost_utils.CostDistribution.from_data(
                        matrix=adj_final[inner_dist.zones],
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
        self, cost_args, cost_distributions, diff_step, running_log_path, params_len
    ):
        del diff_step

        base_mat = np.zeros(self.achieved_distribution.shape)
        for i, dist in enumerate(cost_distributions):
            init_params = cost_args[i * params_len : i * params_len + params_len]
            init_params_kwargs = self._cost_params_to_kwargs(init_params)
            mat_slice = self.cost_function.calculate(
                self.cost_matrix[dist.zones], **init_params_kwargs
            )
            base_mat[dist.zones] = mat_slice
        matrix, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=base_mat,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            tol=1e-1,
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
            for name, val in dist.function_params.items():
                log_costs[f"{name}_{i}"] = cost_args[params_len * i + j]
                j += 1

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

        self.achieved_cost_dist = distributions
        self.achieved_convergence = convergences
        self.achieved_distribution = matrix

        achieved_residuals = np.concatenate(residuals)

        return achieved_residuals


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


# # # FUNCTIONS # # #
