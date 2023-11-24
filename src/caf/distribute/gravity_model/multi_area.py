# -*- coding: utf-8 -*-
"""Implementation of a self-calibrating single area gravity model."""
# Built-Ins
import logging
from dataclasses import dataclass
from typing import Iterator
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
    name: str
    cost_function: cost_functions.CostFunction
    cost_distribution: cost_utils.CostDistribution
    zones: np.ndarray
    function_params: dict[str, float]



class MultiAreaGravityModelCalibrator(core.GravityModelBase):
    # TODO(MB) Implement functionality for the abstract methods
    # actual cost matrix
    # adjusted cost matrix (optional) instead could be callable

    # Need to update functions in base class to allow for giving multiple cost distributions
    # Ideally calibrate and calibrate_with_percieved factors should be defined flexibly
    # in base class

    def __init__(
            self,
            row_targets: np.ndarray,
            col_targets: np.ndarray,
            cost_matrix: np.ndarray,
            target_cost_distributions: list[MultiCostDistribution]
    ):
        self.row_targets = row_targets = pd.Series(row_targets, index=range(1, len(row_targets) + 1))
        self.col_targets = col_targets
        self.cost_matrix = pd.DataFrame(cost_matrix, index=range(1, len(row_targets) + 1), columns=range(1, len(col_targets) + 1))
        self.target_cost_distributions = target_cost_distributions
    def _calculate_perceived_factors(self) -> None:
        raise NotImplementedError("WIP")

    def _calibrate(self,
                   running_log_path: os.PathLike,
                   diff_step: float = 1e-8,
                   ftol: float = 1e-4,
                   xtol: float = 1e-4,
                   grav_max_iters: int = 100,
                   failure_tol: float = 0,
                   n_random_tries: int = 3,
                   default_retry: bool = True,
                   verbose: int = 0,
                   **kwargs,
                   ) -> GravityModelCalibrateResults:
        # TODO(MB) Some small amount of refactoring in core but
        # hopefully can use alot of the same code
        seed_mat_slices = []
        for distribution in self.target_cost_distributions:
            seed_mat_slice = distribution.cost_function.calculate(self.cost_matrix.loc[distribution.zones].values)
            seed_mat_slices.append(pd.DataFrame(seed_mat_slice, index=distribution.zones))
        seed_mat = pd.concat(seed_mat_slices).sort_index()
        best_params = {}
        for distribution in self.target_cost_distributions:
            init_params = distribution.cost_function.default_params
            gravity_kwargs = {
                "running_log_path": running_log_path,
                "target_cost_distribution": distribution.cost_distribution,
                "diff_step": diff_step,
                "seed_mat": seed_mat
            }

            optimise_cost_params = functools.partial(
                optimize.least_squares,
                fun=self._gravity_function,
                method=self._least_squares_method,
                bounds=self._order_bounds(),
                jac=self._jacobian_function,
                verbose=verbose,
                ftol=ftol,
                xtol=xtol,
                max_nfev=grav_max_iters,
                kwargs=gravity_kwargs | kwargs,
            )
            ordered_init_params = self._order_cost_params(init_params)
            result = optimise_cost_params(x0=ordered_init_params)
            LOG.info(
                "%scalibration process ended with "
                "success=%s, and the following message:\n"
                "%s",
                distribution.name,
                result.success,
                result.message,
            )
            # Track the best performance through the runs
            best_convergence = self.achieved_convergence
            best_params = result.x

            # Bad init params might have been given, try default
            if self.achieved_convergence <= failure_tol and default_retry:
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
                if self.achieved_convergence > best_convergence:
                    best_params = result.x

            # Last chance, try again with random values
            if self.achieved_convergence <= failure_tol and n_random_tries > 0:
                LOG.info(
                    "%sachieved a convergence of %s, "
                    "however the failure tolerance is set to %s. Trying again with "
                    "random cost parameters.",
                    self.unique_id,
                    self.achieved_convergence,
                    failure_tol,
                )
                self._attempt_id += 100
                for i in range(n_random_tries):
                    self._attempt_id += i
                    random_params = self.cost_function.random_valid_params()
                    ordered_init_params = self._order_cost_params(random_params)
                    result = optimise_cost_params(x0=ordered_init_params)

                    # Update the best params only if this was better
                    if self.achieved_convergence > best_convergence:
                        best_params = result.x

                    if self.achieved_convergence > failure_tol:
                        break

            # Run one final time with the optimal parameters
            self.optimal_cost_params = self._cost_params_to_kwargs(best_params)
            self._attempt_id = -2
            self._gravity_function(
                cost_args=best_params,
                **(gravity_kwargs | kwargs),
            )
            assert self.achieved_cost_dist is not None
            distribution.function_params = self.optimal_cost_params
            best_params[distribution.name] = distribution
        return best_params

    def _gravity_function(self,
                          cost_args: list[float],
                          running_log_path: os.PathLike,
                          zones: list[int],
                          seed_mat: pd.DataFrame,
                          diff_step: float = 0.0,
                          target_cost_distribution: cost_utils.CostDistribution,
                          **kwargs
                          ) -> np.ndarray:
        del diff_step

        cost_kwargs = self._cost_params_to_kwargs(cost_args)
        cost_matrix = self._apply_perceived_factors(self.cost_matrix[zones])

        matrix, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=seed_mat,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            **kwargs,
        )
        matrix = pd.DataFrame(matrix, index=seed_mat.index)
        cost_mat = pd.DataFrame(self.cost_matrix, index=seed_mat.index)
        # Evaluate the performance only for the target rows
        cost_distribution, achieved_residuals, convergence = core.cost_distribution_stats(
            achieved_trip_distribution=matrix.loc[zones],
            cost_matrix=cost_mat.loc[zones],
            target_cost_distribution=target_cost_distribution,
        )

        # Log this iteration
        end_time = timing.current_milli_time()
        self._log_iteration(
            log_path=running_log_path,
            attempt_id=self._attempt_id,
            loop_num=self._loop_num,
            loop_time=(end_time - self._loop_start_time) / 1000,
            cost_kwargs=cost_kwargs,
            furness_iters=iters,
            furness_rmse=rmse,
            convergence=convergence,
        )

        # Update loop params and return the achieved band shares
        self._loop_num += 1
        self._loop_start_time = timing.current_milli_time()

        # Update performance params
        self.achieved_cost_dist = cost_distribution
        self.achieved_convergence = convergence
        self.achieved_distribution = matrix

        # Store the initial values to log later
        if self.initial_cost_params is None:
            self.initial_cost_params = cost_kwargs
        if self.initial_convergence is None:
            self.initial_convergence = convergence

        return achieved_residuals

    def _jacobian_function(
            self,
            cost_args: list[float],
            running_log_path: os.PathLike,
            target_cost_distribution: cost_utils.CostDistribution,
            zones: list[int],
            seed_mat: pd.DataFrame,
            diff_step: float = 0.0,
            **kwargs
    ):
        del running_log_path
        del kwargs

        # Initialise the output
        jacobian = np.zeros((target_cost_distribution.n_bins, len(cost_args)))

        # Initialise running params
        cost_kwargs = self._cost_params_to_kwargs(cost_args)
        cost_matrix = self._apply_perceived_factors(self.cost_matrix)
        row_targets = self.achieved_distribution.sum(axis=1)
        col_targets = self.achieved_distribution.sum(axis=0)

        cost_df = pd.DataFrame(cost_matrix, index=seed_mat.index)

        furness_factor = np.divide(
            self.achieved_distribution.values,
            seed_mat.values,
            where=seed_mat.values != 0,
            out=np.zeros_like(seed_mat.values),
        )
        # Build the Jacobian matrix.
        for i, cost_param in enumerate(self.cost_function.kw_order):
            cost_step = cost_kwargs[cost_param] * diff_step

            # Get slightly adjusted base matrix
            adj_cost_kwargs = cost_kwargs.copy()
            adj_cost_kwargs[cost_param] += cost_step
            adj_base_mat_slice = self.cost_function.calculate(cost_matrix.loc[zones], **adj_cost_kwargs)
            adj_base_mat_slice = pd.DataFrame(adj_base_mat_slice, index=zones)
            adj_base_mat = self.achieved_distribution.copy()
            adj_base_mat.loc[zones] = adj_base_mat_slice

            # Estimate the impact of the furness
            adj_distribution = adj_base_mat * furness_factor
            if adj_distribution.sum() == 0:
                raise ValueError("estimated furness matrix total is 0")

            # Convert to weights to estimate impact on output
            adj_weights = adj_distribution / adj_distribution.sum()
            adj_final = self.achieved_distribution.sum() * adj_weights

            # Finesse to match row / col targets
            adj_final, *_ = furness.doubly_constrained_furness(
                seed_vals=adj_final,
                row_targets=row_targets,
                col_targets=col_targets,
                tol=1e-6,
                max_iters=20,
                warning=False,
            )

            adj_final = pd.DataFrame(adj_final, index=seed_mat.index)

            # Calculate the Jacobian values for this cost param
            adj_cost_dist = cost_utils.CostDistribution.from_data(
                matrix=adj_final.loc[zones],
                cost_matrix=cost_df.loc[zones],
                bin_edges=target_cost_distribution.bin_edges,
            )

            jacobian_residuals = self.achieved_band_share - adj_cost_dist.band_share_vals
            jacobian[:, i] = jacobian_residuals / cost_step

        return jacobian

def gravity_model(
    row_targets: pd.Series,
    col_targets: np.ndarray,
    cost_distributions: list[MultiCostDistribution],
    cost_mat: pd.DataFrame,
    furness_max_iters: int,
    furness_tol: float
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
        seed_slice = distribution.cost_function.calculate(cost_slice, **distribution.function_params)
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
