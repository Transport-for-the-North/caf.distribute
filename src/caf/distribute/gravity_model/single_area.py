# -*- coding: utf-8 -*-
"""Implementation of a self-calibrating single area gravity model."""
# Built-Ins
import logging
import os
from typing import Any, Optional

# Third Party
import numpy as np
from caf.toolkit import cost_utils, timing, toolbox
from scipy import optimize

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.distribute import cost_functions, furness
from caf.distribute.gravity_model import core

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
class SingleAreaGravityModelCalibrator(core.GravityModelBase):
    """A self-calibrating single area gravity model.

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

    cost_function:
        The cost function to use when calibrating the gravity model. This
        function is applied to `cost_matrix` before Furnessing during
        calibration.

    cost_matrix:
        A matrix detailing the cost between each and every zone. This
        matrix must be the same size as
        `(len(row_targets), len(col_targets))`.
    """

    def __init__(
        self,
        row_targets: np.ndarray,
        col_targets: np.ndarray,
        cost_function: cost_functions.CostFunction,
        cost_matrix: np.ndarray,
    ):
        super().__init__(
            cost_function=cost_function,
            cost_matrix=cost_matrix,
        )

        # Set attributes
        self.row_targets = row_targets
        self.col_targets = col_targets

    def _gravity_function(
        self,
        cost_args: list[float],
        running_log_path: os.PathLike,
        target_cost_distribution: Optional[cost_utils.CostDistribution] = None,
        diff_step: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        # inherited docstring
        # Not used, but need for compatibility with self._jacobian_function
        del diff_step

        # Init
        cost_kwargs = self._cost_params_to_kwargs(cost_args)
        cost_matrix = self._apply_perceived_factors(self.cost_matrix)

        # Furness trips to trip ends
        matrix, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=self.cost_function.calculate(cost_matrix, **cost_kwargs),
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            **kwargs,
        )

        # Evaluate the performance of this run
        cost_distribution, achieved_residuals, convergence = core.cost_distribution_stats(
            achieved_trip_distribution=matrix,
            cost_matrix=self.cost_matrix,
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
        diff_step: float,
        running_log_path: os.PathLike,
        target_cost_distribution: cost_utils.CostDistribution,
        **kwargs,
    ) -> np.ndarray:
        # inherited docstring
        # pylint: disable=too-many-locals
        # Not used, but need for compatibility with self._gravity_function
        del running_log_path
        del kwargs

        # Initialise the output
        jacobian = np.zeros((target_cost_distribution.n_bins, len(cost_args)))

        # Initialise running params
        cost_kwargs = self._cost_params_to_kwargs(cost_args)
        cost_matrix = self._apply_perceived_factors(self.cost_matrix)
        row_targets = self.achieved_distribution.sum(axis=1)
        col_targets = self.achieved_distribution.sum(axis=0)

        # Estimate what the furness does to the matrix
        base_matrix = self.cost_function.calculate(cost_matrix, **cost_kwargs)
        furness_factor = np.divide(
            self.achieved_distribution,
            base_matrix,
            where=base_matrix != 0,
            out=np.zeros_like(base_matrix),
        )

        # Build the Jacobian matrix.
        for i, cost_param in enumerate(self.cost_function.kw_order):
            cost_step = cost_kwargs[cost_param] * diff_step

            # Get slightly adjusted base matrix
            adj_cost_kwargs = cost_kwargs.copy()
            adj_cost_kwargs[cost_param] += cost_step
            adj_base_mat = self.cost_function.calculate(cost_matrix, **adj_cost_kwargs)

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

            # Calculate the Jacobian values for this cost param
            adj_cost_dist = cost_utils.CostDistribution.from_data(
                matrix=adj_final,
                cost_matrix=self.cost_matrix,
                bin_edges=target_cost_distribution.bin_edges,
            )

            jacobian_residuals = self.achieved_band_share - adj_cost_dist.band_share_vals
            jacobian[:, i] = jacobian_residuals / cost_step

        return jacobian



# # # FUNCTIONS # # #
def gravity_model(
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    cost_function: cost_functions.CostFunction,
    costs: np.ndarray,
    furness_max_iters: int,
    furness_tol: float,
    **cost_params,
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
    # Validate additional arguments passed in
    equal, extra, missing = toolbox.compare_sets(
        set(cost_params.keys()),
        set(cost_function.param_names),
    )

    if not equal:
        raise TypeError(
            f"gravity_model() got one or more unexpected keyword arguments.\n"
            f"Received the following extra arguments: {extra}\n"
            f"While missing arguments: {missing}"
        )

    # Calculate initial matrix through cost function
    init_matrix = cost_function.calculate(costs, **cost_params)

    # Furness trips to trip ends
    matrix, iters, rmse = furness.doubly_constrained_furness(
        seed_vals=init_matrix,
        row_targets=row_targets,
        col_targets=col_targets,
        tol=furness_tol,
        max_iters=furness_max_iters,
    )

    return matrix, iters, rmse
