# -*- coding: utf-8 -*-
"""Implementation of a self-calibrating single area gravity model."""
# Built-Ins
import logging
import warnings

from typing import Any
from typing import Optional

# Third Party
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import toolbox
from caf.distribute import furness
from caf.distribute import cost_functions
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

    def gravity_furness(
        self,
        seed_matrix: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, int, float]:
        """Run a doubly constrained furness on the seed matrix.

        Wrapper around furness.doubly_constrained_furness, using class
        attributes to set up the function call.

        Parameters
        ----------
        seed_matrix:
            Initial values for the furness.

        kwargs:
            Additional arguments from the caller to pass to
            `self.doubly_constrained_furness`.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        return furness.doubly_constrained_furness(
            seed_vals=seed_matrix,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            **kwargs,
        )

    def jacobian_furness(
        self,
        seed_matrix: np.ndarray,
        row_targets: np.ndarray,
        col_targets: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run a doubly constrained furness on the seed matrix.

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the jacobian calculation.

        Parameters
        ----------
        seed_matrix:
            Initial values for the furness.

        row_targets:
            The target values for the sum of each row.
            i.e. np.sum(seed_matrix, axis=1)

        col_targets:
            The target values for the sum of each column
            i.e. np.sum(seed_matrix, axis=0)

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        return furness.doubly_constrained_furness(
            seed_vals=seed_matrix,
            row_targets=row_targets,
            col_targets=col_targets,
            tol=1e-6,
            max_iters=20,
            warning=False,
        )


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
