# -*- coding: utf-8 -*-
"""Core abstract functionality for gravity model classes to build on."""
# Built-Ins
import os
import abc
import logging
import warnings

from typing import Any

# Third Party
import numpy as np
import pandas as pd

from scipy import optimize
from caf.toolkit import pandas_utils as pd_utils

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import io
from caf.toolkit import timing
from caf.toolkit import math_utils
from caf.toolkit import cost_utils
from caf.distribute import cost_functions

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
class GravityModelBase(abc.ABC):
    """Base Class for gravity models.

    Contains any shared functionality needed across gravity model
    implementations.
    """

    # pylint: disable=too-many-instance-attributes

    # Class constants
    _avg_cost_col = "ave_km"  # Should be more generic
    _target_cost_distribution_cols = ["min", "max", "trips"] + [_avg_cost_col]
    _least_squares_method = "trf"

    def __init__(
        self,
        cost_function: cost_functions.CostFunction,
        cost_matrix: np.ndarray,
        target_cost_distribution: pd.DataFrame,
        running_log_path: os.PathLike,
        cost_min_max_buf: float = 0.1,
    ):
        # Validate attributes
        target_cost_distribution = pd_utils.reindex_cols(
            target_cost_distribution,
            self._target_cost_distribution_cols,
        )

        if running_log_path is not None:
            dir_name, _ = os.path.split(running_log_path)
            if not os.path.exists(dir_name):
                raise FileNotFoundError(
                    f"Cannot find the defined directory to write out a log. "
                    f"Given the following path: {dir_name}"
                )

            if os.path.isfile(running_log_path):
                warnings.warn(
                    f"Given a log path to a file that already exists. "
                    f"Logs will be appended to the end of the file at: "
                    f"{running_log_path}"
                )

        # Set attributes
        self.cost_function = cost_function
        self.cost_min_max_buf = cost_min_max_buf
        self.cost_matrix = cost_matrix
        self.target_cost_distribution = self._update_tcd(target_cost_distribution)
        self.tcd_bin_edges = self._get_tcd_bin_edges(target_cost_distribution)
        self.running_log_path = running_log_path

        # Running attributes
        self._loop_num: int = -1
        self._loop_start_time: float = -1.0
        self._loop_end_time: float = -1.0
        self._jacobian_mats: dict[str, np.ndarray] = dict()
        self._perceived_factors: np.ndarray = np.ones_like(self.cost_matrix)

        # Additional attributes
        self.initial_cost_params: dict[str, Any] = dict()
        self.optimal_cost_params: dict[str, Any] = dict()
        self.initial_convergence: float = 0
        self.achieved_convergence: float = 0
        self.achieved_band_share: np.ndarray = np.zeros_like(self.target_band_share)
        self.achieved_residuals: np.ndarray = np.full_like(self.target_band_share, np.inf)
        self.achieved_distribution: np.ndarray = np.zeros_like(cost_matrix)

    @property
    def target_band_share(self) -> np.ndarray:
        """The target band share from target cost distribution."""
        return self.target_cost_distribution["band_share"].values

    @staticmethod
    def _update_tcd(tcd: pd.DataFrame) -> pd.DataFrame:
        """Tidy up the cost distribution data where needed.

        Infills the ave_km column where values don't exist based in the middle
        of a bin.
        Converts the trips into band share, so a sum of all the values becomes 1 .
        """
        # Add in ave_km where needed
        tcd["ave_km"] = np.where(
            (tcd["ave_km"] == 0) | np.isnan(tcd["ave_km"]),
            tcd["min"],
            tcd["ave_km"],
        )

        # Generate the band shares using the given data
        tcd["band_share"] = tcd["trips"].copy()
        tcd["band_share"] /= tcd["band_share"].values.sum()

        return tcd

    @staticmethod
    def _get_tcd_bin_edges(target_cost_distribution: pd.DataFrame) -> list[float]:
        min_bounds = target_cost_distribution["min"].tolist()
        max_bounds = target_cost_distribution["max"].tolist()
        return [min_bounds[0]] + max_bounds

    def _initialise_calibrate_params(self) -> None:
        """Set running params to their default values for a run."""
        self._loop_num = 1
        self._loop_start_time = timing.current_milli_time()
        self.initial_cost_params = dict()
        self.initial_convergence = 0
        self._perceived_factors = np.ones_like(self.cost_matrix)

    def _cost_params_to_kwargs(self, args: list[Any]) -> dict[str, Any]:
        """Convert a list of args into kwargs that self.cost_function expects."""
        if len(args) != len(self.cost_function.kw_order):
            raise ValueError(
                f"Received the wrong number of args to convert to cost "
                f"function kwargs. Expected {len(self.cost_function.kw_order)} "
                f"args, but got {len(args)}."
            )

        return dict(zip(self.cost_function.kw_order, args))

    def _order_cost_params(self, params: dict[str, Any]) -> list[Any]:
        """Order params into a list that self.cost_function expects."""
        ordered_params = [0] * len(self.cost_function.kw_order)
        for name, value in params.items():
            index = self.cost_function.kw_order.index(name)
            ordered_params[index] = value

        return ordered_params

    def _order_init_params(self, init_params: dict[str, Any]) -> list[Any]:
        """Order init_params into a list that self.cost_function expects."""
        return self._order_cost_params(init_params)

    def _order_bounds(self) -> tuple[list[Any], list[Any]]:
        """Order min and max into a tuple of lists that self.cost_function expects."""
        min_vals = self._order_cost_params(self.cost_function.param_min)
        max_vals = self._order_cost_params(self.cost_function.param_max)

        min_vals = [x + self.cost_min_max_buf for x in min_vals]
        max_vals = [x - self.cost_min_max_buf for x in max_vals]

        return min_vals, max_vals

    def _cost_distribution(
        self,
        matrix: np.ndarray,
        tcd_bin_edges: list[float],
    ) -> np.ndarray:
        """Calculate the distribution of matrix across tcd_bin_edges."""
        _, normalised = cost_utils.normalised_cost_distribution(
            matrix=matrix,
            cost_matrix=self.cost_matrix,
            bin_edges=tcd_bin_edges,
        )
        return normalised

    def _guess_init_params(
        self,
        cost_args: list[float],
        target_cost_distribution: pd.DataFrame,
    ):
        """Guess the initial cost arguments.

        Internal function of _estimate_init_params().
        Used by the `optimize.least_squares` function.
        """
        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Used to optionally increase the cost of long distance trips
        avg_cost_vals = target_cost_distribution[self._avg_cost_col].values

        # Estimate what the cost function will do to the costs - on average
        estimated_cost_vals = self.cost_function.calculate(avg_cost_vals, **cost_kwargs)
        estimated_band_shares = estimated_cost_vals / estimated_cost_vals.sum()

        # return the residuals to the target
        return target_cost_distribution["band_share"].values - estimated_band_shares

    def _estimate_init_params(
        self,
        init_params: dict[str, Any],
        target_cost_distribution: pd.DataFrame,
    ) -> dict[str, Any]:
        """Guesses what the initial params should be.

        Uses the average cost in each band to estimate what changes in
        the cost_params would do to the final cost distributions. This is a
        very coarse-grained estimation, but can be used to guess around about
        where the best init params are.
        """
        result = optimize.least_squares(
            fun=self._guess_init_params,
            x0=self._order_init_params(init_params),
            method=self._least_squares_method,
            bounds=self._order_bounds(),
            kwargs={"target_cost_distribution": target_cost_distribution},
        )
        init_params = self._cost_params_to_kwargs(result.x)

        # TODO(BT): standardise this
        if self.cost_function.name == "LOG_NORMAL":
            init_params["sigma"] *= 0.8
            init_params["mu"] *= 0.5

        return init_params

    def _calculate_perceived_factors(self) -> None:
        """Update the perceived cost class variables.

        Compares the latest run of the gravity model (as defined by the
        variables: self.achieved_band_share)
        and generates a perceived cost factor matrix, which will be applied
        on calls to self._cost_amplify() in the gravity model.

        This function updates the _perceived_factors class variable.
        """
        # Init
        target_band_share = self.target_cost_distribution["band_share"].values

        # Calculate the adjustment per band in target band share.
        # Adjustment is clipped between 0.5 and 2 to limit affect
        perc_factors = (
            np.divide(
                self.achieved_band_share,
                target_band_share,
                where=target_band_share > 0,
                out=np.ones_like(self.achieved_band_share),
            )
            ** 0.5
        )
        perc_factors = np.clip(perc_factors, 0.5, 2)

        # Initialise loop
        perc_factors_mat = np.ones_like(self.cost_matrix)
        min_vals = self.target_cost_distribution["min"]
        max_vals = self.target_cost_distribution["max"]

        # Convert into factors for the cost matrix
        for min_val, max_val, factor in zip(min_vals, max_vals, perc_factors):
            # Get proportion of all trips that are in this band
            distance_mask = (self.cost_matrix >= min_val) & (self.cost_matrix < max_val)

            perc_factors_mat = np.multiply(
                perc_factors_mat,
                factor,
                where=distance_mask,
                out=perc_factors_mat,
            )

        # Assign to class attribute
        self._perceived_factors = perc_factors_mat

    def _apply_perceived_factors(self, cost_matrix: np.ndarray) -> np.ndarray:
        return cost_matrix * self._perceived_factors

    def _gravity_function(
        self,
        cost_args: list[float],
        diff_step: float,
    ):
        """Calculate residuals to the target cost distribution.

        Runs gravity model with given parameters and converts into achieved
        cost distribution. The residuals are then calculated between the
        achieved and the target.

        Used by the `optimize.least_squares` function.

        This function will populate and update:
            self.achieved_band_share
            self.achieved_convergence
            self.achieved_residuals
            self.achieved_distribution
            self.optimal_cost_params
        """
        # pylint: disable=too-many-locals
        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Used to optionally adjust the cost of long distance trips
        cost_matrix = self._apply_perceived_factors(self.cost_matrix)

        # Calculate initial matrix through cost function
        init_matrix = self.cost_function.calculate(cost_matrix, **cost_kwargs)

        # Do some prep for jacobian calculations
        # TODO(BT): Move this into the Jacobian function. We don't need it here
        #  and it's just using it memory before we need to. Could single loop it
        #  too, so that only one extra cost matrix is needed. NOT n_cost_params
        self._jacobian_mats = {"base": init_matrix.copy()}
        for cost_param in self.cost_function.kw_order:
            # Adjust cost slightly
            adj_cost_kwargs = cost_kwargs.copy()
            adj_cost_kwargs[cost_param] += adj_cost_kwargs[cost_param] * diff_step

            # Calculate adjusted cost
            adj_cost = self.cost_function.calculate(cost_matrix, **adj_cost_kwargs)

            self._jacobian_mats[cost_param] = adj_cost

        # Furness trips to trip ends
        matrix, iters, rmse = self.gravity_furness(seed_matrix=init_matrix)

        # Store for the jacobian calculations
        self._jacobian_mats["final"] = matrix.copy()

        # Convert matrix into an achieved distribution curve
        achieved_band_shares = self._cost_distribution(matrix, self.tcd_bin_edges)

        # Evaluate this run
        target_band_shares = self.target_cost_distribution["band_share"].values
        convergence = math_utils.curve_convergence(target_band_shares, achieved_band_shares)
        achieved_residuals = target_band_shares - achieved_band_shares

        # Calculate the time this loop took
        self._loop_end_time = timing.current_milli_time()
        time_taken = self._loop_end_time - self._loop_start_time

        # ## LOG THIS ITERATION ## #
        log_dict = {
            "loop_number": str(self._loop_num),
            "runtime (s)": time_taken / 1000,
        }
        log_dict.update(cost_kwargs)
        log_dict.update(
            {
                "furness_iters": iters,
                "furness_rmse": np.round(rmse, 6),
                "bs_con": np.round(convergence, 4),
            }
        )

        # Append this iteration to log file
        if self.running_log_path is not None:
            io.safe_dataframe_to_csv(
                pd.DataFrame(log_dict, index=[0]),
                self.running_log_path,
                mode="a",
                header=(not os.path.exists(self.running_log_path)),
                index=False,
            )

        # Update loop params and return the achieved band shares
        self._loop_num += 1
        self._loop_start_time = timing.current_milli_time()
        self._loop_end_time = -1

        # Update performance params
        self.achieved_band_share = achieved_band_shares
        self.achieved_convergence = convergence
        self.achieved_residuals = achieved_residuals
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
        ignore_result: bool = False,
    ):
        """Calculate the Jacobian for _gravity_function.

        Uses the matrices stored in self._jacobian_mats (which were stored in
        the previous call to self._gravity function) to estimate what a change
        in the cost parameters would do to final furnessed matrix. This is
        then formatted into a Jacobian for optimize.least_squares to use.

        Used by the `optimize.least_squares` function.
        """
        # pylint: disable=too-many-locals
        # Initialise the output
        n_bands = len(self.target_cost_distribution["band_share"].values)
        n_cost_params = len(cost_args)
        jacobian = np.zeros((n_bands, n_cost_params))

        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Estimate what the furness does to the matrix
        furness_factor = np.divide(
            self._jacobian_mats["final"],
            self._jacobian_mats["base"],
            where=self._jacobian_mats["base"] != 0,
            out=np.zeros_like(self._jacobian_mats["base"]),
        )

        # Estimate how the final matrix would be different with
        # different input cost parameters
        estimated_mats = dict.fromkeys(
            self.cost_function.kw_order,
            np.zeros_like(self._jacobian_mats["base"]),
        )
        for cost_param in self.cost_function.kw_order:
            # Estimate what the furness would have done
            furness_mat = self._jacobian_mats[cost_param] * furness_factor
            if furness_mat.sum() == 0:
                raise ValueError("estimated furness matrix total is 0")
            adj_weights = furness_mat / furness_mat.sum()
            adj_final = self._jacobian_mats["final"].sum() * adj_weights

            # Place in dictionary to send to Jacobian
            estimated_mats[cost_param] = adj_final

        # Control estimated matrices to final matrix
        controlled_mats = self.jacobian_furness(
            seed_matrices=estimated_mats,
            row_targets=self._jacobian_mats["final"].sum(axis=1),
            col_targets=self._jacobian_mats["final"].sum(axis=0),
            ignore_result=ignore_result,
        )

        # Calculate the Jacobian
        for i, cost_param in enumerate(self.cost_function.kw_order):
            # Turn into bands
            achieved_band_shares = self._cost_distribution(
                matrix=controlled_mats[cost_param],
                tcd_bin_edges=self.tcd_bin_edges,
            )

            # Calculate the Jacobian for this cost param
            jacobian_residuals = self.achieved_band_share - achieved_band_shares
            cost_step = cost_kwargs[cost_param] * diff_step
            cost_jacobian = jacobian_residuals / cost_step

            # Store in the Jacobian
            jacobian[:, i] = cost_jacobian

        return jacobian

    # calibrate_cost_params
    def _calibrate(
        self,
        init_params: dict[str, Any],
        calibrate_params: bool = True,
        diff_step: float = 1e-8,
        ftol: float = 1e-4,
        xtol: float = 1e-4,
        grav_max_iters: int = 100,
        failure_tol: float = 0,
        verbose: int = 0,
    ) -> None:
        """Calibrate the cost parameters to the optimum values.

        Runs the gravity model, and calibrates the optimal cost parameters
        if calibrate params is set to True. Will do a final run of the
        gravity_function with the optimal parameter found before return.
        """
        # Initialise running params
        self._initialise_calibrate_params()

        # Calculate the optimal cost parameters if we're calibrating
        if calibrate_params is True:
            # Build the kwargs, we need them a few times
            ls_kwargs = {
                "fun": self._gravity_function,
                "method": self._least_squares_method,
                "bounds": self._order_bounds(),
                "jac": self._jacobian_function,
                "verbose": verbose,
                "ftol": ftol,
                "xtol": xtol,
                "max_nfev": grav_max_iters,
                "kwargs": {"diff_step": diff_step},
            }

            # Can sometimes fail with infeasible arguments, workaround
            result = None

            try:
                ordered_init_params = self._order_init_params(init_params)
                result = optimize.least_squares(x0=ordered_init_params, **ls_kwargs)
            except ValueError as err:
                if "infeasible" in str(err):
                    LOG.info(
                        "Got the following error while trying to run "
                        "`optimize.least_squares()`. Will try again with the "
                        "`default_params`"
                    )
                else:
                    raise err

            # If performance was terrible, try again with default params
            failed = self.achieved_convergence <= failure_tol
            if result is not None and failed:
                # Not sure what's going on with pylint below, but it raises
                # both these errors for the log call
                # pylint: disable=logging-not-lazy, consider-using-f-string
                LOG.info(
                    "Performance wasn't great with the given `init_params`. "
                    "Achieved '%s', and the `failure_tol` "
                    "is set to %s. Trying again with the "
                    "`default_params`" % (self.achieved_convergence, failure_tol)
                )

            if result is None:
                result = optimize.least_squares(
                    x0=self._order_init_params(self.cost_function.default_params),
                    **ls_kwargs,
                )

            # Make sure we had a successful run
            if result is not None:
                optimal_params = result.x
            else:
                raise RuntimeError(
                    "No result has been set. Check the internal logic! This "
                    "shouldn't be possible."
                )

            # BACKLOG: Try random init_params as a final option

        else:
            optimal_params = self._order_init_params(init_params)

        # Run an optimal version of the gravity
        self.optimal_cost_params = self._cost_params_to_kwargs(optimal_params)
        self._gravity_function(optimal_params, diff_step=diff_step)

    @abc.abstractmethod
    def gravity_furness(
        self,
        seed_matrix: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        """Run a doubly constrained furness on the seed matrix.

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the gravity model.

        Parameters
        ----------
        seed_matrix:
            Initial values for the furness.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        raise NotImplementedError

    @abc.abstractmethod
    def jacobian_furness(
        self,
        seed_matrices: dict[str, np.ndarray],
        row_targets: np.ndarray,
        col_targets: np.ndarray,
        ignore_result: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run a doubly constrained furness on the seed matrix.

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the jacobian calculation.

        Parameters
        ----------
        seed_matrices:
            Dictionary of initial values for the furness.
            Keys are the name of the cost params which has been changed
            to get this new seed matrix.

        row_targets:
            The target values for the sum of each row.
            i.e. np.sum(seed_matrix, axis=1)

        col_targets:
            The target values for the sum of each column
            i.e. np.sum(seed_matrix, axis=0)

        ignore_result:
            Whether to ignore the return result or not. Useful when a Jacobian
            furness is only being called to satisfy other threads.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        raise NotImplementedError


# # # FUNCTIONS # # #
