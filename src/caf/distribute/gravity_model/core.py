# -*- coding: utf-8 -*-
"""Core abstract functionality for gravity model classes to build on."""
# Built-Ins
import os
import abc
import logging
import warnings
import dataclasses

from typing import Any
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

from scipy import optimize

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import io
from caf.toolkit import timing
from caf.toolkit import cost_utils
from caf.distribute import cost_functions

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
@dataclasses.dataclass
class GravityModelResults:
    """A collection of results from a run of the Gravity Model.

    Parameters
    ----------
    cost_distribution:
        The achieved cost distribution of the run.

    cost_convergence:
        The achieved cost convergence value of the run. If
        `target_cost_distribution` is not set, then this should be 0.
        This will be the same as calculating the convergence of
        `cost_distribution` and `target_cost_distribution`.

    value_distribution:
        The achieved distribution of the given values (usually trip values
        between different places).

    target_cost_distribution:
        If set, this will be the cost distribution the gravity
        model was aiming for during its run.

    cost_function:
        If set, this will be the cost function used in the gravity model run.

    cost_params:
        If set, the cost parameters used with the cost_function to achieve
        the results.
    """

    cost_distribution: cost_utils.CostDistribution
    cost_convergence: float
    value_distribution: np.ndarray

    # Targets
    target_cost_distribution: Optional[cost_utils.CostDistribution] = None
    cost_function: Optional[cost_functions.CostFunction] = None
    cost_params: Optional[dict[str, Any]] = None


class GravityModelBase(abc.ABC):
    """Base Class for gravity models.

    Contains any shared functionality needed across gravity model
    implementations.
    """

    # pylint: disable=too-many-instance-attributes

    # Class constants
    _least_squares_method = "trf"

    def __init__(
        self,
        cost_function: cost_functions.CostFunction,
        cost_matrix: np.ndarray,
        cost_min_max_buf: float = 0.1,
    ):
        # Set attributes
        self.cost_function = cost_function
        self.cost_min_max_buf = cost_min_max_buf
        self.cost_matrix = cost_matrix

        # Running attributes
        self._loop_num: int = -1
        self._loop_start_time: float = -1.0
        # self._jacobian_mats: dict[str, np.ndarray] = dict()
        self._perceived_factors: np.ndarray = np.ones_like(self.cost_matrix)

        # Additional attributes
        self.initial_cost_params: dict[str, Any] = dict()
        self.optimal_cost_params: dict[str, Any] = dict()
        self.initial_convergence: float = 0
        self.achieved_convergence: float = 0
        self.achieved_cost_dist: Optional[cost_utils.CostDistribution] = None
        self.achieved_distribution: np.ndarray = np.zeros_like(cost_matrix)

    @property
    def achieved_band_share(self) -> np.ndarray:
        """The achieved band share values during the last run."""
        if self.achieved_cost_dist is None:
            raise ValueError("Gravity model has not been run. achieved_band_share is not set.")
        return self.achieved_cost_dist.band_share_vals

    @staticmethod
    def _validate_running_log(running_log_path: os.PathLike) -> None:
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

    def _initialise_internal_params(self) -> None:
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
        target_cost_distribution: cost_utils.CostDistribution,
    ):
        """Guess the initial cost arguments.

        Internal function of _estimate_init_params().
        Used by the `optimize.least_squares` function.
        """
        # Need kwargs for calling cost function
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Estimate what the cost function will do to the costs - on average
        avg_cost_vals = target_cost_distribution.avg_vals
        estimated_cost_vals = self.cost_function.calculate(avg_cost_vals, **cost_kwargs)
        estimated_band_shares = estimated_cost_vals / estimated_cost_vals.sum()

        return target_cost_distribution.band_share_vals - estimated_band_shares

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

    @staticmethod
    def _should_use_perceived_factors(
        target_convergence: float,
        achieved_convergence: float,
        warn: bool = True,
    ) -> bool:
        # Init
        upper_limit = target_convergence + 0.03
        lower_limit = target_convergence - 0.15

        # Upper limit beaten, all good
        if achieved_convergence > upper_limit:
            return False

        # Warn if the lower limit hasn't been reached
        if achieved_convergence < lower_limit:
            if warn:
                warnings.warn(
                    f"Lower threshold required to use perceived factors was "
                    f"not reached.\n"
                    f"Target convergence: {target_convergence}\n"
                    f"Lower Limit: {lower_limit}\n"
                    f"Achieved convergence: {achieved_convergence}"
                )
            return False

        return True

    @staticmethod
    def _log_iteration(
        log_path: os.PathLike,
        loop_num: int,
        loop_time: float,
        cost_kwargs: dict[str, Any],
        furness_iters: int,
        furness_rmse: float,
        convergence: float,
    ) -> None:
        """Write data from an iteration to a log file.

        Parameters
        ----------
        log_path:
            Path to the file to write the log to. Should be a csv file.

        loop_num:
            The iteration number ID

        loop_time:
            The time taken to complete this iteration.

        cost_kwargs:
            The cost values used in this iteration.

        furness_iters:
            The number of furness iterations completed before exit.

        furness_rmse:
            The achieved rmse score of the furness before exit.

        convergence:
            The achieved convergence values of the curve produced in this
            iteration.

        Returns
        -------
        None
        """
        log_dict = {
            "loop_number": str(loop_num),
            "runtime (s)": loop_time / 1000,
        }
        log_dict.update(cost_kwargs)
        log_dict.update(
            {
                "furness_iters": furness_iters,
                "furness_rmse": np.round(furness_rmse, 6),
                "bs_con": np.round(convergence, 4),
            }
        )

        # Append this iteration to log file
        if log_path is not None:
            io.safe_dataframe_to_csv(
                pd.DataFrame(log_dict, index=[0]),
                log_path,
                mode="a",
                header=(not os.path.exists(log_path)),
                index=False,
            )

    def _calculate_perceived_factors(
        self,
        target_cost_distribution: cost_utils.CostDistribution,
        achieved_band_shares: np.ndarray,
    ) -> None:
        """Update the perceived cost class variable.

        Compares the latest run of the gravity model (as defined by the
        variables: self.achieved_band_share) with the `target_cost_distribution`
        and generates a perceived cost factor matrix, which will be applied
        on calls to self._cost_amplify() in the gravity model.

        This function updates the _perceived_factors class variable.
        """
        # Calculate the adjustment per band in target band share.
        # Adjustment is clipped between 0.5 and 2 to limit affect
        perc_factors = (
            np.divide(
                achieved_band_shares,
                target_cost_distribution.band_share_vals,
                where=target_cost_distribution.band_share_vals > 0,
                out=np.ones_like(achieved_band_shares),
            )
            ** 0.5
        )
        perc_factors = np.clip(perc_factors, 0.5, 2)

        # Initialise loop
        perc_factors_mat = np.ones_like(self.cost_matrix)
        min_vals = target_cost_distribution.min_vals
        max_vals = target_cost_distribution.max_vals

        # Convert factors to matrix resembling the cost matrix
        for min_val, max_val, factor in zip(min_vals, max_vals, perc_factors):
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
        running_log_path: os.PathLike,
        target_cost_distribution: Optional[cost_utils.CostDistribution] = None,
        diff_step: float = 0.0,
        **kwargs,
    ):
        """Calculate residuals to the target cost distribution.

        Runs gravity model with given parameters and converts into achieved
        cost distribution. The residuals are then calculated between the
        achieved and the target.

        Used by the `optimize.least_squares` function.

        This function will populate and update:
            self.achieved_cost_dist
            self.achieved_convergence
            self.achieved_distribution
            self.optimal_cost_params
        """
        # Not used, but need for compatibility with self._jacobian_function
        del diff_step

        # Init
        cost_kwargs = self._cost_params_to_kwargs(cost_args)
        cost_matrix = self._apply_perceived_factors(self.cost_matrix)

        # Furness trips to trip ends
        matrix, iters, rmse = self.gravity_furness(
            seed_matrix=self.cost_function.calculate(cost_matrix, **cost_kwargs),
            **kwargs,
        )

        # Evaluate the performance of this run
        cost_distribution, achieved_residuals, convergence = cost_distribution_stats(
            achieved_trip_distribution=matrix,
            cost_matrix=cost_matrix,
            target_cost_distribution=target_cost_distribution,
        )

        # Log this iteration
        end_time = timing.current_milli_time()
        self._log_iteration(
            log_path=running_log_path,
            loop_num=str(self._loop_num),
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
        diff_step: float,
        cost_args: list[float],
        running_log_path: os.PathLike,
        target_cost_distribution: cost_utils.CostDistribution,
        **kwargs,
    ):
        """Calculate the Jacobian for _gravity_function.

        The Jacobian is shape of (n_cost_bands, n_cost_args), where each index
        indicates the impact a slight change of a cost parameter has on a
        cost band.

        Used by the `optimize.least_squares` function.
        """
        # pylint: disable=too-many-locals
        # Not used, but need for compatibility with self._gravity_function
        del running_log_path
        del kwargs

        # Initialise the output
        jacobian = np.zeros((target_cost_distribution.n_bins, len(cost_args)))

        # Initialise running params
        cost_kwargs = self._cost_params_to_kwargs(cost_args)
        cost_matrix = self._apply_perceived_factors(self.cost_matrix)
        row_targets = (self.achieved_distribution.sum(axis=1),)
        col_targets = (self.achieved_distribution.sum(axis=0),)

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
            adj_final = self.achieved_distribution * adj_weights

            # Finesse to match row / col targets
            adj_final = self.jacobian_furness(
                seed_matrix=adj_final,
                row_targets=row_targets,
                col_targets=col_targets,
            )

            # Calculate the Jacobian values for this cost param
            adj_band_share = self._cost_distribution(
                matrix=adj_final,
                tcd_bin_edges=target_cost_distribution.bin_edges,
            )
            jacobian_residuals = self.achieved_band_share - adj_band_share
            jacobian[:, i] = jacobian_residuals / cost_step

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
        self._initialise_internal_params()

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
                # TODO(BT): Can we make use of result.message and
                #  result.success? Reply useful messages?
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
        **kwargs,
    ) -> tuple[np.ndarray, int, float]:
        """Run a doubly constrained furness on the seed matrix.

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the gravity model.

        Parameters
        ----------
        seed_matrix:
            Initial values for the furness.

        kwargs:
            Additional arguments from the caller - allows arguments to be
            passed to this function.

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
        raise NotImplementedError

    def run_with_perceived_factors(
        self,
        cost_params: dict[str, Any],
        running_log_path: os.PathLike,
        target_cost_distribution: cost_utils.CostDistribution,
        target_cost_convergence: float = 0.9,
        **kwargs,
    ) -> GravityModelResults:
        """Run the gravity model with set cost parameters.

        This function will run a single iteration of the gravity model using
        the given cost parameters. It is similar to the default `run` function
        but uses perceived factors to try to improve the performance of the run.

        Perceived factors can be used to improve model
        performance. These factors slightly adjust the cost across
        bands to help nudge demand towards the expected distribution.
        These factors are only used when the performance is already
        reasonably good, otherwise they are ineffective. Only used when
        the achieved R^2 convergence meets the following criteria:
        `lower_bound = target_cost_convergence - 0.15`
        `upper_bound = target_cost_convergence + 0.03`
        `lower_bound < achieved_convergence < upper_bound`

        Parameters
        ----------
        cost_params:
            The cost parameters to use

        running_log_path:
            Path to output the running log to. This log will detail the
            performance of the run and is written in .csv format.

        target_cost_convergence:
            A value between 0 and 1. Ignored unless `use_perceived_factors`
            is set. Used to define the bounds withing which perceived factors
            can be used to improve final distribution.

        target_cost_distribution:
            If given,

        kwargs:
            Additional arguments passed to self.gravity_furness.
            Empty by default. The calling signature is:
            `self.gravity_furness(seed_matrix, **kwargs)`

        Returns
        -------
        results:
            An instance of GravityModelResults containing the
            results of this run.

        See Also
        --------
        `caf.distribute.furness.doubly_constrained_furness()`
        """
        # Init
        self._validate_running_log(running_log_path)
        self._initialise_internal_params()

        self._gravity_function(
            cost_args=self._order_init_params(cost_params),
            running_log_path=running_log_path,
            target_cost_distribution=target_cost_distribution,
            **kwargs,
        )

        # Run again with perceived factors if good idea
        should_use_perceived = self._should_use_perceived_factors(
            target_cost_convergence, self.achieved_convergence
        )
        if should_use_perceived:
            self._calculate_perceived_factors(
                target_cost_distribution, self.achieved_band_share
            )
            self._gravity_function(
                cost_args=self._order_init_params(cost_params),
                running_log_path=running_log_path,
                target_cost_distribution=target_cost_distribution,
                **kwargs,
            )

        return GravityModelResults(
            cost_distribution=self.achieved_cost_dist,
            cost_convergence=self.achieved_convergence,
            value_distribution=self.achieved_distribution,
            target_cost_distribution=target_cost_distribution,
            cost_function=self.cost_function,
            cost_params=cost_params,
        )

    def run(
        self,
        cost_params: dict[str, Any],
        running_log_path: os.PathLike,
        target_cost_distribution: Optional[cost_utils.CostDistribution] = None,
        **kwargs,
    ) -> GravityModelResults:
        """Run the gravity model with set cost parameters.

        This function will run a single iteration of the gravity model using
        the given cost parameters.

        Parameters
        ----------
        cost_params:
            The cost parameters to use

        running_log_path:
            Path to output the running log to. This log will detail the
            performance of the run and is written in .csv format.

        target_cost_distribution:
            If given, this is used to calculate the residuals in the return.
            The return cost_distribution will also use the same bins
            provided here.

        kwargs:
            Additional arguments passed to self.gravity_furness.
            Empty by default. The calling signature is:
            `self.gravity_furness(seed_matrix, **kwargs)`

        Returns
        -------
        results:
            An instance of GravityModelResults containing the
            results of this run. If a `target_cost_distribution` is not given,
            the returning results.cost_distribution will dynamically create
            its own bins; cost_residuals and cost_convergence will also
            contain dummy values.

        See Also
        --------
        `caf.distribute.furness.doubly_constrained_furness()`
        """
        # Init
        self._validate_running_log(running_log_path)
        self._initialise_internal_params()

        self._gravity_function(
            cost_args=self._order_init_params(cost_params),
            running_log_path=running_log_path,
            target_cost_distribution=target_cost_distribution,
            **kwargs,
        )

        return GravityModelResults(
            cost_distribution=self.achieved_cost_dist,
            cost_convergence=self.achieved_convergence,
            value_distribution=self.achieved_distribution,
            target_cost_distribution=target_cost_distribution,
            cost_function=self.cost_function,
            cost_params=cost_params,
        )


# # # FUNCTIONS # # #
def cost_distribution_stats(
    achieved_trip_distribution: np.ndarray,
    cost_matrix: np.ndarray,
    target_cost_distribution: Optional[cost_utils.CostDistribution] = None,
) -> tuple[cost_utils.CostDistribution, np.ndarray, float]:
    """Generate standard stats for a cost distribution performance.

    Parameters
    ----------
    achieved_trip_distribution:
        The achieved distribution of trips. Must be the same shape as
        `cost_matrix`.

    cost_matrix:
        A matrix describing the zone to zone costs. Must be the same shape as
        `achieved_trip_distribution`.

    target_cost_distribution:
        The cost distribution that `achieved_trip_distribution` and
        `cost_matrix` were aiming to recreate.

    Returns
    -------
    achieved_cost_distribution:
        The achieved cost distribution produced by `achieved_trip_distribution`
        and `cost_matrix`. If `target_cost_distribution` is given, this will
        use the same bins defined, otherwise dynamic bins will be selected.

    achieved_residuals:
        The residual difference between `achieved_cost_distribution` and
        `target_cost_distribution` band share values.
        Will be an array of np.inf if `target_cost_distribution` is not set.

    achieved_convergence:
        A float value between 0 and 1. Values closer to 1 indicate a better
        convergence. Will be -1 if `target_cost_distribution` is not set.

    """
    if target_cost_distribution is not None:
        cost_distribution = cost_utils.CostDistribution.from_data(
            matrix=achieved_trip_distribution,
            cost_matrix=cost_matrix,
            bin_edges=target_cost_distribution.bin_edges,
        )
        cost_residuals = target_cost_distribution.residuals(cost_distribution)
        cost_convergence = target_cost_distribution.convergence(cost_distribution)

    else:
        cost_distribution = cost_utils.CostDistribution.from_data_no_bins(
            matrix=achieved_trip_distribution,
            cost_matrix=cost_matrix,
        )
        cost_residuals = np.full_like(cost_distribution.band_share_vals, np.inf)
        cost_convergence = -1

    return cost_distribution, cost_residuals, cost_convergence
