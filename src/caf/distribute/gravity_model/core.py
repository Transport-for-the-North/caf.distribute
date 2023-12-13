# -*- coding: utf-8 -*-
"""Core abstract functionality for gravity model classes to build on."""
# Built-Ins
import abc
import dataclasses
import logging
import os
import warnings
from typing import Any, Optional

# Third Party
import numpy as np
import pandas as pd
from scipy import optimize
from matplotlib import pyplot as plt
from caf.toolkit import cost_utils, io, timing
from caf.distribute import cost_functions

# Local Imports
# pylint: disable=import-error,wrong-import-position


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
        The cost distribution the gravity model was aiming for during its run.

    cost_function:
        The cost function used in the gravity model run.

    cost_params:
        The cost parameters used with the cost_function to achieve the results.
    """

    cost_distribution: cost_utils.CostDistribution
    cost_convergence: float
    value_distribution: np.ndarray


@dataclasses.dataclass
class GravityModelCalibrateResults(GravityModelResults):
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
        The cost distribution the gravity model was aiming for during its run.

    cost_function:
        The cost function used in the gravity model run.

    cost_params:
        The cost parameters used with the cost_function to achieve the results.
    """

    # Targets
    target_cost_distribution: cost_utils.CostDistribution
    cost_function: cost_functions.CostFunction
    cost_params: dict[str, Any]

    def plot_distributions(self):
        """
        Plot a comparison of the achieved and target distributions.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        df_1 = self.cost_distribution.df
        df_1["normalised"] = (
            df_1[self.target_cost_distribution.trips_col]
            / df_1[self.target_cost_distribution.trips_col].sum()
        )
        df_2 = self.target_cost_distribution.df
        df_2["normalised"] = (
            df_2[self.cost_distribution.trips_col]
            / df_2[self.cost_distribution.trips_col].sum()
        )
        ax.bar(
            df_1[self.cost_distribution.avg_col],
            df_1["normalised"],
            width=df_1[self.cost_distribution.max_col] - df_1[self.cost_distribution.min_col],
            label="Achieved Distribution",
            color="blue",
            alpha=0.7,
        )
        ax.bar(
            df_1[self.cost_distribution.avg_col],
            df_2["normalised"],
            width=df_2[self.target_cost_distribution.max_col]
            - df_2[self.target_cost_distribution.min_col],
            label="Target Distribution",
            color="orange",
            alpha=0.7,
        )

        ax.set_xlabel("Cost")
        ax.set_ylabel("Trips")
        ax.set_title("Distribution Comparison")
        ax.legend()

        return fig


@dataclasses.dataclass
class GravityModelRunResults(GravityModelResults):
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
        unique_id: str = "",
    ):
        # Set attributes
        self.cost_function = cost_function
        self.cost_min_max_buf = cost_min_max_buf
        self.cost_matrix = cost_matrix
        self.unique_id = self._tidy_unique_id(unique_id)

        # Running attributes
        self._attempt_id: int = -1
        self._loop_num: int = -1
        self._loop_start_time: float = -1.0
        self._perceived_factors: np.ndarray = np.ones_like(self.cost_matrix)

        # Additional attributes
        self.initial_cost_params: dict[str, Any] = dict()
        self.optimal_cost_params: dict[str, Any] = dict()
        self.initial_convergence: float = 0
        self.achieved_convergence: float = 0
        self.achieved_cost_dist: Optional[cost_utils.CostDistribution] = None
        self.achieved_distribution: np.ndarray = np.zeros_like(cost_matrix)

    @staticmethod
    def _tidy_unique_id(unique_id: str) -> str:
        """Format the unique_id for internal use."""
        unique_id = unique_id.strip()
        if unique_id == "":
            return unique_id
        return f"{unique_id} "

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
        self._attempt_id = 1
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

    def _order_bounds(self) -> tuple[list[Any], list[Any]]:
        """Order min and max into a tuple of lists that self.cost_function expects."""
        min_vals = self._order_cost_params(self.cost_function.param_min)
        max_vals = self._order_cost_params(self.cost_function.param_max)

        min_vals = [x + self.cost_min_max_buf for x in min_vals]
        max_vals = [x - self.cost_min_max_buf for x in max_vals]

        return min_vals, max_vals

    @staticmethod
    def _should_use_perceived_factors(
        target_convergence: float,
        achieved_convergence: float,
        upper_tol: float = 0.03,
        lower_tol: float = 0.15,
        warn: bool = True,
    ) -> bool:
        """Decide whether to use perceived factors.

        Parameters
        ----------
        target_convergence:
            The desired convergence target.

        achieved_convergence
            The current best achieved convergence.

        upper_tol:
            The upper tolerance to apply to `target_convergence` when
            calculating the upper limit it is acceptable to apply perceived
            factors.

        lower_tol:
            The lower tolerance to apply to `target_convergence` when
            calculating the lower limit it is acceptable to apply perceived
            factors.

        warn:
            Whether to raise a warning when the achieved convergence is too
            low to apply perceived factors.
            i.e. `achieved_convergence` < `target_convergence - lower_tol`

        Returns
        -------
        bool:
            True if
            `target_convergence - lower_tol` < `achieved_convergence` < `target_convergence + upper_tol`
        """
        # Init
        upper_limit = target_convergence + upper_tol
        lower_limit = target_convergence - lower_tol

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
        attempt_id: int,
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

        attempt_id:
            Identifier indicating which section of a run / calibration the
            current log refers to.
            # TODO(BT): Detail what each number means.

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
            "attempt_id": str(attempt_id),
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

    def estimate_optimal_cost_params(
        self,
        init_params: dict[str, Any],
        target_cost_distribution: cost_utils.CostDistribution,
    ) -> dict[str, Any]:
        """Guesses what the initial params should be.

        Uses the average cost in each band to estimate what changes in
        the cost_params would do to the final cost distributions. This is a
        very coarse-grained estimation, but can be used to guess around about
        where the best init params are.
        """
        result = optimize.least_squares(
            fun=self._guess_init_params,
            x0=self._order_cost_params(init_params),
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
    # TODO(MB) Calculate extra stats / metrics
    # r squared, ratio of coincidence, possible others?
    if target_cost_distribution is not None:
        cost_distribution = cost_utils.CostDistribution.from_data(
            matrix=achieved_trip_distribution,
            cost_matrix=cost_matrix,
            bin_edges=target_cost_distribution.bin_edges,
        )
        cost_residuals = target_cost_distribution.band_share_residuals(cost_distribution)
        cost_convergence = target_cost_distribution.band_share_convergence(cost_distribution)

    else:
        cost_distribution = cost_utils.CostDistribution.from_data_no_bins(
            matrix=achieved_trip_distribution,
            cost_matrix=cost_matrix,
        )
        cost_residuals = np.full_like(cost_distribution.band_share_vals, np.inf)
        cost_convergence = -1

    return cost_distribution, cost_residuals, cost_convergence
