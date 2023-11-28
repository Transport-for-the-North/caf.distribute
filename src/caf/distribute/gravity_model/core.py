# -*- coding: utf-8 -*-
"""Core abstract functionality for gravity model classes to build on."""
# Built-Ins
import abc
import dataclasses
import functools
import logging
import os
import warnings
from typing import Any, Optional, Union

# Third Party
import numpy as np
import pandas as pd
from caf.toolkit import cost_utils, io, timing
from scipy import optimize

# Local Imports
# pylint: disable=import-error,wrong-import-position
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
        if isinstance(cost_matrix, pd.DataFrame):
            self._perceived_factors: pd.DataFrame = pd.DataFrame(
                np.ones_like(self.cost_matrix), index=self.cost_matrix.index
            )
        else:
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

    @abc.abstractmethod
    def _gravity_function(
        self,
        cost_args: list[float],
        running_log_path: os.PathLike,
        target_cost_distribution: Optional[cost_utils.CostDistribution] = None,
        diff_step: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
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
        raise NotImplementedError("abstract method to be implemented in child class")

    @abc.abstractmethod
    def _jacobian_function(
        self,
        cost_args: list[float],
        diff_step: float,
        running_log_path: os.PathLike,
        target_cost_distribution: cost_utils.CostDistribution,
        **kwargs,
    ) -> np.ndarray:
        """Calculate the Jacobian for _gravity_function.

        The Jacobian is shape of (n_cost_bands, n_cost_args), where each index
        indicates the impact a slight change of a cost parameter has on a
        cost band.

        Used by the `optimize.least_squares` function.
        """
        raise NotImplementedError("abstract method to be implemented in child class")

    def _calibrate(
        self,
        init_params: dict[str, Any],
        running_log_path: os.PathLike,
        target_cost_distribution: cost_utils.CostDistribution,
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
        # pylint: disable=too-many-arguments, too-many-locals
        # Init
        if init_params == self.cost_function.default_params:
            default_retry = False

        # We use this a couple of times - ensure consistent calls
        gravity_kwargs: dict[str, Any] = {
            "running_log_path": running_log_path,
            "target_cost_distribution": target_cost_distribution,
            "diff_step": diff_step,
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

        # Run the optimisation process and log it
        ordered_init_params = self._order_cost_params(init_params)
        result = optimise_cost_params(x0=ordered_init_params)
        LOG.info(
            "%scalibration process ended with "
            "success=%s, and the following message:\n"
            "%s",
            self.unique_id,
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

        # Populate internal arguments with optimal run results.
        assert self.achieved_cost_dist is not None
        return GravityModelCalibrateResults(
            cost_distribution=self.achieved_cost_dist,
            cost_convergence=self.achieved_convergence,
            value_distribution=self.achieved_distribution,
            target_cost_distribution=target_cost_distribution,
            cost_function=self.cost_function,
            cost_params=self.optimal_cost_params,
        )

    def _calculate_initial_parameters(self) -> dict[str, Any]:
        ...
        # TODO(MB) Make an abstract method for calculating initial parameters
        # This already exists in SingleArea class

    def calibrate(
        self,
        init_params: dict[str, Any],
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
        self.cost_function.validate_params(init_params)
        self._validate_running_log(running_log_path)
        self._initialise_internal_params()
        return self._calibrate(  # type: ignore
            *args,
            init_params=init_params,
            running_log_path=running_log_path,
            **kwargs,
        )

    def calibrate_with_perceived_factors(
        self,
        init_params: dict[str, Any],
        running_log_path: os.PathLike,
        target_cost_distribution: cost_utils.CostDistribution,
        *args,
        failure_tol: float = 0.5,
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
            Also used to determine whether perceived factors should be used,
            passed to `cls._should_use_perceived_factors()`.
            See docs for further info

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
        `cls._should_use_perceived_factors()`
        """
        self.cost_function.validate_params(init_params)
        self._validate_running_log(running_log_path)
        self._initialise_internal_params()

        # Run as normal first
        results = self._calibrate(  # type: ignore
            *args,
            init_params=init_params,
            running_log_path=running_log_path,
            failure_tol=failure_tol,
            target_cost_distribution=target_cost_distribution,
            **kwargs,
        )

        # If performance not good enough, apply perceived factors
        should_use_perceived = self._should_use_perceived_factors(
            failure_tol, results.cost_convergence
        )
        if should_use_perceived:
            # Start with 1000 if perceived factor run
            self._attempt_id = 1000

            self._calculate_perceived_factors(
                target_cost_distribution, self.achieved_band_share
            )
            results = self._calibrate(  # type: ignore
                *args,
                init_params=results.cost_params,
                running_log_path=running_log_path,
                failure_tol=failure_tol,
                target_cost_distribution=target_cost_distribution,
                **kwargs,
            )
        return results

    def run_with_perceived_factors(
        self,
        cost_params: dict[str, Any],
        running_log_path: os.PathLike,
        target_cost_distribution: cost_utils.CostDistribution,
        target_cost_convergence: float = 0.9,
        **kwargs,
    ) -> GravityModelRunResults:
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
            An instance of GravityModelRunResults containing the
            results of this run.

        See Also
        --------
        `caf.distribute.furness.doubly_constrained_furness()`
        """
        # Init
        self._validate_running_log(running_log_path)
        self._initialise_internal_params()

        self._gravity_function(
            cost_args=self._order_cost_params(cost_params),
            running_log_path=running_log_path,
            target_cost_distribution=target_cost_distribution,
            **kwargs,
        )

        # Run again with perceived factors if good idea
        should_use_perceived = self._should_use_perceived_factors(
            target_cost_convergence, self.achieved_convergence
        )
        if should_use_perceived:
            # Start with 1000 if perceived factor run
            self._attempt_id = 1000
            self._calculate_perceived_factors(
                target_cost_distribution, self.achieved_band_share
            )
            self._gravity_function(
                cost_args=self._order_cost_params(cost_params),
                running_log_path=running_log_path,
                target_cost_distribution=target_cost_distribution,
                **kwargs,
            )

        assert self.achieved_cost_dist is not None
        return GravityModelRunResults(
            cost_distribution=self.achieved_cost_dist,
            cost_convergence=self.achieved_convergence,
            value_distribution=self.achieved_distribution,
            target_cost_distribution=target_cost_distribution,
            cost_function=self.cost_function,
            cost_params=cost_params,
        )

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

    def run(
        self,
        cost_params: dict[str, Any],
        running_log_path: os.PathLike,
        target_cost_distribution: Optional[cost_utils.CostDistribution] = None,
        **kwargs,
    ) -> GravityModelRunResults:
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
            An instance of GravityModelRunResults containing the
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
            cost_args=self._order_cost_params(cost_params),
            running_log_path=running_log_path,
            target_cost_distribution=target_cost_distribution,
            **kwargs,
        )

        assert self.achieved_cost_dist is not None
        return GravityModelRunResults(
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
