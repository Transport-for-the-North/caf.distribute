# -*- coding: utf-8 -*-
"""Implementation of a self-calibrating single area gravity model."""
# Built-Ins
import functools
import logging
import os
from typing import Any, Optional

# Third Party
import numpy as np
from scipy import optimize
from caf.toolkit import cost_utils, timing, toolbox

# Local Imports
from caf.distribute import cost_functions, furness
from caf.distribute.gravity_model import (
    GravityModelCalibrateResults,
    GravityModelRunResults,
    core,
)


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
        """Use in the least_squares_optimiser to distribute demand."""
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
            attempt_id=self._attempt_id,  # type: ignore
            loop_num=self._loop_num,
            loop_time=(end_time - self._loop_start_time) / 1000,  # type: ignore
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
        if self.initial_cost_params is None:  # type: ignore
            self.initial_cost_params = cost_kwargs
        if self.initial_convergence is None:  # type: ignore
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
        """
        Use in the least squared optmimiser to produce a jacobian.

        The jacobian informs the optimiser which direction/how much to move
        parameters between iterations.
        """
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
            self._attempt_id += 1  # type: ignore
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
            self._attempt_id += 100  # type: ignore
            for i in range(n_random_tries):
                self._attempt_id += i  # type: ignore
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
