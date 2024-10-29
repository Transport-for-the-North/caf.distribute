# -*- coding: utf-8 -*-
"""Implementation of a self-calibrating single area gravity model."""
from __future__ import annotations

# Built-Ins
import functools
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

# Third Party
import numpy as np
import pandas as pd
from caf.toolkit import BaseConfig, cost_utils, timing
from scipy import optimize

# Local Imports
from caf.distribute import cost_functions, furness
from caf.distribute.gravity_model import core
from caf.distribute.gravity_model.core import GravityModelCalibrateResults

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)
DEFAULT_FURNESS_TOL = 1e-6


# pylint:disable=duplicate-code
# Furness called with same parameters in single and multi-area
# # # CLASSES # # #
class MultiDistInput(BaseConfig):
    """
    Input to multi cost distribution calibrator.

    Parameters
    ----------
    tld_file: Path
        Path to a file containing distributions. This should contain 5 columns,
        the names of which must be specified below.
    tld_lookup_file: Path
        Path to a lookup from distribution areas to zones. Should contain 2
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

    tld_file: Path
    tld_lookup_file: Path
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
class GMCalibParams:
    furness_jac: bool = False
    diff_step: float = 1e-8
    ftol: float = 1e-4
    xtol: float = 1e-4
    furness_tol: float = DEFAULT_FURNESS_TOL
    grav_max_iters: int = 100
    failure_tol: float = 0
    default_retry: bool = True


@dataclass
class MultiCostDistribution:
    """Cost distributions to be used for the multi-cost distribution gravity model

    Parameters
    ----------
    distributions: list[MGMCostDistribution]
        Distributions to be used for the multicost distributions
    """

    distributions: list[MGMCostDistribution]

    @classmethod
    def from_pandas(
        cls,
        ordered_zones: pd.Series,
        tld: pd.DataFrame,
        cat_zone_correspondence: pd.DataFrame,
        func_params: dict[int | str, dict[str, float]],
        *,
        tld_cat_col: str = "category",
        tld_min_col: str = "from",
        tld_max_col: str = "to",
        tld_avg_col: str = "av_distance",
        tld_trips_col: str = "trips",
        lookup_cat_col: str = "category",
        lookup_zone_col: str = "zone_id",
    ) -> MultiCostDistribution:
        """constructor using pandas dataframes

        Parameters
        ----------
        ordered_zones : pd.Series
            list of zones in the same order as other inputs
        tld : pd.DataFrame
            tld data - should contain the tlds for each distribution
            labeled by the `tld_cat_col`
        cat_zone_correspondence : pd.DataFrame
            lookup between categories values within `tld` and zones which
            use the corresponding distribution
        func_params : dict[int  |  str, dict[str, float]]
            starting/run cost function params to use for each distribution
            key: distribution category,  value: dict[param name, param value]
        tld_cat_col : str, optional
            column name for the category column in `tld`, by default "category"
        tld_min_col : str, optional
            column name for the min bin edge column in `tld`, by default "from"
        tld_max_col : str, optional
            column name for the max bin edge column in `tld`, by default "to"
        tld_avg_col : str, optional
            column name for the average distance column in `tld`, by default "av_distance"
        tld_trips_col : str, optional
            column name for the trips column in `tld`, by default "trips"
        lookup_cat_col : str, optional
            column name for the category column in `cat_zone_correspondence`, by default "category"
        lookup_zone_col : str, optional
            column name for the zone column in `cat_zone_correspondence`, by default "zone_id"

        Returns
        -------
        MultiCostDistribution


        Raises
        ------
        KeyError
            when a category value is not founf in the function parameter keys

        See Also
        --------
        `validate`
        """

        distributions: list[MGMCostDistribution] = []

        for category in cat_zone_correspondence[lookup_cat_col].unique():
            if category not in func_params:
                raise KeyError(f"function parameters not provided for {category = }")
            distributions.append(
                MGMCostDistribution.from_pandas(
                    category,
                    pd.Series(ordered_zones),
                    tld,
                    cat_zone_correspondence,
                    func_params[category],
                    tld_cat_col=tld_cat_col,
                    tld_min_col=tld_min_col,
                    tld_max_col=tld_max_col,
                    tld_avg_col=tld_avg_col,
                    tld_trips_col=tld_trips_col,
                    lookup_cat_col=lookup_cat_col,
                    lookup_zone_col=lookup_zone_col,
                )
            )

        cls.validate(distributions)

        return cls(distributions)

    @classmethod
    def validate(cls, distributions: list[MGMCostDistribution]):
        """Validates the distributions passed

        Parameters
        ----------
        distributions : list[MGMCostDistribution]
            Distributions to validate

        Raises
        ------
        ValueError
            The length of the list of the distributions passed is 0
        ValueError
            The same  zones are found in multiple distributions
        """

        if len(distributions) == 0:
            raise ValueError("no distributions provided")

        all_zones: Optional[np.ndarray] = None

        for dist in distributions:
            if all_zones is None:
                all_zones = dist.zones
            else:
                all_zones = np.concatenate((all_zones, dist.zones))

        assert all_zones is not None

        if len(np.unique(all_zones)) != len(all_zones):
            raise ValueError("duplicate found in the distribution zone definition")

    def __iter__(self) -> Iterator[MGMCostDistribution]:
        yield from self.distributions

    def __getitem__(self, x) -> MGMCostDistribution:
        return self.distributions[x]

    def __len__(self) -> int:
        return len(self.distributions)

    def copy(self) -> MultiCostDistribution:
        return MultiCostDistribution(self.distributions)


@dataclass
class MGMCostDistribution:
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

    # cost_distribution: dict[id, cost_utils.CostDistribution]
    # matrix_id_lookup: np.ndarray
    # function_params: dict[id, dict[str,float]]

    name: str | int
    cost_distribution: cost_utils.CostDistribution
    zones: np.ndarray
    function_params: dict[str, float]

    # TODO(kf) validate params

    # TODO(kf) validate cost distributions

    @classmethod
    def from_pandas(
        cls,
        category: str | int,
        ordered_zones: pd.Series,
        tld: pd.DataFrame,
        cat_zone_correspondence: pd.DataFrame,
        func_params: dict[str, float],
        *,
        tld_cat_col: str = "category",
        tld_min_col: str = "from",
        tld_max_col: str = "to",
        tld_avg_col: str = "av_distance",
        tld_trips_col: str = "trips",
        lookup_cat_col: str = "category",
        lookup_zone_col: str = "zone_id",
    ) -> MGMCostDistribution:
        """constructor that uses pandas dataframes and series

        Parameters
        ----------
        category : str | int
            distribution category, used to label gravity model run
        ordered_zones : pd.Series
            zones ordered in the same way as other inputs
        tld : pd.DataFrame
            tld data - should contain the tlds for each distribution
            labeled by the `tld_cat_col`
        cat_zone_correspondence : pd.DataFrame
            lookup between categories values within `tld` and zones which
            use the corresponding distribution
        func_params : dict[int  |  str, dict[str, float]]
            starting/run cost function params to use for each distribution
            key: distribution category,  value: dict[param name, param value]
        tld_cat_col : str, optional
            column name for the category column in `tld`, by default "category"
        tld_min_col : str, optional
            column name for the min bin edge column in `tld`, by default "from"
        tld_max_col : str, optional
            column name for the max bin edge column in `tld`, by default "to"
        tld_avg_col : str, optional
            column name for the average distance column in `tld`, by default "av_distance"
        tld_trips_col : str, optional
            column name for the trips column in `tld`, by default "trips"
        lookup_cat_col : str, optional
            column name for the category column in `cat_zone_correspondence`, by default "category"
        lookup_zone_col : str, optional
            column name for the zone column in `cat_zone_correspondence`, by default "zone_id"

        Returns
        -------
        MGMCostDistribution

        Raises
        ------
        ValueError
            if zones in `cat_zone_correspondence` are not present in `ordered_zones`
        """
        # get a list of zones that use this category of TLD
        cat_zones = cat_zone_correspondence.loc[
            cat_zone_correspondence[lookup_cat_col] == category, lookup_zone_col
        ].to_numpy()

        zones = ordered_zones.to_numpy()

        # tell user if we have zones in cat->lookup that arent in zones
        if not np.all(np.isin(cat_zones, zones)):
            missing_values = cat_zones[~np.isin(cat_zones, zones)]
            raise ValueError(
                f"The following values from cat->zone lookup are not present in the tld zones: {missing_values}"
            )

        # get the indices
        cat_zone_indices = np.where(np.isin(zones, cat_zones))[0]

        # get tld for cat
        cat_tld = tld[tld[tld_cat_col] == category]

        cat_cost_distribution = cost_utils.CostDistribution(
            cat_tld, tld_min_col, tld_max_col, tld_avg_col, tld_trips_col, tld_avg_col
        )

        return cls(category, cat_cost_distribution, cat_zone_indices, func_params)


class MultiAreaGravityModelCalibrator(core.GravityModelBase):
    """
    A self-calibrating multi-area gravity model.

    Parameters
    ----------
    row_targets: np.ndarray
        The targets for each row that the gravity model should be aiming to
        match. This can alternatively be thought of as the rows that wish to
        be distributed.

    col_targets: np.ndarray
        The targets for each column that the gravity model should be
        aiming to match. This can alternatively be thought of as the
        columns that wish to be distributed.

    cost_matrix: np.ndarray
        A matrix detailing the cost between each and every zone. This
        matrix must be the same size as
        `(len(row_targets), len(col_targets))`.

    cost_function: cost_functions.CostFunction
        The cost function to use when calibrating the gravity model. This
        function is applied to `cost_matrix` before Furnessing during
        calibration.
    """

    def __init__(
        self,
        row_targets: np.ndarray,
        col_targets: np.ndarray,
        cost_matrix: np.ndarray,
        cost_function: cost_functions.CostFunction,
        # TODO move these parameters as inputs of calibrate and run
    ):
        super().__init__(cost_function=cost_function, cost_matrix=cost_matrix)

        if row_targets.sum() != col_targets.sum():
            warnings.warn(
                "row and column target totals do not match. This is likely to cause Furnessing to fail."
                f" Difference (row targets - col targets) = {round(row_targets.sum() - col_targets.sum(),2)}"
            )

        checks = {
            "cost matrix": cost_matrix,
            "row targets": row_targets,
            "column targets": col_targets,
        }

        for name, data in checks.items():
            if np.isnan(data).any():
                raise ValueError(f"There are NaNs in {name}")
            if np.isinf(data).any():
                raise ValueError(f"There are Infs in {name}")

            num_zeros = (data == 0).sum()  # casting bool as 1, 0

            LOG.info(
                "There are %s 0s in %s (%s percent)", num_zeros, name, (num_zeros / data.size) * 100
            )

        zero_in_both = np.stack([row_targets == 0, col_targets == 0], axis=1).all(axis=1).sum()

        LOG.info("There are %s zones with both 0 row and column targets.", zero_in_both)

        self.row_targets = row_targets
        self.col_targets = col_targets
        if len(row_targets) != cost_matrix.shape[0]:
            raise IndexError("row_targets doesn't match cost_matrix")
        if len(col_targets) != cost_matrix.shape[1]:
            raise IndexError("col_targets doesn't match cost_matrix")

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
    def calibrate(
        self,
        distributions: MultiCostDistribution,
        running_log_path: Path,
        gm_params: GMCalibParams,
        verbose: int = 0,
        **kwargs,
    ) -> dict[str | int, GravityModelCalibrateResults]:
        """Find the optimal parameters for self.cost_function.

        Optimal parameters are found using `scipy.optimize.least_squares`
        to fit the distributed row/col targets to `target_cost_distribution`.

        NOTE: The achieved distribution is found by accessing self.achieved
        distribution of the object this method is called on. The output of
        this method shows the distribution and results for each individual TLD.

        Parameters
        ----------
        distributions: MultiCostDistribution
            distributions to use for the calibrations
        running_log_path: os.PathLike,
            path to a csv to log the model iterations and results
        gm_params: GMCalibParams
            defines the detailed parameters, see `GMCalibParams` documentation for more info
        *args,
        **kwargs,
        Returns
        -------
        dict[str | int, GravityModelCalibrateResults]:
            containings the achieved distributions for each tld category. To access
            the combined distribution use self.achieved_distribution

        See Also
        --------
        `caf.distribute.furness.doubly_constrained_furness()`
        `scipy.optimize.least_squares()`
        `caf.distribute.gravity_model.multi_area.GMCalibParams`
        """

        self._validate_running_log(running_log_path)
        self._initialise_internal_params()

        params_len = len(distributions[0].function_params)
        ordered_init_params = []

        for dist in distributions:
            self.cost_function.validate_params(dist.function_params)
            params = self._order_cost_params(dist.function_params)
            for val in params:
                ordered_init_params.append(val)

            max_binning = dist.cost_distribution.max_vals.max()
            min_binning = dist.cost_distribution.min_vals.min()

            max_cost = self.cost_matrix[dist.zones, :].max()
            min_cost = self.cost_matrix[dist.zones, :].min()

            if max_cost > max_binning:
                LOG.warning(
                    f"the maximum cost in the cost matrix for"
                    f" category {dist.name}, was {max_cost}, "
                    "whereas the highest bin edge in cost"
                    f" distribution was {max_binning}, "
                    "you will not be fitting to trips"
                    " with a cost greater than the binning"
                )
            if min_cost < min_binning:
                warnings.warn(
                    f"the min cost in the cost matrix for"
                    f" category {dist.name}, was {min_cost}, "
                    "whereas the lowest bin edge in cost"
                    f" distribution was {min_binning}, "
                    "you will not be fitting to trips"
                    " with a cost less than the binning"
                )

        gravity_kwargs: dict[str, Any] = {
            "running_log_path": running_log_path,
            "cost_distributions": distributions,
            "diff_step": gm_params.diff_step,
            "params_len": params_len,
            "furness_jac": gm_params.furness_jac,
            "furness_tol": gm_params.furness_tol,
        }
        optimise_cost_params = functools.partial(
            optimize.least_squares,
            fun=self._gravity_function,
            method=self._least_squares_method,
            bounds=(
                self._order_bounds()[0] * len(distributions),
                self._order_bounds()[1] * len(distributions),
            ),
            jac=self._jacobian_function,
            verbose=verbose,
            ftol=gm_params.ftol,
            xtol=gm_params.xtol,
            max_nfev=gm_params.grav_max_iters,
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

        if (
            not all(self.achieved_convergence) >= gm_params.failure_tol
        ) and gm_params.default_retry:
            LOG.info(
                "%sachieved a convergence of %s, "
                "however the failure tolerance is set to %s. Trying again with "
                "default cost parameters.",
                self.unique_id,
                self.achieved_convergence,
                gm_params.failure_tol,
            )
            self._attempt_id += 1
            ordered_init_params = self._order_cost_params(self.cost_function.default_params)
            result = optimise_cost_params(x0=ordered_init_params)

            # Update the best params only if this was better
            if np.mean(list(self.achieved_convergence.values())) > np.mean(
                list(best_convergence.values())
            ):
                best_params = result.x

        self._attempt_id: int = -2
        self._gravity_function(
            init_params=best_params,
            **(gravity_kwargs | kwargs),
        )

        assert self.achieved_cost_dist is not None
        results = {}
        for i, dist in enumerate(distributions):
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

    def _jacobian_function(
        self,
        init_params: list[float],
        cost_distributions: MultiCostDistribution,
        furness_tol: int,
        diff_step: float,
        furness_jac: bool,
        running_log_path: Path,
        params_len: int,
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
                if furness_jac:
                    adj_dist, *_ = furness.doubly_constrained_furness(
                        seed_vals=adj_dist,
                        row_targets=self.achieved_distribution.sum(axis=1),
                        col_targets=self.achieved_distribution.sum(axis=0),
                        tol=furness_tol / 10,
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
        self,
        init_params: dict[str, float],
        cost_distributions: MGMCostDistribution,
        furness_tol: float,
        running_log_path: os.PathLike,
        params_len: int,
        diff_step: int = 0,
        **_,
    ):
        del diff_step

        base_mat = self._create_seed_matrix(cost_distributions, init_params, params_len)
        matrix, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=base_mat,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            tol=furness_tol,
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
    def run(
        self,
        distributions: MultiCostDistribution,
        running_log_path: Path,
        furness_tol: float = DEFAULT_FURNESS_TOL,
    ) -> dict[int | str, GravityModelCalibrateResults]:
        """
        Run the gravity_model without calibrating.

        This should be done when you have calibrating previously to find the
        correct parameters for the cost function.

        Parameters
        ----------
        distributions : MultiCostDistribution
            Distributions to use to run the gravity model
        running_log_path : Path
            Csv path to log results and info
        furness_tol : float, optional
            tolerance for difference in target and achieved value,
            at which to stop furnessing, by default DEFAULT_FURNESS_TOL

        """
        params_len = len(distributions[0].function_params)
        cost_args = []
        for dist in distributions:
            for param in dist.function_params.values():
                cost_args.append(param)

        self._gravity_function(
            init_params=cost_args,
            cost_distributions=distributions,
            running_log_path=running_log_path,
            params_len=params_len,
            furness_tol=furness_tol,
        )

        assert self.achieved_cost_dist is not None
        results = {}
        for i, dist in enumerate(distributions):
            result_i = GravityModelCalibrateResults(
                cost_distribution=self.achieved_cost_dist[i],
                cost_convergence=self.achieved_convergence[dist.name],
                value_distribution=self.achieved_distribution[dist.zones],
                target_cost_distribution=dist.cost_distribution,
                cost_function=self.cost_function,
                cost_params=self._cost_params_to_kwargs(
                    cost_args[i * params_len : i * params_len + params_len]
                ),
            )

            results[dist.name] = result_i
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
