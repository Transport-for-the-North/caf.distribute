# -*- coding: utf-8 -*-
"""Tests for the gravity_model.single_area module"""
from __future__ import annotations

# Built-Ins
import os
import json
import pathlib
import dataclasses

from typing import Any

# Third Party
import pytest
import numpy as np


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import cost_utils
from caf.distribute import cost_functions
from caf.distribute.gravity_model import GravityModelResults
from caf.distribute.gravity_model import SingleAreaGravityModelCalibrator

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
TEST_DATA_PATH = pathlib.Path(__file__).parent.resolve() / "data"


# # # Classes # # #
@dataclasses.dataclass
class GMCreator:
    """Can create a gravity mode class from disk"""

    row_targets: np.ndarray
    col_targets: np.ndarray
    cost_function: cost_functions.CostFunction
    cost_matrix: np.ndarray
    target_cost_distribution: cost_utils.CostDistribution
    running_log_path: os.PathLike

    @staticmethod
    def _read_row_targets(home: pathlib.Path) -> np.ndarray:
        return np.loadtxt(home / "row_targets.csv", delimiter=",")

    @staticmethod
    def _read_col_targets(home: pathlib.Path) -> np.ndarray:
        return np.loadtxt(home / "col_targets.csv", delimiter=",")

    @staticmethod
    def _read_cost_matrix(home: pathlib.Path) -> np.ndarray:
        return np.loadtxt(home / "cost_matrix.csv", delimiter=",")

    @staticmethod
    def _read_cost_distribution(home: pathlib.Path) -> cost_utils.CostDistribution:
        path = home / "target_cost_distribution.csv"
        return cost_utils.CostDistribution.from_file(path)

    @classmethod
    def get_common_constructor_kwargs(cls, path: pathlib.Path) -> dict[str, Any]:
        return {
            "row_targets": cls._read_row_targets(path),
            "col_targets": cls._read_col_targets(path),
            "cost_matrix": cls._read_cost_matrix(path),
            "target_cost_distribution": cls._read_cost_distribution(path),
        }

    @classmethod
    def from_file(
        cls,
        path: pathlib.Path,
        running_log_path: os.PathLike,
        cost_function: cost_functions.CostFunction,
    ) -> GMCreator:
        """Load data from files to create this test"""
        return cls(
            cost_function=cost_function,
            running_log_path=running_log_path,
            **cls.get_common_constructor_kwargs(path),
        )

    def create_gravity_model(self) -> SingleAreaGravityModelCalibrator:
        return SingleAreaGravityModelCalibrator(
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            cost_function=self.cost_function,
            cost_matrix=self.cost_matrix,
        )

    def create_and_run_gravity_model(
        self,
        cost_params: dict[str, Any],
        target_convergence: float = 0.9,
        furness_max_iters: int = 1000,
        furness_tol: float = 1e-3,
        use_perceived_factors: bool = False,
    ) -> GravityModelResults:
        gm = SingleAreaGravityModelCalibrator(
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            cost_function=self.cost_function,
            cost_matrix=self.cost_matrix,
        )

        if use_perceived_factors:
            return gm.run_with_perceived_factors(
                cost_params=cost_params,
                running_log_path=self.running_log_path,
                target_cost_distribution=self.target_cost_distribution,
                target_cost_convergence=target_convergence,
                max_iters=furness_max_iters,
                tol=furness_tol,
            )

        return gm.run(
            cost_params=cost_params,
            running_log_path=self.running_log_path,
            target_cost_distribution=self.target_cost_distribution,
            max_iters=furness_max_iters,
            tol=furness_tol,
        )


@dataclasses.dataclass
class GMCalibrateResults(GMCreator):
    """Stores the expected results alongside the inputs for calibration"""

    convergence: float
    band_share: np.ndarray
    distribution: np.ndarray
    residuals: np.ndarray
    best_params: dict[str, Any]

    @staticmethod
    def _read_convergence(home: pathlib.Path) -> np.ndarray:
        return np.loadtxt(home / "convergence.csv", delimiter=",")

    @staticmethod
    def _read_band_share(home: pathlib.Path) -> np.ndarray:
        return np.loadtxt(home / "band_share.csv", delimiter=",")

    @staticmethod
    def _read_distribution(home: pathlib.Path) -> np.ndarray:
        return np.loadtxt(home / "distribution.csv", delimiter=",")

    @staticmethod
    def _read_residuals(home: pathlib.Path) -> np.ndarray:
        return np.loadtxt(home / "residuals.csv", delimiter=",")

    @staticmethod
    def _read_best_params(home: pathlib.Path) -> np.ndarray:
        with open(home / "best_params.json", "r") as fp:
            return json.load(fp)

    @classmethod
    def get_specific_constructor_kwargs(cls, path: pathlib.Path) -> dict[str, Any]:
        return {
            "convergence": GMCalibrateResults._read_convergence(path),
            "band_share": GMCalibrateResults._read_band_share(path),
            "distribution": GMCalibrateResults._read_distribution(path),
            "residuals": GMCalibrateResults._read_residuals(path),
            "best_params": GMCalibrateResults._read_best_params(path),
        }

    @classmethod
    def from_file(
        cls,
        path: pathlib.Path,
        running_log_path: os.PathLike,
        cost_function: cost_functions.CostFunction,
    ) -> GMCreator:
        """Load data from files to create this test"""
        calib_path = path / cost_function.name.lower() / "calibrate"
        return cls(
            cost_function=cost_function,
            running_log_path=running_log_path,
            **cls.get_common_constructor_kwargs(path),
            **cls.get_specific_constructor_kwargs(calib_path),
        )

    def assert_results(self, gm_results: GravityModelResults) -> None:
        """Assert that all the results are as expected"""
        # Check the scalar values
        for key, val in self.best_params.items():
            assert key in gm_results.cost_params
            np.testing.assert_almost_equal(gm_results.cost_params[key], val, decimal=5)

        np.testing.assert_almost_equal(
            gm_results.cost_convergence,
            self.convergence,
            decimal=5,
        )
        # Check the matrices
        np.testing.assert_allclose(
            gm_results.cost_distribution.band_share_vals,
            self.band_share,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            self.target_cost_distribution.residuals(gm_results.cost_distribution),
            self.residuals,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            gm_results.value_distribution,
            self.distribution,
            rtol=1e-3,
        )


@dataclasses.dataclass
class GMCalibratePerceivedResults(GMCalibrateResults):
    """Stores the expected results alongside the inputs for calibration"""

    @classmethod
    def from_file(
        cls,
        path: pathlib.Path,
        running_log_path: os.PathLike,
        cost_function: cost_functions.CostFunction,
    ) -> GMCreator:
        """Load data from files to create this test"""
        calib_path = path / cost_function.name.lower() / "calibrate_perceived"
        return cls(
            cost_function=cost_function,
            running_log_path=running_log_path,
            **cls.get_common_constructor_kwargs(path),
            **cls.get_specific_constructor_kwargs(calib_path),
        )


@dataclasses.dataclass
class GMRunResults(GMCalibrateResults):
    """Stores the expected results alongside the inputs for run"""

    @classmethod
    def from_file(
        cls,
        path: pathlib.Path,
        running_log_path: os.PathLike,
        cost_function: cost_functions.CostFunction,
    ) -> GMCreator:
        """Load data from files to create this test"""
        calib_path = path / cost_function.name.lower() / "run"
        return cls(
            cost_function=cost_function,
            running_log_path=running_log_path,
            **cls.get_common_constructor_kwargs(path),
            **cls.get_specific_constructor_kwargs(calib_path),
        )

    def get_optimal_params(self) -> dict[str, Any]:
        """Get the optimal parameters from disk"""
        return self.best_params


@dataclasses.dataclass
class GMRunPerceivedResults(GMRunResults):
    """Stores the expected results alongside the inputs for calibration"""

    @classmethod
    def from_file(
        cls,
        path: pathlib.Path,
        running_log_path: os.PathLike,
        cost_function: cost_functions.CostFunction,
    ) -> GMCreator:
        """Load data from files to create this test"""
        calib_path = path / cost_function.name.lower() / "run_perceived"
        return cls(
            cost_function=cost_function,
            running_log_path=running_log_path,
            **cls.get_common_constructor_kwargs(path),
            **cls.get_specific_constructor_kwargs(calib_path),
        )


# # # FIXTURES # # #
def simple_gm_calib_results(tmp_path, cost_function) -> GMCalibrateResults:
    """Load in the small_and_simple test"""
    running_log_path = tmp_path / "run_log.csv"
    data_path = TEST_DATA_PATH / "small_and_simple"
    return GMCalibrateResults.from_file(
        path=data_path,
        running_log_path=running_log_path,
        cost_function=cost_function,
    )


def real_gm_calib_results(tmp_path, cost_function) -> GMCalibrateResults:
    """Load in the real world test"""
    running_log_path = tmp_path / "run_log.csv"
    data_path = TEST_DATA_PATH / "realistic"
    return GMCalibrateResults.from_file(
        path=data_path,
        running_log_path=running_log_path,
        cost_function=cost_function,
    )


def real_gm_calib_perceived_results(tmp_path, cost_function) -> GMCalibrateResults:
    """Load in the real world test"""
    running_log_path = tmp_path / "run_log.csv"
    data_path = TEST_DATA_PATH / "realistic"
    return GMCalibratePerceivedResults.from_file(
        path=data_path,
        running_log_path=running_log_path,
        cost_function=cost_function,
    )


def simple_gm_run_results(tmp_path, cost_function) -> GMRunResults:
    """Load in the small_and_simple test"""
    running_log_path = tmp_path / "run_log.csv"
    data_path = TEST_DATA_PATH / "small_and_simple"
    return GMRunResults.from_file(
        path=data_path,
        running_log_path=running_log_path,
        cost_function=cost_function,
    )


def real_gm_run_results(tmp_path, cost_function) -> GMRunResults:
    """Load in the real world test"""
    running_log_path = tmp_path / "run_log.csv"
    data_path = TEST_DATA_PATH / "realistic"
    return GMRunResults.from_file(
        path=data_path,
        running_log_path=running_log_path,
        cost_function=cost_function,
    )


def real_gm_run_perceived_results(tmp_path, cost_function) -> GMRunResults:
    """Load in the real world test"""
    running_log_path = tmp_path / "run_log.csv"
    data_path = TEST_DATA_PATH / "realistic"
    return GMRunPerceivedResults.from_file(
        path=data_path,
        running_log_path=running_log_path,
        cost_function=cost_function,
    )


@pytest.fixture(name="simple_log_normal_calib")
def fixture_simple_log_normal_calib(tmp_path) -> GMCalibrateResults:
    """Load in the small_and_simple log normal test"""
    path = tmp_path / "simple_log_normal_calib"
    path.mkdir()
    return simple_gm_calib_results(
        tmp_path=path,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )


@pytest.fixture(name="simple_tanner_calib")
def fixture_simple_tanner_calib(tmp_path) -> GMCalibrateResults:
    """Load in the small_and_simple log normal test"""
    path = tmp_path / "simple_tanner_calib"
    path.mkdir()
    return simple_gm_calib_results(
        tmp_path=path,
        cost_function=cost_functions.BuiltInCostFunction.TANNER.get_cost_function(),
    )


@pytest.fixture(name="simple_log_normal_run")
def fixture_simple_log_normal_run(tmp_path) -> GMRunResults:
    """Load in the small_and_simple log normal test"""
    path = tmp_path / "simple_log_normal_run"
    path.mkdir()
    return simple_gm_run_results(
        tmp_path=path,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )


@pytest.fixture(name="simple_tanner_run")
def fixture_simple_tanner_run(tmp_path) -> GMRunResults:
    """Load in the small_and_simple log normal test"""
    path = tmp_path / "simple_tanner_run"
    path.mkdir()
    return simple_gm_run_results(
        tmp_path=path,
        cost_function=cost_functions.BuiltInCostFunction.TANNER.get_cost_function(),
    )


@pytest.fixture(name="real_log_normal_calib")
def fixture_real_log_normal_calib(tmp_path) -> GMCalibrateResults:
    """Load in the realistic log normal test"""
    path = tmp_path / "real_log_normal_calib"
    path.mkdir()
    return real_gm_calib_results(
        tmp_path=path,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )


@pytest.fixture(name="real_log_normal_calib_perceived")
def fixture_real_log_normal_calib_perceived(tmp_path) -> GMCalibrateResults:
    """Load in the realistic log normal test"""
    path = tmp_path / "real_log_normal_calib_perceived"
    path.mkdir()
    return real_gm_calib_perceived_results(
        tmp_path=path,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )


@pytest.fixture(name="real_log_normal_run")
def fixture_real_log_normal_run(tmp_path) -> GMRunResults:
    """Load in the realistic log normal test"""
    path = tmp_path / "real_log_normal_run"
    path.mkdir()
    return real_gm_run_results(
        tmp_path=path,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )


@pytest.fixture(name="real_log_normal_run_perceived")
def fixture_real_log_normal_run_perceived(tmp_path) -> GMRunResults:
    """Load in the realistic log normal test"""
    path = tmp_path / "real_log_normal_run_perceived"
    path.mkdir()
    return real_gm_run_perceived_results(
        tmp_path=path,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )


# # # TESTS # # #
@pytest.mark.usefixtures("simple_log_normal_calib", "simple_log_normal_run")
class TestSimpleLogNormal:
    """Tests the log normal calibrator with a simple example"""

    def test_correct_calibrate(self, simple_log_normal_calib: GMCalibrateResults):
        """Test that the gravity model correctly calibrates."""
        gm = simple_log_normal_calib.create_gravity_model()
        best_params = gm.calibrate()
        simple_log_normal_calib.assert_results(
            best_params=best_params,
            calibrated_gm=gm,
        )

    def test_correct_run(self, simple_log_normal_run: GMRunResults):
        """Test that the gravity model correctly runs."""
        gm = simple_log_normal_run.create_gravity_model()
        best_params = simple_log_normal_run.get_optimal_params()
        best_params = gm.calibrate(init_params=best_params, calibrate_params=False)
        simple_log_normal_run.assert_results(
            best_params=best_params,
            calibrated_gm=gm,
        )

    def test_correct_perceived(self):
        """Test that the gravity model correctly calibrates with perceived factors."""
        # Use cost function as a param

        # Make GM
        # Run
        # Assert
        pass


@pytest.mark.usefixtures("simple_tanner_calib", "simple_tanner_run")
class TestSimpleTanner:
    """Tests the tanner calibrator with a simple example"""

    def test_correct_calibrate(self, simple_tanner_calib: GMCalibrateResults):
        """Test that the gravity model correctly calibrates."""
        gm = simple_tanner_calib.create_gravity_model()
        best_params = gm.calibrate()
        simple_tanner_calib.assert_results(
            best_params=best_params,
            calibrated_gm=gm,
        )

    def test_correct_run(self, simple_tanner_run: GMRunResults):
        """Test that the gravity model correctly runs."""
        gm = simple_tanner_run.create_gravity_model()
        best_params = simple_tanner_run.get_optimal_params()
        best_params = gm.calibrate(init_params=best_params, calibrate_params=False)
        simple_tanner_run.assert_results(
            best_params=best_params,
            calibrated_gm=gm,
        )


@pytest.mark.usefixtures("simple_tanner_run", "simple_log_normal_run", "real_log_normal_run")
class TestRunMethods:
    """Thoroughly tests the run functions using a simple example."""

    @pytest.mark.parametrize(
        "fixture_str",
        ["simple_tanner_run", "simple_log_normal_run", "real_log_normal_run"],
    )
    def test_normal_run(self, fixture_str, request):
        """Test a default run."""
        run_and_results = request.getfixturevalue(fixture_str)
        best_params = run_and_results.get_optimal_params()
        gm_results = run_and_results.create_and_run_gravity_model(best_params)
        run_and_results.assert_results(
            gm_results=gm_results,
        )

    @pytest.mark.parametrize(
        "fixture_str",
        ["simple_tanner_run", "simple_log_normal_run", "real_log_normal_run_perceived"],
    )
    def test_perceived_run(self, fixture_str, request):
        """Test a perceived factor run."""
        run_and_results = request.getfixturevalue(fixture_str)
        best_params = run_and_results.get_optimal_params()
        gm_results = run_and_results.create_and_run_gravity_model(
            best_params, use_perceived_factors=True
        )
        run_and_results.assert_results(
            gm_results=gm_results,
        )


@pytest.mark.usefixtures(
    "real_log_normal_calib",
    "real_log_normal_calib_perceived",
    "real_log_normal_run",
)
class TestRealLogNormal:
    """Test the log normal calibrator with real world data."""

    def test_correct_calibrate(self, real_log_normal_calib: GMCalibrateResults):
        """Test that the gravity model correctly calibrates."""
        gm = real_log_normal_calib.create_gravity_model()
        best_params = gm.calibrate()
        real_log_normal_calib.assert_results(
            best_params=best_params,
            calibrated_gm=gm,
        )

    def test_correct_calibrate_perceived(
        self,
        real_log_normal_calib_perceived: GMCalibratePerceivedResults,
    ):
        """Test that the gravity model correctly calibrates."""
        msg = "Calibration with perceived factors was not able to reach the target_convergence"
        with pytest.warns(UserWarning, match=msg):
            gm = real_log_normal_calib_perceived.create_gravity_model(
                use_perceived_factors=True
            )
            best_params = gm.calibrate()
            real_log_normal_calib_perceived.assert_results(
                best_params=best_params,
                calibrated_gm=gm,
            )

    def test_correct_run(self, real_log_normal_run: GMRunResults):
        """Test that the gravity model correctly runs."""
        gm = real_log_normal_run.create_gravity_model()
        best_params = real_log_normal_run.get_optimal_params()
        best_params = gm.calibrate(init_params=best_params, calibrate_params=False)
        real_log_normal_run.assert_results(
            best_params=best_params,
            calibrated_gm=gm,
        )
