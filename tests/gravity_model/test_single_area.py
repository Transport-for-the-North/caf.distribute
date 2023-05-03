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
import pandas as pd


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.distribute import cost_functions
from caf.distribute.gravity_model import SingleAreaGravityModelCalibrator

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
TEST_DATA_PATH = pathlib.Path("data")


# # # Classes # # #
@dataclasses.dataclass
class GMCreator:
    """Can create a gravity mode class from disk"""

    row_targets: np.ndarray
    col_targets: np.ndarray
    cost_function: cost_functions.CostFunction
    cost_matrix: np.ndarray
    target_cost_distribution: pd.DataFrame
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
    def _read_cost_distribution(home: pathlib.Path) -> np.ndarray:
        return pd.read_csv(home / "target_cost_distribution.csv")

    @staticmethod
    def from_file(
        path: pathlib.Path,
        running_log_path: os.PathLike,
        cost_function: cost_functions.CostFunction,
    ) -> GMCreator:
        """Load data from files to create this test"""
        return GMCreator(
            row_targets=GMCreator._read_row_targets(path),
            col_targets=GMCreator._read_col_targets(path),
            cost_matrix=GMCreator._read_cost_matrix(path),
            target_cost_distribution=GMCreator._read_cost_distribution(path),
            cost_function=cost_function,
            running_log_path=running_log_path,
        )

    def create_gravity_model(
        self,
        target_convergence: float = 0.9,
        furness_max_iters: int = 1000,
        furness_tol: float = 1e-3,
        use_perceived_factors: bool = False,
    ) -> SingleAreaGravityModelCalibrator:
        return SingleAreaGravityModelCalibrator(
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            cost_function=self.cost_function,
            cost_matrix=self.cost_matrix,
            target_cost_distribution=self.target_cost_distribution,
            running_log_path=self.running_log_path,
            target_convergence=target_convergence,
            furness_max_iters=furness_max_iters,
            furness_tol=furness_tol,
            use_perceived_factors=use_perceived_factors,
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

    @staticmethod
    def from_file(
        path: pathlib.Path,
        running_log_path: os.PathLike,
        cost_function: cost_functions.CostFunction,
    ) -> GMCreator:
        """Load data from files to create this test"""
        calib_path = path / cost_function.name / "calibrate"
        return GMCalibrateResults(
            row_targets=GMCalibrateResults._read_row_targets(path),
            col_targets=GMCalibrateResults._read_col_targets(path),
            cost_matrix=GMCalibrateResults._read_cost_matrix(path),
            target_cost_distribution=GMCalibrateResults._read_cost_distribution(path),
            cost_function=cost_function,
            running_log_path=running_log_path,
            convergence=GMCalibrateResults._read_convergence(calib_path),
            band_share=GMCalibrateResults._read_band_share(calib_path),
            distribution=GMCalibrateResults._read_distribution(calib_path),
            residuals=GMCalibrateResults._read_residuals(calib_path),
            best_params=GMCalibrateResults._read_best_params(calib_path),
        )

    def assert_results(
        self,
        best_params: dict[str, Any],
        calibrated_gm: SingleAreaGravityModelCalibrator,
    ) -> None:
        """Assert that all the results are as expected"""
        # Check the scalar values
        for key, val in self.best_params.items():
            assert key in best_params
            np.testing.assert_almost_equal(best_params[key], val, decimal=5)

        np.testing.assert_almost_equal(
            calibrated_gm.achieved_convergence, self.convergence, decimal=5
        )

        # Check the matrices
        np.testing.assert_allclose(calibrated_gm.achieved_band_share, self.band_share, rtol=1e-4)
        np.testing.assert_allclose(calibrated_gm.achieved_residuals, self.residuals, rtol=1e-4)
        np.testing.assert_allclose(calibrated_gm.achieved_distribution, self.distribution, rtol=1e-4)


# # # FIXTURES # # #
@pytest.fixture(name="simple_gm_results")
def fixture_simple_gm_results(tmp_path, request) -> GMCalibrateResults:
    """Load in the small_and_simple test"""
    running_log_path = tmp_path / "run_log.log"
    running_log_path.touch()
    data_path = TEST_DATA_PATH / "small_and_simple"
    return GMCalibrateResults.from_file(
        path=data_path,
        running_log_path=running_log_path,
        cost_function=request.param,
    )


# # # TESTS # # #
@pytest.mark.usefixtures("simple_gm_results")
class TestSingleAreaGravityModelCalibrator:
    """Tests the single area gravity model class"""

    @pytest.mark.parametrize(
        "simple_gm_results",
        [
            cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
            cost_functions.BuiltInCostFunction.TANNER.get_cost_function(),
        ],
        indirect=True,
    )
    def test_correct_calibrate(self, simple_gm_results: GMCalibrateResults):
        """Test that the gravity model correctly calibrates."""
        gm = simple_gm_results.create_gravity_model()
        best_params = gm.calibrate()
        simple_gm_results.assert_results(
            best_params=best_params,
            calibrated_gm=gm,
        )

    def test_correct_run(self):
        """Test that the gravity model correctly runs."""
        # Use cost function as a param

        # Make GM
        # Run
        # Assert
        pass

    def test_correct_perceived(self):
        """Test that the gravity model correctly calibrates with perceived factors."""
        # Use cost function as a param

        # Make GM
        # Run
        # Assert
        pass
