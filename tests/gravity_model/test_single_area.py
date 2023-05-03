# -*- coding: utf-8 -*-
"""Tests for the gravity_model.single_area module"""
from __future__ import annotations

# Built-Ins
import os
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
class GravityModelCalibResults:
    row_targets: np.ndarray
    col_targets: np.ndarray
    cost_function: cost_functions.CostFunction
    cost_matrix: np.ndarray
    target_cost_distribution: pd.DataFrame
    running_log_path: os.PathLike

    @staticmethod
    def from_file(path: pathlib.Path, running_log_path: os.PathLike) -> GravityModelCalibResults:
        """Load data from files to create this test"""
        return GravityModelCalibResults(
            row_targets=np.loadtxt(path / "row_targets.csv", delimiter=","),
            col_targets=np.loadtxt(path / "col_targets.csv", delimiter=","),
            cost_matrix=np.loadtxt(path / "cost_matrix.csv", delimiter=","),
            target_cost_distribution=pd.read_csv(path / "target_cost_distribution.csv"),
            cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
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

    def check_results(self):
        # Only if more than one to check?
        pass


# # # FIXTURES # # #
@pytest.fixture(name="simple_gm_results")
def fixture_simple_gm_results(tmp_path) -> GravityModelCalibResults:
    """Load in the small_and_simple test"""
    running_log_path = tmp_path / "run_log.log"
    running_log_path.touch()
    data_path = TEST_DATA_PATH / "small_and_simple"
    return GravityModelCalibResults.from_file(data_path, running_log_path)


# # # TESTS # # #
@pytest.mark.usefixtures(
    "simple_gm_results"
)
class TestSingleAreaGravityModelCalibrator:
    """Tests the single area gravity model class"""

    def test_correct_calibrate(self, simple_gm_results: GravityModelCalibResults):
        """Test that the gravity model correctly calibrates."""
        # Use cost function as a param

        gm = simple_gm_results.create_gravity_model()
        res = gm.calibrate()
        print(res)

        print(gm.achieved_convergence)
        print(gm.achieved_band_share)       # Rename to cost distrubution
        print(simple_gm_results.target_cost_distribution)
        print(gm.achieved_residuals)
        print(gm.achieved_distribution)

        # Make GM
        # Run
        # Assert
        pass

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
