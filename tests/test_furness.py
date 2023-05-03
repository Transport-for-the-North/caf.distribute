# -*- coding: utf-8 -*-
"""Tests for the {} module"""
# Built-Ins
import dataclasses

from typing import Any

# Third Party
import pytest
import numpy as np


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.distribute import furness

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # CLASSES # # #
@dataclasses.dataclass
class DoubleFurnessResults:
    """Collection of I/O data for a doubly constrained furness"""

    # Input
    seed_vals: np.ndarray
    row_targets: np.ndarray
    col_targets: np.ndarray

    # Results
    furness_mat: np.ndarray
    iter_num: int
    rmse: float

    def input_kwargs(
        self,
        tol: float = 1e-9,
        max_iters: int = 5000,
        warning: bool = True,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "seed_vals": self.seed_vals,
            "row_targets": self.row_targets,
            "col_targets": self.col_targets,
            "tol": tol,
            "max_iters": max_iters,
            "warning": warning,
        }

    def check_results(
        self,
        furness_mat: np.ndarray,
        iter_num: int,
        rmse: float,
        ignore_rmse: bool = False,
    ):
        """Assert the returned results"""
        np.testing.assert_almost_equal(furness_mat, self.furness_mat)
        np.testing.assert_equal(iter_num, self.iter_num)
        if not ignore_rmse:
            np.testing.assert_almost_equal(rmse, self.rmse)


# # # FIXTURES # # #
@pytest.fixture(name="no_furness", scope="class")
def fixture_no_furness():
    """Create inputs that don't need furnessing"""
    seed_vals = np.array([[46, 49, 19], [42, 36, 26], [23, 58, 24]]).astype(float)
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=seed_vals.sum(axis=1),
        col_targets=seed_vals.sum(axis=0),
        furness_mat=seed_vals,
        iter_num=1,
        rmse=0,
    )


@pytest.fixture(name="no_furness_int", scope="class")
def fixture_no_furness_int():
    """Create integer inputs that don't need furnessing"""
    seed_vals = np.array([[46, 49, 19], [42, 36, 26], [23, 58, 24]])
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=seed_vals.sum(axis=1),
        col_targets=seed_vals.sum(axis=0),
        furness_mat=seed_vals,
        iter_num=1,
        rmse=0,
    )


@pytest.fixture(name="zero_target_furness", scope="class")
def fixture_zero_target_furness():
    """Create integer inputs that don't need furnessing"""
    seed_vals = np.arange(9).reshape((3, 3))
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=np.array([0, 0, 0]),
        col_targets=np.array([0, 0, 0]),
        furness_mat=np.zeros_like(seed_vals),
        iter_num=0,
        rmse=np.inf,
    )


@pytest.fixture(name="simple_furness", scope="class")
def fixture_simple_furness():
    """Create inputs that need furnessing

    Example taken from:
    https://u.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFTWOD.pdf
    """
    seed_vals = np.array([[1, 2, 1], [3, 5, 5], [6, 2, 2]])
    result = np.array(
        [
            [1.5129424, 2.3095188, 1.1775388],
            [4.2025351, 5.3460034, 5.4514615],
            [5.2845225, 1.3444778, 1.3709997],
        ]
    )
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=np.array([5, 15, 8]),
        col_targets=np.array([11, 9, 8]),
        furness_mat=result,
        iter_num=11,
        rmse=0,
    )


@pytest.fixture(name="difficult_furness", scope="class")
def fixture_difficult_furness():
    """Create inputs that need furnessing"""
    seed_vals = np.array(
        [
            [26.4144054, 23.59083712, 11.84785436],
            [16.99996583, 84.35729962, 62.02252805],
            [94.97465589, 67.97859293, 71.59835939],
        ]
    )
    result = np.array(
        [
            [21.7052596, 0.9060783, 2.3886621],
            [34.3193291, 7.9599735, 30.7206974],
            [6.5658874, 0.2196626, 1.21445],
        ]
    )
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=np.array([25, 73, 8]),
        col_targets=np.array([62, 9, 34]),
        furness_mat=result,
        iter_num=5000,
        rmse=0.3919445870717492,
    )


@pytest.fixture(name="impossible_furness", scope="class")
def fixture_impossible_furness():
    """Create inputs that cannot succeed furnessing"""
    seed_vals = np.zeros((3, 3))
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=np.array([25, 73, 8]),
        col_targets=np.array([62, 9, 34]),
        furness_mat=seed_vals,
        iter_num=5000,
        rmse=0.3919445870717492,
    )


@pytest.fixture(name="nan_furness", scope="class")
def fixture_nan_furness():
    """Create inputs that need furnessing"""
    seed_vals = np.array([[1, 2, 1], [3, np.nan, 5], [6, 2, 2]])
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=np.array([5, 15, 8]),
        col_targets=np.array([11, 9, 8]),
        furness_mat=np.zeros_like(seed_vals),
        iter_num=0,
        rmse=np.inf,
    )


@pytest.fixture(name="nan_target", scope="class")
def fixture_nan_target():
    """Create inputs that need furnessing"""
    seed_vals = np.array([[1, 2, 1], [3, 5, 5], [6, 2, 2]])
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=np.array([5, np.nan, 8]),
        col_targets=np.array([11, 9, 8]),
        furness_mat=np.zeros_like(seed_vals),
        iter_num=0,
        rmse=np.inf,
    )


# # # TESTS # # #
@pytest.mark.usefixtures(
    "no_furness",
    "no_furness_int",
    "simple_furness",
    "difficult_furness",
    "impossible_furness",
    "zero_target_furness",
    "nan_furness",
    "nan_target",
)
class TestDoublyConstrainedFurness:
    """Tests for the doubly_constrained_furness function"""

    @pytest.mark.parametrize(
        "fixture_str",
        ["no_furness", "no_furness_int", "simple_furness"],
    )
    def test_correct(self, fixture_str: str, request):
        """Check the correct results are achieved"""
        furness_results = request.getfixturevalue(fixture_str)
        results = furness.doubly_constrained_furness(**furness_results.input_kwargs())
        furness_results.check_results(*results)

    @pytest.mark.parametrize(
        "fixture_str",
        ["impossible_furness", "difficult_furness"],
    )
    @pytest.mark.parametrize("warn", [True, False])
    def test_difficult_warning(self, fixture_str: str, warn: bool, request):
        """Check that a warning is raised when hard to converge"""
        furness_results = request.getfixturevalue(fixture_str)
        if warn:
            msg = "furness exhausted its max number of loops"
            with pytest.warns(UserWarning, match=msg):
                results = furness.doubly_constrained_furness(**furness_results.input_kwargs())
        else:
            results = furness.doubly_constrained_furness(**furness_results.input_kwargs())
        furness_results.check_results(*results, ignore_rmse=True)

    def test_zero_target_warning(self, zero_target_furness: DoubleFurnessResults):
        """Check that a warning is raised when seed is only 0s"""
        msg = "Furness given targets of 0"
        with pytest.warns(UserWarning, match=msg):
            results = furness.doubly_constrained_furness(**zero_target_furness.input_kwargs())
        zero_target_furness.check_results(*results)

    def test_nan_warning(self, nan_furness: DoubleFurnessResults):
        """Check that a warning is raised when nan is found"""
        msg = "np.nan value found in the rmse calculation"
        with pytest.warns(UserWarning, match=msg):
            results = furness.doubly_constrained_furness(**nan_furness.input_kwargs())
        nan_furness.check_results(*results)

    def test_nan_target(self, nan_target: DoubleFurnessResults):
        """Check that a warning is raised when nan is found"""
        msg = "np.nan found in the targets"
        with pytest.raises(ValueError, match=msg):
            furness.doubly_constrained_furness(**nan_target.input_kwargs())
