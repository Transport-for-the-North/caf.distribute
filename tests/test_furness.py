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
    ):
        """Assert the returned results"""
        np.testing.assert_almost_equal(furness_mat, self.furness_mat)
        np.testing.assert_equal(iter_num, self.iter_num)
        np.testing.assert_almost_equal(rmse, self.rmse)


# # # FIXTURES # # #
@pytest.fixture(name="no_furness", scope="class")
def fixture_no_furness():
    """Create a furness that needs to furnessing"""
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
    """Create a furness that needs to furnessing"""
    seed_vals = np.array([[46, 49, 19], [42, 36, 26], [23, 58, 24]])
    return DoubleFurnessResults(
        seed_vals=seed_vals,
        row_targets=seed_vals.sum(axis=1),
        col_targets=seed_vals.sum(axis=0),
        furness_mat=seed_vals,
        iter_num=1,
        rmse=0,
    )


# # # TESTS # # #
@pytest.mark.usefixtures("no_furness", "no_furness_int")
class TestDoublyConstrainedFurness:
    """Tests for the doubly_constrained_furness function"""

    @pytest.mark.parametrize(
        "fixture_str",
        ["no_furness", "no_furness_int"],
    )
    def test_correct(self, fixture_str: str, request):
        """Check the correct results are achieved"""
        furness_results = request.getfixturevalue(fixture_str)
        results = furness.doubly_constrained_furness(**furness_results.input_kwargs())
        furness_results.check_results(*results)
