# -*- coding: utf-8 -*-
"""Implementation of a self-calibrating single area gravity model."""
# Built-Ins
import logging

from typing import Iterator

# Third Party
import numpy as np
from scipy import optimize
from caf.distribute.gravity_model.core import GravityModelCalibrateResults

# Local Imports
from caf.toolkit import cost_utils
from caf.distribute.gravity_model import core


# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
class MultiCostDistributions:
    # TODO(MB) Implement functionality
    # Class for storing multiple cost distributions
    # with indices denoting area of the matrix they
    # apply to

    id_matrix: np.array
    id_to_tld: dict[int, cost_utils.CostDistribution]

    def __iter__(self) -> Iterator[tuple[np.ndarray, cost_utils.CostDistribution]]:
        # Iterate through tlds and retun masks for area
        raise NotImplementedError("WIP")


class MultiAreaGravityModelCalibrator(core.GravityModelBase):
    # TODO(MB) Implement functionality for the abstract methods

    ### Inputs
    # actual cost matrix
    # adjusted cost matrix (optional) instead could be callable

    # Need to update functions in base class to allow for giving multiple cost distributions
    # Ideally calibrate and calibrate_with_percieved factors should be defined flexibly
    # in base class

    def _calculate_perceived_factors(self) -> None:
        raise NotImplementedError("WIP")

    def _calibrate(self) -> GravityModelCalibrateResults:
        # TODO(MB) Some small amount of refactoring in core but
        # hopefully can use alot of the same code
        raise NotImplementedError("WIP")

    def _gravity_function(self):
        raise NotImplementedError("WIP")

    def _jacobian_function(self):
        raise NotImplementedError("WIP")


# # # FUNCTIONS # # #
