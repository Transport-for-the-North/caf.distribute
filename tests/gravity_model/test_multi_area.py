import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from caf.toolkit import cost_utils
from caf.distribute import gravity_model as gm, cost_functions
from caf.distribute import utils


@pytest.fixture(name="data_dir", scope="session")
def fixture_data_dir():
    return Path(
        r"C:\Users\IsaacScott\Documents\Github\caf.distribute\tests\gravity_model\data\multi_area_unit"
    )

@pytest.fixture(name="mock_dir", scope="session")
def fixture_mock_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("main")
    return path


@pytest.fixture(name="trip_ends", scope="session")
def fixture_tripends(data_dir):
    trip_ends = pd.read_csv(
        data_dir / "trip_ends.csv", names=["zone", "origin", "destination"]
    )
    return trip_ends.set_index("zone")


@pytest.fixture(name="costs", scope="session")
def fixture_costs(data_dir):
    costs = pd.read_csv(data_dir / "costs.csv", names=["origin", "destination", "cost"])
    return costs.set_index(["origin", "destination"])


@pytest.fixture(name="distributions", scope="session")
def fixture_distributions(data_dir):
    dists = pd.read_csv(data_dir / "distributions.csv")
    return dists


@pytest.fixture(name="dists_lookup", scope="session")
def fixture_lookup(data_dir):
    lookup = pd.read_csv(data_dir / "distributions_lookup.csv", index_col=0)
    return lookup


@pytest.fixture(name="infilled", scope="session")
def fixture_infill(costs):
    wide_costs = costs.unstack().fillna(np.inf) / 1000
    infilled = utils.infill_cost_matrix(wide_costs.values)
    return infilled


@pytest.fixture(name="expected_infilled", scope="session")
def fixture_infilled(data_dir):
    costs_df = pd.read_csv(data_dir / "costs_infilled.csv", index_col=0)
    return costs_df.values


@pytest.fixture(name="distributions_sorted", scope="session")
def fixture_dists(data_dir, distributions):
    tld_df = distributions
    lookup = pd.read_csv(data_dir / "distributions_lookup.csv")
    sorted = utils.process_tlds(
        tld_df[tld_df["cat"] != "Minor"],
        "cat",
        "lower",
        "upper",
        "avg",
        "trips",
        lookup,
        "cat",
        "zone",
        {"mu": 1, "sigma": 2},
    )
    return sorted


@pytest.fixture(name="cal_results", scope="session")
def fixture_cal_res(data_dir, infilled, distributions_sorted, trip_ends, mock_dir):
    row_targets = trip_ends["origin"].values
    col_targets = trip_ends["destination"].values
    model = gm.MultiAreaGravityModelCalibrator(
        row_targets=row_targets,
        col_targets=col_targets,
        cost_matrix=infilled,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )
    results = model.calibrate(
        running_log_path=mock_dir / "temp_log.csv", init_params=distributions_sorted
    )
    return results


class TestUtils:
    def test_infill_costs(self, costs, infilled, expected_infilled):
        assert np.array_equal(np.round(expected_infilled, 3), np.round(infilled, 3))

    def test_tld_processing_length(self, distributions_sorted):
        assert len(distributions_sorted) == 4

    def test_tld_processing_type(self, distributions_sorted):
        assert isinstance(distributions_sorted[0], gm.MultiCostDistribution)


class TestDist:
    @pytest.mark.parametrize("area", ['City', 'Town', 'External', 'Village'])
    def test_convergence(self, cal_results, area):
        dist = cal_results[area]
        assert dist.cost_convergence > 0.85

    @pytest.mark.parametrize("area", ['City', 'Town', 'External', 'Village'])
    def test_params(self, cal_results, area):
        dist = cal_results[area]
        mu = dist.cost_params['mu']
        sigma = dist.cost_params['sigma']
        check = (sigma > 0) & (sigma < 3) & (mu > 0) & (mu < 3)
        assert check


