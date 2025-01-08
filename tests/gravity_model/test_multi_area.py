from __future__ import annotations

# Built-Ins
from pathlib import Path

# Third Party
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest
from caf.toolkit import cost_utils

# Local Imports
from caf.distribute import cost_functions
from caf.distribute import gravity_model as gm
from caf.distribute import utils
from caf.distribute.gravity_model import GravityModelResults


@pytest.fixture(name="cost_from_code", scope="session")
def fixture_code_costs():
    np.random.seed(42)
    data = np.random.randint(1, 101, size=(10, 10)).astype(float)
    np.fill_diagonal(data, 0)
    df = pd.DataFrame(
        data, columns=[i + 1 for i in range(10)], index=[i + 1 for i in range(10)]
    )
    return df


@pytest.fixture(name="infilled_expected", scope="session")
def fix_infilled_exp(cost_from_code):
    mat = cost_from_code.values
    np.fill_diagonal(mat, 101)
    min = np.min(mat, axis=1)
    np.fill_diagonal(mat, min / 2)
    return mat


@pytest.fixture(name="data_dir", scope="session")
def fixture_data_dir():
    return Path(__file__).parent.resolve() / "data" / "multi_area_unit"


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


@pytest.fixture(name="infilled_from_code", scope="session")
def fixture_infilled_from_code(cost_from_code):
    return utils.infill_cost_matrix(cost_from_code.values)


@pytest.fixture(name="expected_infilled", scope="session")
def fixture_infilled(data_dir):
    costs_df = pd.read_csv(data_dir / "costs_infilled.csv", index_col=0)
    return costs_df.values


@pytest.fixture(name="distributions_sorted", scope="session")
def fixture_dists(data_dir, distributions):
    tld_df = distributions
    lookup = pd.read_csv(data_dir / "distributions_lookup.csv")
    sorted_ = utils.process_tlds(
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
    return sorted_


@pytest.fixture(name="no_furness_jac_conf", scope="session")
def fixture_conf(data_dir, mock_dir):
    conf = gm.MultiDistInput(
        tld_file=data_dir / "distributions.csv",
        tld_lookup_file=data_dir / "distributions_lookup.csv",
        cat_col="cat",
        min_col="lower",
        max_col="upper",
        ave_col="avg",
        trips_col="trips",
        lookup_cat_col="cat",
        lookup_zone_col="zone",
        init_params={"mu": 1, "sigma": 2},
        log_path=mock_dir / "log.csv",
        furness_tolerance=0.1,
        furness_jac=False,
    )
    return conf


@pytest.fixture(name="furness_jac_conf", scope="session")
def fixture_jac_furn(data_dir, mock_dir):
    conf = gm.MultiDistInput(
        tld_file=data_dir / "distributions.csv",
        tld_lookup_file=data_dir / "distributions_lookup.csv",
        cat_col="cat",
        min_col="lower",
        max_col="upper",
        ave_col="avg",
        trips_col="trips",
        lookup_cat_col="cat",
        lookup_zone_col="zone",
        init_params={"mu": 1, "sigma": 2},
        log_path=mock_dir / "log.csv",
        furness_tolerance=0.1,
        furness_jac=True,
    )
    return conf


@pytest.fixture(name="multi_tld", scope="session")
def _multi_tld(data_dir, mock_dir):
    tld_lookup = pd.read_csv(data_dir / "distributions_lookup.csv")

    ordered_zones = tld_lookup["zone"]

    func_parameters = {}
    for cat in tld_lookup["cat"].unique():
        func_parameters[cat] = {"mu": 1, "sigma": 2}

    multitld = gm.MultiCostDistribution.from_pandas(
        ordered_zones,
        pd.read_csv(data_dir / "distributions.csv"),
        tld_lookup,
        func_parameters,
        tld_cat_col="cat",
        tld_min_col="lower",
        tld_max_col="upper",
        tld_avg_col="avg",
        tld_trips_col="trips",
        lookup_cat_col="cat",
        lookup_zone_col="zone",
    )
    return multitld


@pytest.fixture(name="cal_no_furness", scope="session")
def fixture_cal_no_furness(infilled, multi_tld, trip_ends, mock_dir):
    row_targets = trip_ends["origin"].values
    col_targets = trip_ends["destination"].values
    model = gm.MultiAreaGravityModelCalibrator(
        row_targets=row_targets,
        col_targets=col_targets,
        cost_matrix=infilled,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )
    results = model.calibrate(
        multi_tld,
        running_log_path=mock_dir / "temp_log.csv",
        gm_params=gm.GMCalibParams(furness_jac=False),
    )
    return results


@pytest.fixture(name="cal_furness", scope="session")
def fixture_cal_furness(infilled, multi_tld, trip_ends, mock_dir):
    row_targets = trip_ends["origin"].values
    col_targets = trip_ends["destination"].values
    model = gm.MultiAreaGravityModelCalibrator(
        row_targets=row_targets,
        col_targets=col_targets,
        cost_matrix=infilled,
        cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
    )
    results = model.calibrate(
        multi_tld,
        running_log_path=mock_dir / "temp_log.csv",
        gm_params=gm.GMCalibParams(furness_jac=True),
    )
    return results


class TestUtils:
    # TODO(IS) only one test currently so leaving in this file
    def test_infill_costs(self, infilled_from_code, infilled_expected):
        """Test the method to infill a cost matrix (mainly intrazonal costs)"""
        assert np.array_equal(np.round(infilled_expected, 3), np.round(infilled_from_code, 3))


class TestDist:
    @pytest.mark.parametrize("area", ["City", "Town", "External", "Village"])
    @pytest.mark.parametrize("cal_results", ["cal_furness", "cal_no_furness"])
    @pytest.mark.filterwarnings("ignore:Given a log path:UserWarning")
    def test_convergence(self, cal_results, area, request):
        cal_results = request.getfixturevalue(cal_results)
        dist = cal_results[area]
        assert dist.cost_convergence > 0.85

    @pytest.mark.parametrize("area", ["City", "Town", "External", "Village"])
    @pytest.mark.parametrize("cal_results", ["cal_furness", "cal_no_furness"])
    @pytest.mark.filterwarnings("ignore:Given a log path:UserWarning")
    def test_params(self, cal_results, area, request):
        cal_results = request.getfixturevalue(cal_results)
        dist = cal_results[area]
        mu = dist.cost_params["mu"]
        sigma = dist.cost_params["sigma"]
        assert 0 < sigma < 3
        assert 0 < mu < 3


class TestResults:
    @pytest.mark.parametrize("cal_results", ["cal_furness", "cal_no_furness"])
    def test_results(self, cal_results, request):
        """Test the the GravityModelResults object methods run as expected"""
        cal_results = request.getfixturevalue(cal_results)
        assert isinstance(cal_results, dict)
        for result in cal_results.values():
            assert isinstance(result, GravityModelResults)
            assert isinstance(result.summary(), pd.Series)
            assert isinstance(result.plot_distributions(), plt.Figure)
