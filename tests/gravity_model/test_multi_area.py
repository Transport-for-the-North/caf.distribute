import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from caf.toolkit import cost_utils
from caf.distribute.gravity_model import MultiCostDistribution
from caf.distribute import utils

@pytest.fixture(name='data_dir', scope='session')
def fixture_data_dir():
    return Path(r"C:\Users\IsaacScott\Documents\Github\caf.distribute\tests\gravity_model\data\multi_area_unit")

@pytest.fixture(name="trip_ends", scope="session")
def fixture_tripends(data_dir):
    trip_ends = pd.read_csv(data_dir / "trip_ends.csv",
                            names=['zone', 'origin', 'destination'])
    return trip_ends.set_index('zone')

@pytest.fixture(name="costs", scope="session")
def fixture_costs(data_dir):
    costs = pd.read_csv(data_dir / "costs.csv",
                            names=['origin', 'destination', 'cost'])
    return costs.set_index(['origin', 'destination'])

@pytest.fixture(name="distributions", scope="session")
def fixture_distributions(data_dir):
    dists = pd.read_csv(data_dir / "distributions.csv", index_col=0)
    return dists

@pytest.fixture(name="dists_lookup", scope="session")
def fixture_lookup(data_dir):
    lookup = pd.read_csv(data_dir / "distributions_lookup.csv", index_col=0)
    return lookup

@pytest.fixture(name="infilled_costs", scope="session")
def fixture_infilled(data_dir):
    costs_df = pd.read_csv(data_dir / "costs_infilled.csv", index_col=0)
    return costs_df.values

class TestUtils:

    def test_infill_costs(self, costs, infilled_costs):
        wide_costs = costs.unstack().fillna(np.inf)
        infilled = utils.infill_cost_matrix(wide_costs.values)
        assert np.array_equal(wide_costs, infilled)

    def test_tld_processing(self):
        return None




