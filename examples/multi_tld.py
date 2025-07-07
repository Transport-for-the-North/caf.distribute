import pandas as pd
import caf.base as cb
from caf.distribute import cost_functions, gravity_model
from caf.distribute.gravity_model import multi_area

normits = cb.ZoningSystem.get_zoning('normits')
productions = cb.DVector.load(r"E:\tem\outputs\full_test\Core\hb_productions\hb_normits_tem_segmented_2023_dvec.h5").aggregate_comp_zones(normits).aggregate(['p','m','tp'])
attractions = cb.DVector.load(r"E:\tem\outputs\full_test\Core\hb_attractions\hb_normits_tem_segmented_2023_dvec.h5").aggregate_comp_zones(normits).aggregate(['p','m','tp'])
costs = pd.read_csv(r"E:\costs\CSVs\hwnet_cost_am-distance.csv", index_col=0)
zones_lookup = pd.read_csv(r"E:\costs\NorMITs_zone.csv")


cost_matrix_validated = costs.to_numpy()


# Define the cost function and parameters
cost_function = cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function()


for p in productions.segmentation.get_segment('p').values:
    tld = pd.read_csv(rf"I:\NTS\outputs\tld\NTS_tld_m3_p{p}_hb_fr.csv")
    tld['from'] = tld['trav_dist'].shift().fillna(0)
    tld.loc[tld['from'] > tld['trav_dist'], 'from'] = 0
    func_params = {i: cost_function.default_params for i in zones_lookup['tld_area'].unique()}
    cost_distributions = gravity_model.MultiCostDistribution.from_pandas(
        pd.Series(normits.zone_ids),
        tld,
        zones_lookup,
        func_params,
        tld_cat_col=";index",
        tld_min_col="from",
        tld_max_col="trav_dist",
        tld_avg_col="ave_dist",
        tld_trips_col="trips",
        lookup_cat_col="tld_area",
        lookup_zone_col=";normits_v3.3_id",
    )

    calib_gm = gravity_model.MultiAreaGravityModelCalibrator(
        productions.data.loc[p,3,1].to_numpy(),
        attractions.data.loc[p,3,1].to_numpy(),
        cost_matrix_validated,
        cost_function,
    )

    gravity_model_results = calib_gm.calibrate(
        cost_distributions,
        r"E:\costs\log.log",
        multi_area.GMCalibParams(furness_jac=True),
        verbose=2,
    )

    matrix = pd.DataFrame(calib_gm.achieved_distribution, columns=normits.zone_ids, index=normits.zone_ids) # save somewhere
