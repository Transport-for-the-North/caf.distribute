import pandas as pd
import numpy as np
from caf.toolkit import cost_utils
from caf.distribute import gravity_model as gm, furness, cost_functions

def tld_from_costs(costs, matrix, area_lookup):
    tlds = {}
    for area in area_lookup.index.unique():
        zones = area_lookup.loc[area, 'zone']
        inner_matrix = matrix.loc[zones]
        inner_costs = costs.loc[zones]
        bins = cost_utils.create_log_bins(inner_costs.max().max())
        tld = cost_utils.cost_distribution(inner_matrix, inner_costs, bin_edges=bins)
        tld = pd.DataFrame({'demand': tld, 'bin_max': bins[1:], 'bin_min': bins[:-1]})
        tlds[area] = tld.set_index(['bin_min', 'bin_max'])
    return pd.concat(tlds)

def run_sectoral(zonal_tripends, tld, translation, zonal_costs, tld_lookup, sec_mat):
    inp = gm.MultiDistInput(tld_file=tld,
                            tld_lookup_file=tld_lookup,
                            init_params={'mu':2, 'sigma':1},
                            log_path=r"E:\noham\localisation\log.log",
                            lookup_zone_col='noham_local_id',
                            furness_jac=True
                            )
    model = gm.MultiAreaGravityModelCalibrator(row_targets=zonal_tripends['o'].values,
                                               col_targets=zonal_tripends['d'].values,
                                               cost_matrix=zonal_costs,
                                               cost_function=cost_functions.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
                                               params=inp)

    sec_inputs = furness.SectoralConstraintInputs(translation,
                                                  'noham_local_id',
                                                  'noham_base_id',
                                                  'noham_local_to_noham_base',
                                                  sec_mat,
                                                  zonal_zones=zonal_tripends.index.values,
                                                  outer_max_iters=100
                                                  )
    sec_results = model.run(four_d_inputs=sec_inputs)

    return sec_results, model.achieved_distribution

def main(sec_tripend_path,
         zon_tripend_path,
         tlds_path,
         tld_lookup_path,
         trans_path,
         zon_costs_path,
         target_mat_path,
         ):
    sec_tripends = pd.read_csv(sec_tripend_path, index_col=0)
    zon_tripends = pd.read_csv(zon_tripend_path, index_col=0).sort_index()
    tld_lookup = pd.read_csv(tld_lookup_path, index_col=0)
    tlds = pd.read_csv(tlds_path)
    tlds.columns = ['cat', 'min', 'max', 'trips']
    tlds['ave'] = (tlds['min'] + tlds['max']) / 2
    trans = pd.read_csv(trans_path)
    zon_costs = pd.read_csv(zon_costs_path).set_index(['o','d'])['dist'].unstack()

    # zon_costs.columns = zon_costs.columns.droplevel(0)
    mat = pd.read_csv(
        target_mat_path,
        index_col=0, names=range(1, 2771))
    mat = mat.loc[np.sort(trans['noham_base_id'].unique()), np.sort(trans['noham_base_id'].unique())].to_numpy()
    # sec_tripends = sec_tripends.loc[trans['noham_base_id'].unique()]
    missing = pd.DataFrame(
        {'zone': [i for i in range(1, 2771) if i not in tld_lookup['zone'].values]})
    missing['cat'] = 'External'
    tld_lookup = pd.concat([tld_lookup, missing]).set_index('zone').sort_index().loc[
        sec_tripends.index]
    tld_lookup.index.name = 'zone'
    tld_lookup.reset_index(inplace=True)
    local_lookup = tld_lookup.merge(trans, right_on='noham_base_id', left_on='zone', how='inner')[
        ['cat', 'noham_local_id']].sort_values('noham_local_id')
    results, dist = run_sectoral(zon_tripends, tlds, trans, zon_costs.values, local_lookup, mat)
    dist_df = pd.DataFrame(dist, columns=zon_tripends.index, index=zon_tripends.index)
    return results, dist_df

