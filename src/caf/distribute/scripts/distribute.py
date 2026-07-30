"""
Implementation of a self-calibrating multi area gravity model with
post ME furness adjustment.
"""

# Built-Ins
import logging
import os
import pathlib

# Third Party
import caf.base as cb
import pandas as pd
from caf.distribute import cost_functions, furness, gravity_model, utils
from caf.toolkit.concurrency import multiprocess
from caf.toolkit.config_base import BaseConfig

# Local Imports
from caf.mat.matrices import MatrixFiles, MatrixType

_ADJ_CHOICE_CONFIG = {
    (True, False, False, False): ("0_dist_adj", ["dist"]),
    (True, True, True, False): ("1_dist_od_adj", ["dist", "origin", "dest"]),
    (True, True, True, True): ("2_dist_od_sec_adj", ["dist", "origin", "dest", "sec"]),
    (True, False, False, True): ("3_dist_sec_adj", ["dist", "sec"]),
    (False, True, True, False): ("4_od_adj", ["origin", "dest"]),
    (False, True, True, True): ("5_od_sec_adj", ["origin", "dest", "sec"]),
    (False, False, False, True): ("6_sec_adj", ["sec"]),
}

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


class DistributeConf(BaseConfig):
    """
    Configuration dataclass for the distribute module, defining all necessary parameters
    and file paths for executing the 4D constraint gravity model with post ME furness
    adjustment.
    This class is designed to be loaded from a YAML configuration file
    (distribute_config.yml) and provides structured access to all configuration options
    used in the main execution flow.

    Attributes - data types are defined in code, descriptions are provided in comments
    ----------
    mode_subset : Subset of modes to include in the run.
    timeperiod_subset : Subset of time periods to include in the run.
    purpose_subset : Subset of purposes to include in the run.
    direction_subset : Subset of directions to include in the run.
    tld_lookup_path : File path to the TLD lookup CSV file.
    zone_system : Name of the zoning system used for matrices.
    cost_files : Dictionary containing configuration for cost matrix files, including
                naming order, folder path, and filename template.
    tld_files : Dictionary containing configuration for TLD files, including naming
                order, folder path, and filename template.
    trip_ends : Dictionary containing file paths for trip ends data, including production
                and attraction vectors for NHB and HB (fr and to).
    gm_run_name : Identifier name for the gravity model run, used in output naming.
    output_path : Root output directory where results will be written.
    max_process : Maximum number of processes for parallel execution.
    """

    mode_subset: int | list[int]
    timeperiod_subset: int | list[int]
    purpose_subset: int | list[int]
    direction_subset: int | list[int]
    tld_lookup_path: pathlib.Path
    zone_system: str
    cost_files: dict[str, list[str] | pathlib.Path | str]
    tld_files: dict[str, list[str] | pathlib.Path | str]
    trip_ends: dict[str, pathlib.Path]
    gm_run_name: str
    output_path: pathlib.Path
    max_process: int


# pylint: disable=too-many-locals
def seg_furness(
    current_slice,
    cost_distributions,
    calib_gm: gravity_model.MultiAreaGravityModelCalibrator,
    out_dir: pathlib.Path,
    tld_lookup: pd.Series,
):
    """
    Execute gravity model calibration and/or Furness adjustment for a given slice,
    outputs results and relevant diagnostic files to the specified output directory.
    Designed to be run in parallel across multiple slices.

    Parameters
    ----------
    current_slice : dict-like
        Slice object containing slice-specific information. Must have methods
        generate_name() and get(key) for extracting slice identifier and purpose.
    cost_distributions : object
        Cost distribution object with a distributions attribute containing
        distribution data for gravity model calibration.
    constraint_area_trans : pd.DataFrame
        DataFrame mapping zone IDs (normits_id) to sector IDs (noham_sector_id).
    sector_target_furnessed : pd.DataFrame
        Sector target matrix containing target values by origin and destination sector
        for the adjustment algorithm.
    calib_gm : gravity_model.MultiAreaGravityModelCalibrator
        Calibrated or pre-configured gravity model object used for calibration
        and distribution calculations.
    out_dir : pathlib.Path
        Root output directory where results will be written. Results are
        then organized by purpose subdirectories.
    tld_lookup : pd.Series
        Series mapping internal zone indices to zone names/IDs for matrix indexing.

    Returns
    -------
    None
        Results are written directly to disk in output directories organized by
        purpose, slice and adjustment type.
    """

    slice_name = current_slice.generate_name()
    purpose = f"p{current_slice.get('p')}"
    os.makedirs(os.path.join(out_dir, purpose), exist_ok=True)
    csv_logging_path = out_dir / purpose / f"{slice_name}_log.csv"
    output_path = out_dir / purpose / slice_name
    band_targets, band_lookup = calib_gm.multi_props(cost_distributions.distributions)
    band_targets.index.names = ["area", "band_start", "band_end"]
    used = pd.MultiIndex.from_frame(
        band_lookup.reset_index()[["area", "band_start", "band_end"]].drop_duplicates()
    )
    dropped = band_targets.drop(used).sum()
    LOG.warning(
        "Total demand dropped from band targets due to missing in band_lookup: %s",
        dropped.sum(),
    )

    band_targets = band_targets.loc[used]

    gravity_model_results, dists = calib_gm.calibrate(  # pylint: disable=unused-variable
        cost_distributions,
        csv_logging_path,
        output_path,
        gravity_model.GMCalibParams(furness_jac=True, ftol=1e-2, xtol=1e-2),
        return_distributions=True,
    )


# pylint: disable=too-many-arguments
def _4d_constraint_gravity_model(
    row_trip_ends: cb.DVector,
    col_trip_ends: cb.DVector,
    name: str,
    tlds: dict[str, pathlib.Path],
    tld_zones: pd.Series,
    cost_matrix: MatrixFiles,
    out_dir: pathlib.Path,
    max_process: int,
):
    """
    Execute a 4D constraint gravity model with optional calibration and adjustment.
    This function processes trip distribution data using a gravity model approach,
    handling multiple time periods (and applying sector-level constraints and
    adjustments). It prepares inputs for parallel processing of gravity model
    calibration and furness adjustments using seg_furness().

    Parameters
    ----------
    row_trip_ends : cb.DVector
        Origin (row) trip ends vector with segmentation information.
    col_trip_ends : cb.DVector
        Destination (column) trip ends vector with segmentation information.
    name : str
        Identifier name for the gravity model run.
    tlds : dict[str, pathlib.Path]
        Dictionary mapping segment identifiers to Trip Length Distribution (TLD)
        file paths.
    tld_zones : pd.Series
        Series containing TLD zone information indexed by zone identifiers.
    cost_matrix : MatrixFiles
        Cost matrix object containing distance file information.
    constraint_area_trans : pd.DataFrame
        Translation/mapping dataframe between normits_id and noham_sector_id for
        spatial constraint areas.
    sector_target_matrix : MatrixFiles
        MatrixFiles object for target matrices for sector-level constraint targets.
    calibrate : bool
        Flag indicating whether to calibrate the gravity model (currently unused).
    out_dir : pathlib.Path
        Output directory path for results.
    run_gm : bool
        Flag to execute gravity model; if False, disables multiprocessing.
    max_process : int
        Maximum number of processes for parallel execution.
        Will be set to 0 if run_gm is False (no calibrate) as reading multiple files
        into the same variable causes issues with multiprocessing.
        Value recommendations:
            2 is recommended for full runs due to RAM constraints (usage can spike
                to 50-60GB),
            3-4 can be used for testing individual time periods.

    Returns
    -------
    None
        Results are written to out_dir through seg_furness() and multiprocessing callback.

    Raises
    ------
    ValueError
        If run_adjust is True but no adjustment targets are enabled in
        adj_target_options.

    Warnings
    --------
    - Sector target matrix is rescaled by row sum ratio, potentially overwriting
      furness adjustment results.
    - The calibrate parameter is defined but not utilized in function logic.
    - Empty adj_target_options dict will raise ValueError if run_adjust=True.

    Notes
    -----
    - Processing occurs iteratively per segment slice from row_trip_ends.
    - Distance distributions are calculated using logarithmic normal cost function.
    - Furness adjustment is applied at sector level before gravity model execution.
    - Multiprocessing count is conditional on run_gm flag.
    """
    inputs = []
    for current_slice in row_trip_ends.segmentation.iter_slices():
        row = row_trip_ends.get_slice(current_slice)
        col = col_trip_ends.get_slice(current_slice)
        LOG.info(
            "Difference between the input trip end productions and attractions: %s, as a percentage of the input productions: %s%%",
            row.sum() - col.sum(),
            (row.sum() - col.sum()) / row.sum() * 100
        )
        cost = cost_matrix.get_matrix(
            current_slice.aggregate(cost_matrix.segmentation.naming_order)
        )

        # infill intrazonal costs and restore row/col labels
        # change from meters to kms to match tld
        cost_index = cost.data.index
        cost_columns = cost.data.columns
        cost_infill = pd.DataFrame(utils.infill_cost_matrix(cost.data.to_numpy()))
        cost_infill.index = cost_index
        cost_infill.columns = cost_columns
        cost.data = cost_infill / 1000
 
        tld = pd.read_csv(tlds[current_slice.aggregate(["p", "direction_od"]).generate_name()])
        tld["from"] = tld["trav_dist"].shift().fillna(0)
        tld.loc[tld["from"] > tld["trav_dist"], "from"] = 0
        tld["ave_dist"] = (tld["trav_dist"] + tld["from"]) / 2

        LOG.info("Running Gravity Model: %s, for slice %s.", name, current_slice)

        cost_function = cost_functions.BuiltInCostFunction.GAUSSIAN.get_cost_function()

        cost_distributions = gravity_model.MultiCostDistribution.from_pandas(
            tld=tld,
            cat_zone_correspondence=tld_zones,
            func_params={i: cost_function.default_params for i in tld_zones["tld_area"].unique()},
            tld_cat_col=";index",
            tld_min_col="from",
            tld_max_col="trav_dist",
            tld_avg_col="ave_dist",
            tld_trips_col="trips",
            lookup_cat_col="tld_area",
            lookup_zone_col="cumbria_local_id",
        )

        calib_gm = gravity_model.MultiAreaGravityModelCalibrator(
            row,
            col,
            cost.data,
            cost_function,
        )

        inputs.append(
            (
                current_slice,
                cost_distributions,
#                constraint_area_trans,
                calib_gm,
                out_dir,
                tld_zones["cumbria_local_id"],  # change id
                # run_gm ?
            )
        )
    seg_furness(*inputs[0])
    multiprocess(seg_furness, arg_list=inputs, process_count=max_process)


def _use_as_list(input_list: int | list[int]) -> list[int]:
    """
    Function to ensure that input is always returned as a list,
    even if a single integer is provided. This is because the subsets
    parameter in SegmentationInput requires a dictionary with list values.

    Parameters
    ----------
    input_list : int or list of int
        The input value(s) to be converted to a list.

    Returns
    -------
    list of int
        The input value(s) as a list.
    """
    return input_list if isinstance(input_list, list) else [input_list]


def main(cfg: DistributeConf):
    """
    Main function to execute the 4D constraint gravity model with post ME furness adjustment.
    All inputs are defined in the distribute_config.yml file and loaded into the DistributeConf dataclass.
    """
    log_path = pathlib.Path(cfg.output_path).parent / f"{cfg.gm_run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOG.setLevel(logging.INFO)
    LOG.addHandler(file_handler)

    # Save config file contents to the LOG for reference
    LOG.info(
        "Distribution config file contents: ===================================================================\n%s\n===================================================================\n",
        cfg.to_yaml(),
    )

    # To improve memory efficiency, each tp is processed separately.
    # This ensures the loaded DVectors aren't too large but multiprocessing will still work
    for tp_subset in cfg.timeperiod_subset:

        # --- Segment subsets -------------------------------------------------------
        m_subset = cfg.mode_subset
        p_subset = cfg.purpose_subset
        direction_od_subset = cfg.direction_subset
        direction_od_list = _use_as_list(direction_od_subset)

        # --- Lookups and zoning systems --------------------------------------------
        tld_lookup = pd.read_csv(cfg.tld_lookup_path)
        zoning = cb.ZoningSystem.get_zoning(cfg.zone_system)

        # z2s_cfg removed

        # --- Split p into HB and NHB -----------------------------------------------
        hb_p_subset = (
            [p for p in p_subset if p in list(range(1, 9))]
            if isinstance(p_subset, list)
            else (p_subset if p_subset in list(range(1, 9)) else None)
        )
        nhb_p_subset = (
            [p for p in p_subset if p in list(range(11, 19))]
            if isinstance(p_subset, list)
            else (p_subset if p_subset in list(range(11, 19)) else None)
        )

        # adj_target_purpose removed 

        # --- Cost files ------------------------------------------------------------
        cost_cfg = cfg.cost_files
        cost_seg = cb.Segmentation(
            cb.SegmentationInput(
                enum_segments=cost_cfg["naming_order"],
                naming_order=cost_cfg["naming_order"],
                subsets={"m": _use_as_list(m_subset)},
            )
        )
        costs = MatrixFiles(
            cost_seg,
            zoning,
            MatrixType.OD,
            pathlib.Path(cost_cfg["folder_path"]),
            filename_template=cost_cfg["filename_template"],
        )

        # --- TLD files -------------------------------------------------------------
        tld_cfg = cfg.tld_files
        tld_seg = cb.Segmentation(
            cb.SegmentationInput(
                enum_segments=tld_cfg["naming_order"],
                naming_order=tld_cfg["naming_order"],
                subsets={
                    "p": _use_as_list(p_subset),
                    "direction_od": _use_as_list(direction_od_subset),
                },
            )
        )
        tlds = {}
        tld_dir = pathlib.Path(tld_cfg["folder_path"])
        tld_template = tld_cfg["filename_template"]
        for tld_slice in tld_seg.iter_slices():
            tld_name = tld_slice.generate_name()
            file_name = tld_name.replace("fr", "hb_fr").replace("to", "hb_to")
            tlds[tld_name] = tld_dir / tld_template.format(slice_name=file_name)

        # --- Trip ends -------------------------------------------------------------
        te_cfg = cfg.trip_ends
        dvec_map = {}
        if 0 in direction_od_list:
            prod_nhb = (
                cb.DVector.load(te_cfg["prod_nhb"])
                .aggregate(["p", "m", "tp"])
                .filter_segment_value("p", nhb_p_subset, keep_filtered=True)
                .filter_segment_value("m", m_subset, keep_filtered=True)
                .filter_segment_value("tp", tp_subset, keep_filtered=True)
                .aggregate_comp_zones(zoning)
            )
            attr_nhb = (
                cb.DVector.load(te_cfg["attr_nhb"])
                .aggregate(["p", "m", "tp"])
                .filter_segment_value("p", nhb_p_subset, keep_filtered=True)
                .filter_segment_value("m", m_subset, keep_filtered=True)
                .filter_segment_value("tp", tp_subset, keep_filtered=True)
                .aggregate_comp_zones(zoning)
            )
            dvec_map[0] = (prod_nhb, attr_nhb)
        if 1 in direction_od_list:
            hb_prod_fr = (
                cb.DVector.load(te_cfg["hb_prod_fr"])
                .aggregate(["p", "m", "tp"])
                .filter_segment_value("p", hb_p_subset, keep_filtered=True)
                .filter_segment_value("m", m_subset, keep_filtered=True)
                .filter_segment_value("tp", tp_subset, keep_filtered=True)
                .aggregate_comp_zones(zoning)
            )
            hb_attr_fr = (
                cb.DVector.load(te_cfg["hb_attr_fr"])
                .aggregate(["p", "m", "tp"])
                .filter_segment_value("p", hb_p_subset, keep_filtered=True)
                .filter_segment_value("m", m_subset, keep_filtered=True)
                .filter_segment_value("tp", tp_subset, keep_filtered=True)
                .aggregate_comp_zones(zoning)
            )
            dvec_map[1] = (hb_prod_fr, hb_attr_fr)
        if 2 in direction_od_list:
            hb_prod_to = (
                cb.DVector.load(te_cfg["hb_prod_to"])
                .aggregate(["p", "m", "tp"])
                .filter_segment_value("p", hb_p_subset, keep_filtered=True)
                .filter_segment_value("m", m_subset, keep_filtered=True)
                .filter_segment_value("tp", tp_subset, keep_filtered=True)
                .aggregate_comp_zones(zoning)
            )
            hb_attr_to = (
                cb.DVector.load(te_cfg["hb_attr_to"])
                .aggregate(["p", "m", "tp"])
                .filter_segment_value("p", hb_p_subset, keep_filtered=True)
                .filter_segment_value("m", m_subset, keep_filtered=True)
                .filter_segment_value("tp", tp_subset, keep_filtered=True)
                .aggregate_comp_zones(zoning)
            )
            dvec_map[2] = (hb_prod_to, hb_attr_to)

        prod = pd.concat({k: v[0].data for k, v in dvec_map.items()})
        attr = pd.concat({k: v[1].data for k, v in dvec_map.items()})
        prod.index.names = ["direction_od", "p", "m", "tp"]  # enum segments + naming order. Subsets is m-1, no p, tp, 1,2,3,4
        full_seg_p = cb.Segmentation(
            cb.SegmentationInput(
                enum_segments=prod.index.names,
                naming_order = prod.index.names,
                subsets={
                    "m": _use_as_list(m_subset),
                    "p": _use_as_list(p_subset),
                    "tp": _use_as_list(tp_subset),
                    "direction_od": _use_as_list(direction_od_subset),
                }
            )
        )
        prod = cb.DVector(import_data=prod, segmentation=full_seg_p, zoning_system=zoning)
        attr.index.names = ["direction_od", "p", "m", "tp"]
        attr = cb.DVector(import_data=attr, segmentation=full_seg_p, zoning_system=zoning)

        # --- Run -------------------------------------------------------------------
#        run_opts = cfg.run_options
        _4d_constraint_gravity_model(
            prod,
            attr,
            cfg.gm_run_name,
            tlds,
            tld_lookup,
            costs,
            pathlib.Path(cfg.output_path),
            cfg.max_process,
        )

if __name__ == "__main__":

    distribute_config = DistributeConf.load_yaml(
        pathlib.Path(__file__).parent / "distribute_config.yml"
    )

    main(distribute_config)