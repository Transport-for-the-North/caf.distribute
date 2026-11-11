# aggregate trips to MSOA

import re

import pandas as pd
import caf.toolkit as ctk

from pathlib import Path

input_name = "normits"  # "cumbria_local"
aggregation = "lad"  # "normits" #   # "msoa"

lookup = pd.read_csv(rf"D:\normits\data\lookups\output\{input_name}_to_{aggregation}_lookup.csv")
# lookup = lookup[(lookup["cumbria_local_id"] >= 1000000) & (lookup["cumbria_local_id"] < 2000000)]
zone_correspondence = ctk.translation.ZoneCorrespondence(
    vector = lookup,
    from_col_name = f"{input_name}_id",
    to_col_name = f"{aggregation}_id",
    factors_col_name = f"{input_name}_to_{aggregation}"
)

# Provide output path
demand_folder = Path(r"I:\Prior adjustment\distribution\distribute_outputs\fullrun_06_03_v2")
# Path(r"D:\normits\localisation\distribution\distribute_outputs\car")

#rows = []

for p_folder in demand_folder.glob("p1*"):
    if not p_folder.is_dir():
        continue

    for subfolder in p_folder.iterdir():
        if not subfolder.is_dir():
            continue

        matrix_path = subfolder / "overall_matrix.csv"
        matrix = ctk.io.read_csv_matrix(matrix_path, format_="square")

        #matrix_sum = matrix.sum().sum()
        #rows.append({
        #    "file_name": subfolder.name,
        #    "total_trips": matrix_sum
        #    })
        matrix_agg = ctk.translation.pandas_matrix_zone_translation(
            matrix=matrix,
            zone_correspondence=zone_correspondence 
            )

        matrix_agg.to_csv(Path(subfolder, f"{aggregation}_matrix.csv"))

## Combine all rows
#result = pd.DataFrame(rows)
#
## Use filenames as index
#result.index = result["file_name"]
#result.index.name = "csv_file"
#
#result = result.loc[
#    sorted(
#        result.index,
#        key=lambda x: (
#            int(re.search(r"_p(\d+)_", x).group(1)),
#            int(re.search(r"_ts(\d+)", x).group(1)),
#            x,
#        ),
#    )
#]
#
#output_file = demand_folder / "total_sum.csv"
#result.to_csv(output_file)
