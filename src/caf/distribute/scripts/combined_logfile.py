from pathlib import Path
import re
import pandas as pd

base_dir = Path(r"D:\normits\localisation\distribution\distribute_outputs\car")

rows = []

for p_folder in sorted(base_dir.glob("p*")):
    if not p_folder.is_dir():
        continue

    for csv_file in sorted(p_folder.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)

            if len(df) == 0:
                continue

            # Take final row
            last_row = df.iloc[-1].copy()

            # Store filename as identifier
            last_row.name = csv_file.stem

            rows.append(last_row)

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

# Combine all rows
result = pd.DataFrame(rows)

# Use filenames as index
result.index.name = "csv_file"

result = result.loc[
    sorted(
        result.index,
        key=lambda x: (
            int(re.search(r"_p(\d+)_", x).group(1)),
            int(re.search(r"_ts(\d+)", x).group(1)),
            x,
        ),
    )
]

output_file = base_dir / "complete_log_file.csv"
result.to_csv(output_file)

print(f"Saved to: {output_file}")
print(f"Rows collected: {len(result)}")
