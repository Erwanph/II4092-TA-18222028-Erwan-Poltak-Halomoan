import pandas as pd

from pathlib import Path

# Setup paths relative to the data directory
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent

inflasi_path = data_dir / "processed" / "inflasi.csv"
target_path = data_dir / "processed" / "target_inflasi.csv"
output_path = data_dir / "raw" / "external" / "Inflation_Gap.csv"

# Load Data
inflasi = pd.read_csv(inflasi_path)
target = pd.read_csv(target_path)

# Prepare Date
inflasi["period"] = pd.to_datetime(inflasi["period"])
inflasi["Year"] = inflasi["period"].dt.year

# Merge Target ke Bulanan
df = inflasi.merge(target[["Year", "Inflation_Target_Center"]], on="Year", how="left")

# Hitung Inflation Gap
df["Inflation_Gap_Pct"] = df["Inflation_YoY_Pct"] - df["Inflation_Target_Center"]

# Final Format
out = df[["period", "Inflation_Gap_Pct"]].copy()
out["period"] = out["period"].dt.strftime("%Y-%m-%d")

# Save
out.to_csv(output_path, index=False)

print(f"Inflation_Gap.csv berhasil dibuat di:\n{output_path}")
