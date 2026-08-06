import pandas as pd
from pathlib import Path
import calendar
import math

pdb_dir = Path(__file__).parent.parent / 'raw' / 'bps' / 'pdb'
output_file = Path(__file__).parent.parent / 'raw' / 'external' / 'GDP_Growth_YoY_Pct.csv'

start_year = 2011
end_year = 2025

# Map to store quarter-end-month period -> GDP yoy value
gdp_map = {}

for year in range(start_year, end_year + 1):
    file_name = f"[Seri 2010] Laju Pertumbuhan PDB Seri 2010, {year}.csv"
    file_path = pdb_dir / file_name
    if not file_path.exists():
        print(f"Warning: file missing: {file_name}")
        continue

    df = pd.read_csv(file_path, header=None)

    # Find the row for 'c. Produk Domestik Bruto'
    target_idx = None
    for idx, row in df.iterrows():
        first = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
        if 'PRODUK DOMESTIK BRUTO' in first.upper():
            target_idx = idx
            break

    if target_idx is None:
        print(f"Warning: target row not found in {file_name}")
        continue

    row = df.iloc[target_idx]

    # Y-on-y quarterly values (Triwulan I–IV)
    vals = []
    for col in [11, 12, 13, 14]:
        if col < len(row):
            try:
                vals.append(float(row.iloc[col]))
            except Exception:
                vals.append(float('nan'))
        else:
            vals.append(float('nan'))

    # Correct: quarter -> end-of-quarter month
    quarter_months = {
        1: 3,   # Q1 -> March
        2: 6,   # Q2 -> June
        3: 9,   # Q3 -> September
        4: 12   # Q4 -> December
    }

    for q in range(1, 5):
        month = quarter_months[q]
        day = calendar.monthrange(year, month)[1]
        period = f"{year:04d}-{month:02d}-{day:02d}"
        gdp_map[period] = vals[q - 1]

# Build output monthly periods (keep template as-is)
output_rows = []
start_year2, start_month = 2005, 7
end_year2, end_month = 2025, 12

cy, cm = start_year2, start_month
while (cy, cm) <= (end_year2, end_month):
    day = calendar.monthrange(cy, cm)[1]
    period = f"{cy:04d}-{cm:02d}-{day:02d}"

    val = gdp_map.get(period, float('nan'))
    output_rows.append({
        'period': period,
        'GDP_Growth_YoY_Pct': '' if (isinstance(val, float) and math.isnan(val)) else round(val, 2)
    })

    cm += 1
    if cm > 12:
        cm = 1
        cy += 1

out_df = pd.DataFrame(output_rows)
out_df.to_csv(output_file, index=False)

print(f"Saved quarterly GDP YoY to: {output_file}")
filled = out_df['GDP_Growth_YoY_Pct'].astype(str).str.strip().ne('').sum()
print(f"Total periods: {len(out_df)}")
print(f"Filled quarter-end months: {filled}")
