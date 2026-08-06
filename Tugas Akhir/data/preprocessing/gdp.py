from pathlib import Path

import pandas as pd

# Path relatif terhadap folder data/
data_dir = Path(__file__).resolve().parent.parent

template_path = data_dir / "raw" / "external" / "GDP_Growth_YoY_Pct.csv"
output_path = data_dir / "processed" / "GDP_Growth_YoY_Monthly_Final.csv"

# 1. Load Data
template = pd.read_csv(template_path)
template["period"] = pd.to_datetime(template["period"])

# Force numeric for template value
template["GDP_Growth_YoY_Pct"] = pd.to_numeric(
    template["GDP_Growth_YoY_Pct"], errors="coerce"
)

# Sort
template = template.sort_values("period")

# 2. Sebar nilai kuartalan ke tiap bulan DALAM kuartal yang sama.
# PDB rilis BPS bersifat kuartalan; satu laju pertumbuhan berlaku untuk ketiga
# bulan kuartal itu (representasi sah variabel kuartalan pada frekuensi bulanan).
template['year'] = template['period'].dt.year
template['quarter'] = template['period'].dt.quarter
template["GDP_Growth_YoY_Pct"] = template.groupby(["year", "quarter"])["GDP_Growth_YoY_Pct"].transform(
    lambda x: x.bfill().ffill()
)
template = template.drop(columns=['year', 'quarter'])

# Bulan pada kuartal yang BELUM dirilis BPS (mis. 2005-2010 sebelum data ada, atau
# kuartal terkini yang belum terbit) DIBIARKAN kosong (NaN). Imputasinya ditentukan
# secara ilmiah oleh faktor imputation pada eksperimen, bukan diisi paksa di sini.

# 3. Save Output
template.to_csv(output_path, index=False)

n_isi = template["GDP_Growth_YoY_Pct"].notna().sum()
print(f"PDB diproses: nilai kuartalan disebar dalam kuartal ({n_isi}/{len(template)} "
      f"bulan terisi). Bulan tanpa rilis BPS dibiarkan kosong untuk diimputasi eksperimen.")
