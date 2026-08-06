"""Bangun ketiga dataset (raw_integrated, final <=2025, holdout 2026) dari
berkas sumber di data/raw/external. Jalankan: python data/build_datasets.py
"""

import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent
RAW = DATA / "raw" / "external"
PROC = DATA / "processed"

ORDER = ["ID", "Bulan", "Tahun"]

# Sumber dataset MENTAH (apa adanya: PDB & kredit kuartalan, cadangan miliar)
RAW_SOURCES = [
    RAW / "Inflation_YoY_Pct.csv",
    RAW / "Inflation_Gap.csv",
    RAW / "GDP_Growth_YoY_Pct.csv",                       # kuartalan
    RAW / "USD_IDR_Monthly_Avg.csv",
    RAW / "M2_Triliun_Rp.csv",
    PROC / "Credit_Total_SEKI_Monthly.csv",               # kredit bulanan (level, BI-SEKI)
    RAW / "IHSG_End_of_Month.csv",
    PROC / "Foreign_Reserves_Miliar_Rp.csv",              # juta USD (skala konsisten)
    RAW / "Federal_Funds_Rate_Pct.csv",
    RAW / "Oil_Price_Brent_USD_per_Bbl.csv",
    RAW / "Gold_Price_USD_per_Oz.csv",
    RAW / "VIX_Volatility_Index.csv",
    RAW / "BI_Rate_Pct.csv",
]

# Sumber dataset FINAL (versi terproses untuk PDB, kredit, cadangan devisa)
FINAL_SOURCES = [
    RAW / "Inflation_YoY_Pct.csv",
    RAW / "Inflation_Gap.csv",
    PROC / "GDP_Growth_YoY_Monthly_Final.csv",            # diisi bulanan
    RAW / "USD_IDR_Monthly_Avg.csv",
    RAW / "M2_Triliun_Rp.csv",
    PROC / "Credit_Growth_YoY_Pct.csv",                   # kredit %YoY (BI-SEKI)
    RAW / "IHSG_End_of_Month.csv",
    PROC / "Foreign_Reserves_Miliar_Rp.csv",              # juta USD
    RAW / "Federal_Funds_Rate_Pct.csv",
    RAW / "Oil_Price_Brent_USD_per_Bbl.csv",
    RAW / "Gold_Price_USD_per_Oz.csv",
    RAW / "VIX_Volatility_Index.csv",
    RAW / "BI_Rate_Pct.csv",
]


def _eu_fix(x):
    """Normalisasi format ribuan Eropa, mis. '32.208.4' -> '32208.4'."""
    x = str(x)
    if x.count(".") <= 1:
        return x
    head, tail = x.rsplit(".", 1)
    return head.replace(".", "") + "." + tail


def _base():
    base = pd.read_csv(RAW / "ID, Bulan, Tahun.csv")
    base["period"] = pd.to_datetime(
        base["Tahun"].astype(str) + "-" + base["Bulan"].astype(str) + "-01"
    ).dt.strftime("%Y-%m")
    return base.set_index("period")


def integrate(sources, coerce_numeric):
    base = _base()
    for path in sources:
        if not path.exists():
            print(f"  PERINGATAN: {path.name} tidak ada, dilewati.")
            continue
        df = pd.read_csv(path)
        if "period" not in df.columns:
            continue
        df["period"] = pd.to_datetime(df["period"]).dt.strftime("%Y-%m")
        df = df.drop_duplicates("period").set_index("period")
        new_cols = [c for c in df.columns if c not in base.columns]
        if not new_cols:
            continue
        col = new_cols[0]
        if coerce_numeric:
            df[col] = pd.to_numeric(df[col].astype(str).map(_eu_fix), errors="coerce")
        base = base.join(df[[col]], how="left")

    base = base.reset_index()
    cols = [c for c in base.columns if c not in ORDER + ["period"]]
    return base[ORDER + cols]


# Batas tahun pemodelan: data <= 2025 untuk latih/uji, 2026 sebagai holdout terpisah
HOLDOUT_YEAR = 2026


def main():
    # Dataset mentah penuh (s.d. data terbaru) untuk Bab III Data Understanding
    raw = integrate(RAW_SOURCES, coerce_numeric=True)
    raw.to_csv(PROC / "dataset_raw_integrated.csv", index=False)

    # Dataset terproses dipisah: historis (pemodelan) vs holdout 2026 (evaluasi)
    fin = integrate(FINAL_SOURCES, coerce_numeric=False)
    hist = fin[fin["Tahun"].astype(int) < HOLDOUT_YEAR]
    holdout = fin[fin["Tahun"].astype(int) >= HOLDOUT_YEAR]
    hist.to_csv(PROC / "dataset_final.csv", index=False)
    holdout.to_csv(PROC / "dataset_2026.csv", index=False)

    print(f"dataset_raw_integrated.csv : {raw.shape}  (mentah penuh, Bab III)")
    print(f"dataset_final.csv          : {hist.shape}  (historis s.d. 2025, pemodelan)")
    print(f"dataset_2026.csv           : {holdout.shape}  (holdout 2026, evaluasi)")


if __name__ == "__main__":
    main()
