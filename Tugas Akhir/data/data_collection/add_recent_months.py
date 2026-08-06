"""Tambahkan bulan terbaru: isi nilai domestik manual di berkas ini, lalu
jalankan (fetch pasar global -> ekstraksi PDB/inflasi/kredit -> build dataset).
Jalankan: python data/data_collection/add_recent_months.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1]
RAW = DATA / "raw" / "external"
PROC = DATA / "processed"

PERIODS = ["2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30", "2026-05-31"]
MONTHS = [(1, 2026), (2, 2026), (3, 2026), (4, 2026), (5, 2026)]

# Nilai domestik diisi manual dari rilis resmi BI/BPS (urutan Jan..Mei 2026);
# variabel pasar global di-fetch otomatis oleh fetch_global_markets.py.
SERIES = {
    # Sumber: Rapat Dewan Gubernur BI (rilis resmi)
    RAW / "BI_Rate_Pct.csv":                  [4.75, 4.75, 4.75, 4.75, 5.25],
    # Sumber: Siaran pers BPS (inflasi YoY bulanan)
    RAW / "Inflation_YoY_Pct.csv":            [3.55, 4.76, 3.48, 2.42, 3.08],
    # Sumber: Statistik Uang Beredar M2 — BI; Mar & Mei belum dirilis → kosong (NaN)
    RAW / "M2_Triliun_Rp.csv":                [10117.8, 10089.9, "", 10253.0, ""],
    # Sumber: Statistik Cadangan Devisa — BI; Mei belum dirilis → kosong (NaN)
    RAW / "Foreign_Reserves_Miliar_Rp.csv":   [154.6, 151.9, 148.2, 146.2, ""],
    # PDB mentah: kuartalan dari BPS; hanya Maret (akhir Q1-2026 = 5,61%).
    # Apr/Mei (Q2) belum dirilis → kosong; gdp.py tidak meng-carry-forward.
    RAW / "GDP_Growth_YoY_Pct.csv":           ["", "", 5.61, "", ""],
    # Berkas perantara terproses
    PROC / "inflasi.csv":                     [3.55, 4.76, 3.48, 2.42, 3.08],
    PROC / "Foreign_Reserves_Miliar_Rp.csv":  [154600.0, 151900.0, 148200.0, 146200.0, ""],
    # Catatan: kredit kini dari BI-SEKI (extract_credit_seki.py membaca berkas Excel
    # data/raw/Kredit_Bank_Umum_SEKI.xls), bukan ditambah manual di sini.
}


def append_period_csv(path, values):
    """Tambahkan baris bulanan ke CSV berkolom 'period' (idempoten)."""
    if not path.exists():
        print(f"  LEWAT (tidak ada): {path.name}")
        return 0
    df = pd.read_csv(path, dtype=str)
    valcol = next(c for c in df.columns if c != "period")
    existing = set(df["period"].astype(str).str[:7])
    rows = [{"period": p, valcol: ("" if v == "" else str(v))}
            for p, v in zip(PERIODS, values) if p[:7] not in existing]
    if rows:
        pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    print(f"  + {path.name}: {len(rows)} baris")
    return len(rows)


def append_master():
    path = RAW / "ID, Bulan, Tahun.csv"
    df = pd.read_csv(path)
    have = set(zip(df["Bulan"], df["Tahun"]))
    next_id = int(df["ID"].max()) + 1
    rows = []
    for m, y in MONTHS:
        if (m, y) not in have:
            rows.append({"ID": next_id + len(rows), "Bulan": m, "Tahun": y})
    if rows:
        pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    print(f"  + master ID/Bulan/Tahun: {len(rows)} baris")


def append_target_inflasi():
    path = PROC / "target_inflasi.csv"
    df = pd.read_csv(path)
    if 2026 in df["Year"].values:
        print("  target_inflasi: 2026 sudah ada")
        return
    df = pd.concat([df, pd.DataFrame([{
        "Year": 2026, "Inflation_Target_Center": 2.5,
        "Inflation_Target_Lower": 1.5, "Inflation_Target_Upper": 3.5,
        "Inflation_Actual_YoY": "",
    }])], ignore_index=True)
    df.to_csv(path, index=False)
    print("  + target_inflasi: 2026 (center 2,5%)")


def run_pipeline():
    steps = [
        # 1. Auto-fetch variabel pasar global dari FRED & Yahoo Finance
        Path(__file__).parent / "fetch_global_markets.py",
        # 2. PDB kuartalan -> bulanan (sebar dalam kuartal yang dirilis)
        DATA / "preprocessing" / "gdp.py",
        # 3. inflasi - target -> Inflation_Gap
        DATA / "preprocessing" / "inflation_gap.py",
        # 4. Kredit bulanan dari berkas Excel BI-SEKI (level + %YoY)
        DATA / "preprocessing" / "extract_credit_seki.py",
        # 5. Integrasi kedua dataset (mentah & final)
        DATA / "build_datasets.py",
    ]
    for s in steps:
        print(f"\n>> {s.name}")
        subprocess.run([sys.executable, str(s)], check=True)


def main():
    print("Menulis nilai bulan baru ke berkas sumber...")
    append_master()
    append_target_inflasi()
    for path, values in SERIES.items():
        append_period_csv(path, values)
    print("\nRegenerasi dataset melalui pipeline...")
    run_pipeline()
    print("\nSelesai. Bulan tanpa rilis resmi dibiarkan kosong (diimputasi eksperimen);"
          "\njalankan ulang skrip ini saat data resmi sudah tersedia.")


if __name__ == "__main__":
    main()
