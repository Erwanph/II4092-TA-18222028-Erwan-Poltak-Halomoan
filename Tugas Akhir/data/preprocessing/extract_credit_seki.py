"""Ekstrak total kredit bulanan (bank umum + BPR) dari Excel BI-SEKI
Tabel I.4 menjadi level + %YoY.
"""

import re
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "raw" / "Kredit_Bank_Umum_SEKI.xls"
OUT = Path(__file__).resolve().parents[1] / "processed" / "Credit_Total_SEKI_Monthly.csv"
OUT_YOY = Path(__file__).resolve().parents[1] / "processed" / "Credit_Growth_YoY_Pct.csv"

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _find_jumlah_row(df: pd.DataFrame):
    """Baris total 'JUMLAH (1 s.d. 5)' (abaikan spasi, mis. 'J U M L A H')."""
    for col in range(min(4, df.shape[1])):
        for i, v in df[col].items():
            s = re.sub(r"\s+", "", str(v)).lower()
            if s.startswith("jumlah") or s.startswith("6.jumlah") or "jumlah(1" in s:
                return i
    return None


def _date_columns(df: pd.DataFrame):
    """Petakan kolom -> Timestamp akhir bulan. Dukung header datetime maupun
    baris tahun+bulan (tahun di-increment saat bulan kembali ke 'Jan')."""
    # Kasus A: ada baris berisi tanggal penuh (Timestamp ATAU string 'YYYY-MM-DD').
    # Pilih baris dengan tanggal terbanyak (sheet 2002-2017).
    def _is_date(x):
        return isinstance(x, pd.Timestamp) or bool(
            re.fullmatch(r"(19|20)\d{2}-\d{2}-\d{2}.*", str(x)))
    best_r, best_n = None, 0
    for r in range(8):
        n = df.iloc[r].apply(_is_date).sum()
        if n > best_n:
            best_r, best_n = r, n
    if best_n >= 6:
        out = {}
        for c, v in df.iloc[best_r].items():
            if _is_date(v):
                dt = pd.to_datetime(str(v).split()[0], errors="coerce")
                if pd.notna(dt):
                    out[c] = dt + pd.offsets.MonthEnd(0)
        return out

    # Kasus B: baris tahun + baris bulan. Sel tahun BI sering ber-merged di posisi
    # tak konsisten, jadi kita pakai HANYA tahun eksplisit PERTAMA sebagai awal,
    # lalu tentukan tahun murni dari urutan bulan (increment saat balik ke Jan).
    year_row = month_row = None
    for r in range(8):
        cells = df.iloc[r].astype(str).str.strip()
        if cells.str.fullmatch(r"(19|20)\d{2}(\.0)?").sum() >= 1 and year_row is None:
            year_row = r
        if cells.str.lower().str[:3].isin(MONTHS).sum() >= 3 and month_row is None:
            month_row = r
    if month_row is None or year_row is None:
        return {}

    start_year = None
    for v in df.iloc[year_row].tolist():
        mt = re.fullmatch(r"((19|20)\d{2})(\.0)?", str(v).strip())
        if mt:
            start_year = int(mt.group(1))
            break
    if start_year is None:
        return {}

    out = {}
    cur_year = start_year
    prev_m = None
    for c in range(df.shape[1]):
        mraw = str(df.iat[month_row, c]).strip().lower()[:3]
        if mraw not in MONTHS:
            continue
        m = MONTHS[mraw]
        if prev_m is not None and m == 1 and prev_m != 1:
            cur_year += 1  # bulan membungkus Des->Jan
        prev_m = m
        out[c] = pd.Timestamp(year=cur_year, month=m, day=1) + pd.offsets.MonthEnd(0)
    return out


def parse_sheet(xl, sheet):
    df = pd.read_excel(xl, sheet_name=sheet, header=None)
    rj = _find_jumlah_row(df)
    if rj is None:
        return pd.Series(dtype=float)
    cols = _date_columns(df)
    rows = {}
    for c, dt in cols.items():
        v = df.iat[rj, c]
        v = pd.to_numeric(v, errors="coerce")
        if pd.notna(v):
            rows[dt] = float(v) / 1000.0  # Miliar -> Triliun
    return pd.Series(rows).sort_index()


def main():
    xl = pd.ExcelFile(SRC)
    # urutan prioritas: sheet terbaru menimpa yang lama saat overlap
    order = ["Th 2002-2017", "Th 2016-2024", "I.4_1", "I.4_3"]
    sheets = [s for s in order if s in xl.sheet_names]
    merged = pd.Series(dtype=float)
    for s in sheets:
        ser = parse_sheet(xl, s)
        if ser.empty:
            print(f"  {s}: JUMLAH tidak ditemukan / kosong")
            continue
        print(f"  {s}: {len(ser)} bulan, {ser.index.min().date()}..{ser.index.max().date()}")
        merged = ser.combine_first(merged)  # nilai sheet ini diprioritaskan
        merged.update(ser)
    merged = merged.sort_index()

    out = pd.DataFrame({
        "period": merged.index.strftime("%Y-%m-%d"),
        "Credit_Triliun_Rp": merged.values.round(6),
    })
    out.to_csv(OUT, index=False)
    print(f"\nTersimpan: {OUT.name} ({len(out)} bulan, "
          f"{out['period'].iloc[0]}..{out['period'].iloc[-1]})")
    print(f"  Terbaru = {out['Credit_Triliun_Rp'].iloc[-1]:,.1f} Triliun Rp")

    # Pertumbuhan kredit YoY (%) dari level bulanan — fitur stasioner untuk pemodelan
    yoy = (merged.pct_change(12) * 100).round(6)
    out_yoy = pd.DataFrame({
        "period": yoy.index.strftime("%Y-%m-%d"),
        "Credit_Growth_YoY_Pct": yoy.values,
    }).dropna(subset=["Credit_Growth_YoY_Pct"])
    out_yoy.to_csv(OUT_YOY, index=False)
    print(f"Tersimpan: {OUT_YOY.name} ({len(out_yoy)} bulan, "
          f"{out_yoy['period'].iloc[0]}..{out_yoy['period'].iloc[-1]})")


if __name__ == "__main__":
    main()
