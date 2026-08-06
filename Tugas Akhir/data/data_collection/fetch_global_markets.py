"""Fetch variabel pasar global (FRED + Yahoo Finance); hanya menimpa
baris >= 2026 agar data historis tidak berubah.
"""

import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

RAW = Path(__file__).resolve().parents[1] / "raw" / "external"

# Batas refresh: HANYA bulan mulai tanggal ini yang di-fetch & ditimpa.
# Periode sebelum ini (historis s.d. 2025) tidak pernah disentuh.
REFRESH_FROM = pd.Timestamp("2026-01-01")


def _end_of_month(dt: pd.Timestamp) -> pd.Timestamp:
    return dt + pd.offsets.MonthEnd(0)


def _target_months() -> pd.DatetimeIndex:
    """Bulan akhir-bulan dari REFRESH_FROM hingga bulan lalu (yang sudah lengkap)."""
    last_complete = _end_of_month(
        pd.Timestamp.now().replace(day=1) - pd.DateOffset(months=1)
    )
    return pd.date_range(start=_end_of_month(REFRESH_FROM), end=last_complete, freq="ME")


def _query_start(months: pd.DatetimeIndex) -> str:
    """Tanggal awal query = hari pertama bulan target paling awal (agar bulan itu utuh)."""
    return months.min().replace(day=1).strftime("%Y-%m-%d")


def _merge_and_save(path: Path, valcol: str, fetched: pd.Series) -> int:
    """Gabungkan nilai hasil fetch (>= REFRESH_FROM) ke CSV, sisakan baris lama apa adanya."""
    df = pd.read_csv(path, dtype=str)
    df["period"] = pd.to_datetime(df["period"])

    # Pertahankan SEMUA baris historis < REFRESH_FROM tanpa perubahan.
    kept = df[df["period"] < REFRESH_FROM].copy()

    new_rows = pd.DataFrame({"period": fetched.index, valcol: fetched.values})
    combined = pd.concat([kept, new_rows], ignore_index=True).sort_values("period")
    combined["period"] = combined["period"].dt.strftime("%Y-%m-%d")
    combined[["period", valcol]].to_csv(path, index=False)
    return len(new_rows)


def _fred_daily(series_id: str, start: str, end: str, retries: int = 4) -> pd.Series:
    """Ambil seri harian dari endpoint CSV FRED (lebih andal untuk payload besar)."""
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={series_id}&cosd={start}&coed={end}")
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = ["date", "val"]
            d["date"] = pd.to_datetime(d["date"])
            d["val"] = pd.to_numeric(d["val"], errors="coerce")
            return d.dropna().set_index("date")["val"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    retry {attempt + 1}/{retries} ({type(e).__name__})")
            time.sleep(5)
    raise RuntimeError(f"FRED gagal untuk {series_id}: {last_err}")


def fetch_fred(series_id: str, valcol: str, path: Path) -> None:
    months = _target_months()
    if months.empty:
        print(f"  {path.name}: tidak ada bulan target")
        return
    start = _query_start(months)
    end = months.max().strftime("%Y-%m-%d")
    daily = _fred_daily(series_id, start, end)

    monthly = daily.resample("ME").mean().round(6)
    monthly.index = monthly.index.map(_end_of_month)
    monthly = monthly[monthly.index.isin(months)].dropna()

    n = _merge_and_save(path, valcol, monthly)
    print(f"  {path.name}: {n} baris >= {REFRESH_FROM.year} (FRED:{series_id})")


def fetch_yahoo(ticker: str, valcol: str, path: Path, how: str) -> None:
    """how='mean' (rata-rata bulanan) atau 'eom' (penutupan akhir bulan)."""
    months = _target_months()
    if months.empty:
        print(f"  {path.name}: tidak ada bulan target")
        return
    start = _query_start(months)
    end = (months.max() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        print(f"  {path.name}: tidak ada data dari Yahoo ({ticker})")
        return

    raw.index = pd.to_datetime(raw.index)
    close = raw["Close"]
    if hasattr(close, "columns"):  # MultiIndex (yfinance versi baru)
        close = close.iloc[:, 0]
    monthly = (close.resample("ME").mean() if how == "mean"
               else close.resample("ME").last()).round(6)
    monthly.index = monthly.index.map(_end_of_month)
    monthly = monthly[monthly.index.isin(months)].dropna()

    n = _merge_and_save(path, valcol, monthly)
    tag = "rata-rata" if how == "mean" else "EOM close"
    print(f"  {path.name}: {n} baris >= {REFRESH_FROM.year} (Yahoo:{ticker}, {tag})")


def main():
    print(f"Fetching variabel pasar global (hanya >= {REFRESH_FROM.date()})...")

    fetch_fred("FEDFUNDS",     "Federal_Funds_Rate_Pct",     RAW / "Federal_Funds_Rate_Pct.csv")
    fetch_fred("DCOILBRENTEU", "Oil_Price_Brent_USD_per_Bbl", RAW / "Oil_Price_Brent_USD_per_Bbl.csv")
    fetch_yahoo("^VIX",     "VIX_Volatility_Index",  RAW / "VIX_Volatility_Index.csv", how="mean")
    fetch_yahoo("GC=F",     "Gold_Price_USD_per_Oz", RAW / "Gold_Price_USD_per_Oz.csv", how="mean")
    fetch_yahoo("USDIDR=X", "USD_IDR_Monthly_Avg",   RAW / "USD_IDR_Monthly_Avg.csv", how="mean")
    fetch_yahoo("^JKSE",    "IHSG_End_of_Month",     RAW / "IHSG_End_of_Month.csv", how="eom")

    print("\nSelesai. Data historis s.d. 2025 tidak disentuh.")


if __name__ == "__main__":
    main()
