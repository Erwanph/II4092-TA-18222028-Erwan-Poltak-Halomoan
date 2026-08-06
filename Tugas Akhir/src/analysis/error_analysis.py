"""Analisis galat: sebaran per tahun, per jenis bulan, dan studi kasus terburuk.

Bab VI banyak melaporkan angka ringkas tetapi belum memaknai di mana dan kapan
model gagal. Modul ini membedah galat pada himpunan uji dari artefak prediksi
kanonis (`pred_vs_actual_full.csv`), sehingga tidak diperlukan pelatihan ulang dan
angkanya konsisten dengan registry.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRED_PATH = Path("results/experiments/analysis/pred_vs_actual_full.csv")
OUT_DIR = Path("results/experiments/analysis")
TEX_TAHUN = Path("../Tugas Akhir - Dokumen/tables/Tabel_Galat_Per_Tahun.tex")
TEX_TERBURUK = Path("../Tugas Akhir - Dokumen/tables/Tabel_Prediksi_Terburuk.tex")

MODELS = ["RF", "BiLSTM", "LSTM", "XGBoost"]
STEP = 0.25


def load():
    df = pd.read_csv(PRED_PATH)
    df["aktual_prev"] = df["aktual"].shift(1)
    df["tahun"] = df["tanggal"].str.split("-").str[0].astype(int)
    return df


def per_tahun(df):
    """RMSE tiap model per tahun pada himpunan uji, berikut galat ramalan naif."""
    te = df[df["split"] == "test"].copy()
    rows = []
    for tahun, g in te.groupby("tahun"):
        y = g["aktual"].to_numpy(float)
        e_rw = y - g["aktual_prev"].to_numpy(float)
        n_ubah = int((np.round(np.abs(e_rw) / STEP) != 0).sum())
        rec = {"tahun": int(tahun), "n_bulan": len(g), "n_bulan_ubah": n_ubah,
               "RMSE_naif": float(np.sqrt(np.mean(e_rw ** 2)))}
        for m in MODELS:
            e = y - g[f"pred_{m}"].to_numpy(float)
            rec[f"RMSE_{m}"] = float(np.sqrt(np.mean(e ** 2)))
        rows.append(rec)
    return pd.DataFrame(rows)


def terburuk(df, topn=5):
    """Bulan dengan galat terbesar, dirata-ratakan lintas keempat model."""
    te = df[df["split"] == "test"].copy()
    err = np.stack([np.abs(te["aktual"] - te[f"pred_{m}"]) for m in MODELS])
    te["galat_rata"] = err.mean(axis=0)
    te["delta_aktual_grid"] = (np.round((te["aktual"] - te["aktual_prev"]) / STEP)
                               * STEP)
    cols = (["tanggal", "aktual", "aktual_prev", "delta_aktual_grid", "galat_rata"]
            + [f"pred_{m}" for m in MODELS])
    return te.nlargest(topn, "galat_rata")[cols].reset_index(drop=True)


def _n(x, dec=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{dec}f}".replace(".", ",").replace("-", "$-$")


def write_tahun_tex(df, path=TEX_TAHUN):
    lines = [
        "% Auto-generated oleh src/analysis/error_analysis.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Sebaran RMSE per tahun pada himpunan uji}",
        "\\label{tbl:galat-per-tahun}",
        "\\begin{tabular}{|c|c|c|r|r|r|r|}", "\\hline",
        "\\textbf{Tahun} & \\textbf{Bulan} & \\textbf{Bulan ubah} & "
        "\\textbf{RF} & \\textbf{BiLSTM} & \\textbf{LSTM} & "
        "\\textbf{XGBoost} \\\\ \\hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{int(r.tahun)} & {int(r.n_bulan)} & {int(r.n_bulan_ubah)} & "
            f"{_n(r.RMSE_RF)} & {_n(r.RMSE_BiLSTM)} & "
            f"{_n(r.RMSE_LSTM)} & {_n(r.RMSE_XGBoost)} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def write_terburuk_tex(df, path=TEX_TERBURUK):
    lines = [
        "% Auto-generated oleh src/analysis/error_analysis.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Lima bulan dengan galat prediksi terbesar pada himpunan uji}",
        "\\label{tbl:prediksi-terburuk}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|c|c|c|r|r|r|r|}", "\\hline",
        "\\textbf{Bulan} & \\textbf{Aktual} & \\textbf{Bulan lalu} & "
        "\\textbf{Perubahan} & \\textbf{RF} & \\textbf{BiLSTM} & \\textbf{LSTM} & "
        "\\textbf{XGBoost} \\\\ \\hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r.tanggal} & {_n(r.aktual, 2)} & {_n(r.aktual_prev, 2)} & "
            f"{_n(r.delta_aktual_grid, 2)} & {_n(r.pred_RF, 3)} & "
            f"{_n(r.pred_BiLSTM, 3)} & {_n(r.pred_LSTM, 3)} & "
            f"{_n(r.pred_XGBoost, 3)} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    df = load()

    print("\n=== RMSE PER TAHUN (HIMPUNAN UJI) ===")
    tah = per_tahun(df)
    print(tah.to_string(index=False))

    print("\n=== LIMA BULAN DENGAN GALAT TERBESAR ===")
    wor = terburuk(df)
    print(wor.to_string(index=False))

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tah.to_csv(OUT_DIR / "error_per_year.csv", index=False)
        wor.to_csv(OUT_DIR / "worst_predictions.csv", index=False)
        write_tahun_tex(tah)
        write_terburuk_tex(wor)


if __name__ == "__main__":
    main()
