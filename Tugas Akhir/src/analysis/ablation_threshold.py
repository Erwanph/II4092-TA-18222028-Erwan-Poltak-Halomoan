"""Ambang derau yang benar untuk membaca hasil ablasi leave-one-out.

Latar belakang. Bab VI semula memakai simpangan baku antarpelatihan
($\\sigma \\approx$ 0,012--0,015) langsung sebagai ambang untuk menyatakan sebuah
efek ablasi ``terbaca''. Ambang itu terlalu longgar. $\\Delta$RMSE bukan hasil
satu pelatihan, melainkan SELISIH dua pelatihan yang sama-sama berderau
(konfigurasi penuh dikurangi konfigurasi tanpa satu fitur), sehingga simpangan
bakunya

    sigma_delta = sqrt(sigma^2 + sigma^2) = sqrt(2) * sigma

yaitu sekitar 1,41 kali lebih besar daripada ambang yang dipakai semula.

Selain itu, uji dilakukan berulang kali: 17 fitur x 4 model = 68 perbandingan.
Pada 68 perbandingan, beberapa selisih sebesar 2 sigma diharapkan muncul semata
karena kebetulan sehingga ambang per-perbandingan tidak boleh dibaca sebagai
tingkat kepercayaan menyeluruh.

Modul ini memakai simpangan baku antarpelatihan yang DIUKUR, bukan yang
dikira-kira: nilainya diambil dari results/experiments/analysis/seed_stability.csv,
yaitu simpangan baku RMSE uji atas 120 ensembel tiga-seed pada konfigurasi
kanonis (L2 = 0). Untuk Random Forest dan XGBoost yang deterministik,
sigma = 0 sehingga selisih sekecil apa pun tetap dapat direproduksi.

Jalankan: python -m src.analysis.ablation_threshold --write
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("results/experiments/analysis")
ABLASI = OUT_DIR / "ablation_per_feature.csv"
SEED = OUT_DIR / "seed_stability.csv"
TEX = Path("../Tugas Akhir - Dokumen/tables/Tabel_Ambang_Ablasi.tex")

PENUH = "(konfigurasi penuh)"
Z = 2.0          # kriteria "terbaca": |delta| >= 2 * sigma_delta


def sigma_per_model(seed_path=SEED):
    """Simpangan baku antarpelatihan tiap model pada konfigurasi kanonis."""
    sig = {"RF": 0.0, "XGBoost": 0.0}      # deterministik
    if seed_path.exists():
        s = pd.read_csv(seed_path)
        s = s[s["l2"] == 0.0]
        for _, r in s.iterrows():
            sig[r["model"]] = float(r["rmse_ens_std"])
    return sig


def klasifikasi(ablasi_path=ABLASI, seed_path=SEED, z=Z):
    """Tandai tiap efek ablasi: terbaca atau di dalam derau."""
    df = pd.read_csv(ablasi_path)
    df = df[df["fitur"] != PENUH].copy()
    sig = sigma_per_model(seed_path)

    df["sigma"] = df["model"].map(sig)
    df["sigma_delta"] = df["sigma"] * np.sqrt(2)
    df["ambang"] = df["sigma_delta"] * z
    df["rasio_sigma"] = np.where(df["sigma_delta"] > 0,
                                 df["delta_RMSE"].abs() / df["sigma_delta"],
                                 np.inf)
    df["terbaca"] = df["delta_RMSE"].abs() >= df["ambang"]
    return df.reset_index(drop=True)


def ringkas(df):
    """Berapa efek yang bertahan dengan ambang yang benar, per model."""
    g = (df.groupby("model")
           .agg(n_fitur=("fitur", "count"),
                sigma=("sigma", "first"),
                ambang=("ambang", "first"),
                n_terbaca=("terbaca", "sum"))
           .reset_index())
    g["n_perbandingan_total"] = len(df)
    return g


def _n(x, dec=4):
    return f"{x:.{dec}f}".replace(".", ",")


def write_tex(df, path=TEX):
    """Tabel: efek ablasi yang tetap terbaca setelah ambang dikoreksi."""
    urut = ["RF", "XGBoost", "BiLSTM", "LSTM"]
    lines = [
        "% Auto-generated oleh src/analysis/ablation_threshold.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Efek ablasi yang tetap terbaca setelah ambang derau "
        "dikoreksi menjadi $2\\sqrt{2}\\sigma$}",
        "\\label{tbl:ambang-ablasi}",
        "\\begin{tabular}{|l|r|l|r|r|}", "\\hline",
        "\\textbf{Model} & \\textbf{Ambang} & \\textbf{Fitur} & "
        "\\textbf{$\\Delta$RMSE} & \\textbf{Kelipatan $\\sigma_\\Delta$} \\\\",
        "\\hline",
    ]
    for m in urut:
        sub = df[(df.model == m) & df.terbaca].sort_values(
            "delta_RMSE", ascending=False)
        amb = df[df.model == m]["ambang"].iloc[0]
        amb_s = _n(amb) if amb > 0 else "deterministik"
        if sub.empty:
            lines.append(f"{m} & {amb_s} & (tidak ada) & -- & -- \\\\")
            lines.append("\\hline")
            continue
        for k, (_, r) in enumerate(sub.iterrows()):
            nama = f"\\multirow{{{len(sub)}}}{{*}}{{{m}}}" if k == 0 else ""
            ambc = f"\\multirow{{{len(sub)}}}{{*}}{{{amb_s}}}" if k == 0 else ""
            rasio = ("--" if not np.isfinite(r.rasio_sigma)
                     else _n(r.rasio_sigma, 1))
            lines.append(f"{nama} & {ambc} & {r.fitur_tampil} & "
                         f"{_n(r.delta_RMSE)} & {rasio} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    df = klasifikasi()
    print("\n=== AMBANG PER MODEL ===")
    print(ringkas(df).to_string(index=False))
    print("\n=== EFEK YANG TETAP TERBACA (model berderau saja) ===")
    dl = df[(df.sigma > 0)]
    print(dl[dl.terbaca][["model", "fitur_tampil", "delta_RMSE", "ambang",
                          "rasio_sigma"]].to_string(index=False))
    print("\n=== EFEK YANG GUGUR SETELAH KOREKSI (|delta| >= sigma lama 0,015 "
          "tetapi < ambang baru) ===")
    gugur = dl[(dl.delta_RMSE.abs() >= 0.015) & (~dl.terbaca)]
    print(gugur[["model", "fitur_tampil", "delta_RMSE", "ambang",
                 "rasio_sigma"]].to_string(index=False))

    if args.write:
        df.to_csv(OUT_DIR / "ablation_threshold.csv", index=False)
        write_tex(df)
        print(f"\nCSV: {OUT_DIR}/ablation_threshold.csv")


if __name__ == "__main__":
    main()
