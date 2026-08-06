"""Korelasi Pearson tiap variabel makroekonomi terhadap BI-Rate.

Latar belakang. Tabel korelasi pada Bab III semula ditulis tangan dari
`dataset_raw_integrated.csv` (251 baris, mencakup lima bulan holdout 2026) dan
memakai kolom `Credit_Triliun_Rp`, yaitu POSISI kredit. Kolom itu tidak pernah
masuk ke matriks fitur model: yang dipakai seluruh eksperimen adalah
`Credit_Growth_YoY_Pct`, yaitu PERTUMBUHAN kredit. Karena tabel itu tidak punya
modul pembangkit, angkanya tidak pernah ikut diperbarui ketika dataset final
disusun.

Modul ini menghitung ulang korelasi dari dataset yang benar-benar dipakai
model (`dataset_final.csv`, 246 observasi historis Juli 2005--Desember 2025)
lalu menulis:

  - tables/Tabel_Korelasi.tex          (tabel Bab III)
  - results/.../correlation_target.csv (artefak angka)
  - images/07_correlation_heatmap.png  (matriks korelasi antarvariabel)
  - images/08_correlation_with_target.png (diagram batang korelasi target)

Kedua gambar diregenerasi dari sumber yang sama dengan tabel supaya tidak ada
lagi versi angka yang berbeda antara tabel dan gambar.

Catatan. Korelasi dihitung berpasangan (`pairwise`), jadi kolom dengan data
hilang memakai observasi yang tersedia saja. Jumlahnya dilaporkan pada kolom
`n` supaya pembaca tahu Pertumbuhan PDB hanya bersandar pada 180 observasi.

Jalankan: python -m src.analysis.correlation_table --write
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.modeling.config import DATASET_PATH, TARGET_COLUMN
from src.evaluation_core import DISPLAY

OUT_DIR = Path("results/experiments/analysis")
DOC_DIR = Path("../Tugas Akhir - Dokumen")
TEX = DOC_DIR / "tables/Tabel_Korelasi.tex"
IMG_DIR = DOC_DIR / "images"

# Ambang kekuatan korelasi yang dipakai Bab III.
AMBANG = [(0.80, "Sangat kuat"), (0.60, "Kuat"), (0.30, "Sedang"),
          (0.10, "Lemah"), (0.02, "Sangat lemah")]


def kekuatan(r):
    """Label kekuatan + arah korelasi, mis. 'Kuat ($-$)'."""
    a = abs(r)
    for batas, label in AMBANG:
        if a >= batas:
            return f"{label} ({'+' if r > 0 else '$-$'})"
    return "Tidak ada"


def hitung(path=DATASET_PATH, target=TARGET_COLUMN):
    """Korelasi Pearson tiap fitur terhadap target, terurut menurun."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("ID", "Bulan", "Tahun") if c in df.columns])
    num = df.select_dtypes(include=[np.number])

    r = num.corr(numeric_only=True)[target].drop(target)
    n = num.notna().mul(num[target].notna(), axis=0).sum().drop(target)

    out = pd.DataFrame({
        "kolom": r.index,
        "variabel": [DISPLAY.get(c, c) for c in r.index],
        "n": n.reindex(r.index).astype(int).values,
        "korelasi": r.values,
    })
    out["kekuatan"] = out["korelasi"].map(kekuatan)
    return out.sort_values("korelasi", ascending=False).reset_index(drop=True)


def _n(x, dec=3):
    s = f"{x:.{dec}f}".replace(".", ",")
    return s.replace("-", "$-$")


def write_tex(df, path=TEX):
    lines = [
        "% Auto-generated oleh src/analysis/correlation_table.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Korelasi Pearson tiap variabel terhadap BI-Rate}",
        "\\label{tbl:korelasi-birate}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|l|r|r|l|}", "\\hline",
        "\\textbf{Variabel} & \\textbf{Kolom dataset} & \\textbf{n} & "
        "\\textbf{Korelasi} & \\textbf{Kekuatan} \\\\", "\\hline",
    ]
    for _, r in df.iterrows():
        kolom = r["kolom"].replace("_", "\\_")
        lines.append(f"{r['variabel']} & {kolom} & {int(r['n'])} & "
                     f"{_n(r['korelasi'])} & {r['kekuatan']} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def write_figures(path=DATASET_PATH, target=TARGET_COLUMN, img_dir=IMG_DIR):
    """Regenerasi Gambar heatmap korelasi dan diagram batang korelasi target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({"font.size": 12, "figure.dpi": 200, "savefig.dpi": 200})
    sns.set_style("whitegrid")

    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("ID", "Bulan", "Tahun") if c in df.columns])
    num = df.select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)
    label = [DISPLAY.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="YlOrRd", vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=0.8,
                linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "Korelasi Pearson"},
                ax=ax, annot_kws={"size": 9, "fontweight": "bold"})
    ax.set_xticklabels(label, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(label, rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(img_dir / "07_correlation_heatmap.png",
                bbox_inches="tight", pad_inches=0.3)
    plt.close()

    tc = corr[target].drop(target).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in tc.values]
    bars = ax.barh([DISPLAY.get(c, c) for c in tc.index], tc.values,
                   color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Korelasi Pearson")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=0.7, color="orange", linestyle="--", alpha=0.5,
               label="Ambang kuat (|r| = 0,7)")
    ax.axvline(x=-0.7, color="orange", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, tc.values):
        ax.text(val + 0.01 if val >= 0 else val - 0.06,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}".replace(".", ","), va="center",
                fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(img_dir / "08_correlation_with_target.png",
                bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"Gambar: {img_dir}/07_correlation_heatmap.png & "
          f"08_correlation_with_target.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    df = hitung()
    print("\n=== KORELASI PEARSON TERHADAP BI-RATE ===")
    print(df.to_string(index=False))

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "correlation_target.csv", index=False)
        write_tex(df)
        if not args.no_figures:
            write_figures()
        print(f"\nCSV: {OUT_DIR}/correlation_target.csv")


if __name__ == "__main__":
    main()
