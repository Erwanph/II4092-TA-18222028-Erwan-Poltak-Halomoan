"""Konsolidasi atribusi XAI: apa yang sebenarnya menggerakkan BI-Rate menurut
SELURUH model, bukan menurut satu model saja.

Latar belakang. Peringkat SHAP global pada Bab VI dihitung dari Random Forest,
dan pada model itu persistensi (lag BI-Rate) menguasai hampir seluruh atribusi.
Pembacaan sepintas karena itu mudah menyimpulkan bahwa satu-satunya penggerak
BI-Rate adalah nilai bulan sebelumnya. Kesimpulan itu tidak sahih: dominasi
persistensi merupakan sifat kelompok model berbasis pohon yang meramalkan level,
sedangkan LSTM dan BiLSTM meramalkan koreksi di sekitar jangkar $y_{t-1}$
sehingga persistensi sudah keluar dari matriks fitur sebelum SHAP bekerja.
Justru kedua model itulah yang penjelasannya dinilai paling baik oleh pakar.

Modul ini menyusun jawaban gabungan dalam dua lapis.

1. Pangsa persistensi per model (`xai_pangsa_persistensi.csv`) - menunjukkan
   secara kuantitatif bahwa dominasi lag hanya berlaku pada RF dan XGBoost.

2. Peringkat konsolidasi pendorong makroekonomi (`xai_konsolidasi.csv`) -
   persistensi dikeluarkan, kontribusi sisanya dinormalisasi menjadi pangsa,
   lalu peringkat dirata-ratakan atas 4 model x 2 bulan kontrol = 8 penilaian.
   Dilaporkan pula berapa kali sebuah fitur masuk tiga besar dari 8 penilaian
   tersebut, sebagai ukuran kekonsistenan yang tidak bergantung pada besaran.

Kedua bulan kontrol sama dengan bahan penilaian pakar: Juli 2025 (seluruh model
tepat) dan April 2024 (seluruh model meleset).

Jalankan: python -m src.analysis.xai_konsolidasi --write
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.evaluation_core import DISPLAY
from src.analysis.xai_consistency import kontribusi_semua, MODELS, PERSISTENSI

OUT_DIR = Path("results/experiments/analysis")
TEX_KONSOLIDASI = Path("../Tugas Akhir - Dokumen/tables/Tabel_Konsolidasi_XAI.tex")
TEX_PERSISTENSI = Path("../Tugas Akhir - Dokumen/tables/Tabel_Pangsa_Persistensi.tex")
TEX_PERBULAN = Path("../Tugas Akhir - Dokumen/tables/Tabel_XAI_Per_Bulan.tex")

BULAN = ["2025-Jul", "2024-Apr"]
TOPN = 3


def pangsa_persistensi(hasil_per_bulan):
    """Pangsa |SHAP| lag BI-Rate terhadap total |SHAP| tiap model tiap bulan."""
    rows = []
    for bulan, hasil in hasil_per_bulan.items():
        for model, contrib in hasil.items():
            tot = sum(abs(v) for v in contrib.values()) or 1.0
            lag = sum(abs(v) for k, v in contrib.items() if k in PERSISTENSI)
            rows.append({"bulan": bulan, "model": model,
                         "pangsa_persistensi": lag / tot * 100,
                         "pangsa_makro": (tot - lag) / tot * 100})
    return pd.DataFrame(rows)


def konsolidasi(hasil_per_bulan, topn=TOPN):
    """Peringkat rata-rata pendorong makro atas seluruh model x bulan."""
    semua = [c for h in hasil_per_bulan.values() for c in h.values()]
    bersama = sorted(set.intersection(*(set(c) for c in semua)) - PERSISTENSI)

    kolom_rank, kolom_pangsa, arah = {}, {}, {}
    for bulan, hasil in hasil_per_bulan.items():
        for model, contrib in hasil.items():
            v = np.array([abs(contrib[f]) for f in bersama], float)
            pangsa = v / (v.sum() or 1.0) * 100
            urut = (-pangsa).argsort()
            rank = np.empty(len(bersama), int)
            rank[urut] = np.arange(1, len(bersama) + 1)
            tag = f"{model}_{bulan}"
            kolom_rank[tag] = rank
            kolom_pangsa[tag] = pangsa
            arah[tag] = np.array([np.sign(contrib[f]) for f in bersama])

    df = pd.DataFrame({"fitur": bersama,
                       "fitur_tampil": [DISPLAY.get(f, f) for f in bersama]})
    R = pd.DataFrame(kolom_rank)
    P = pd.DataFrame(kolom_pangsa)
    A = pd.DataFrame(arah)

    df["rank_rata"] = R.mean(axis=1).values
    # Galat baku rata-rata peringkat. Diperlukan karena keempat model terbukti
    # tidak sepakat (Spearman tak bermakna), sehingga rata-rata peringkat
    # memiliki ketidakpastian yang sebanding dengan jarak antarperingkat.
    df["rank_se"] = (R.std(axis=1, ddof=1) / np.sqrt(R.shape[1])).values
    df["pangsa_rata"] = P.mean(axis=1).values
    df["n_top3"] = (R <= topn).sum(axis=1).values
    df["n_penilaian"] = R.shape[1]
    # arah dominan: berapa banyak penilaian yang menyatakan "menahan turun"
    df["n_menahan"] = (A < 0).sum(axis=1).values
    for tag in kolom_pangsa:
        df[f"pangsa_{tag}"] = kolom_pangsa[tag]
        df[f"rank_{tag}"] = kolom_rank[tag]
    return df.sort_values("rank_rata").reset_index(drop=True)


def _n(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",").replace("-", "$-$")


def write_persistensi_tex(df, path=TEX_PERSISTENSI):
    urut = ["RF", "XGBoost", "BiLSTM", "LSTM"]
    lines = [
        "% Auto-generated oleh src/analysis/xai_konsolidasi.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Pangsa kontribusi persistensi (lag BI-Rate) terhadap total "
        "kontribusi SHAP tiap model pada kedua bulan kontrol}",
        "\\label{tbl:pangsa-persistensi}",
        "\\begin{tabular}{|l|r|r|r|r|}", "\\hline",
        "\\multirow{2}{*}{\\textbf{Model}} & "
        "\\multicolumn{2}{c|}{\\textbf{Juli 2025}} & "
        "\\multicolumn{2}{c|}{\\textbf{April 2024}} \\\\ \\cline{2-5}",
        " & \\textbf{Persistensi} & \\textbf{Makro} & "
        "\\textbf{Persistensi} & \\textbf{Makro} \\\\ \\hline",
    ]
    for m in urut:
        sel = {r.bulan: r for _, r in df[df.model == m].iterrows()}
        j, a = sel["2025-Jul"], sel["2024-Apr"]
        lines.append(
            f"{m} & {_n(j.pangsa_persistensi)}\\% & {_n(j.pangsa_makro)}\\% & "
            f"{_n(a.pangsa_persistensi)}\\% & {_n(a.pangsa_makro)}\\% \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}",
              "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def write_konsolidasi_tex(df, path=TEX_KONSOLIDASI, topn=8):
    lines = [
        "% Auto-generated oleh src/analysis/xai_konsolidasi.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Peringkat konsolidasi pendorong makroekonomi menurut gabungan "
        "keempat model pada kedua bulan kontrol}",
        "\\label{tbl:konsolidasi-xai}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|c|l|c|r|r|c|}", "\\hline",
        "\\textbf{Peringkat} & \\textbf{Pendorong makroekonomi} & "
        "\\textbf{Masuk 3 besar} & \\textbf{Peringkat rata-rata} & "
        "\\textbf{Pangsa rata-rata} & \\textbf{Arah dominan} \\\\ \\hline",
    ]
    for i, r in df.head(topn).iterrows():
        arah = ("menahan turun" if r.n_menahan > r.n_penilaian / 2
                else "mendorong naik" if r.n_menahan < r.n_penilaian / 2
                else "campuran")
        se = f" $\\pm$ {_n(r.rank_se, 2)}" if "rank_se" in df.columns else ""
        lines.append(
            f"{i + 1} & {r.fitur_tampil} & "
            f"{int(r.n_top3)}/{int(r.n_penilaian)} & "
            f"{_n(r.rank_rata, 2)}{se} & "
            f"{_n(r.pangsa_rata)}\\% & {arah} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}",
              "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def write_perbulan_tex(df, path=TEX_PERBULAN, topn=TOPN):
    """Tiga pendorong makro teratas tiap model pada tiap bulan kontrol.

    Menjawab pertanyaan yang tidak terjawab tabel konsolidasi: bukan peringkat
    rata-rata, melainkan apa yang sebenarnya dikatakan tiap model pada bulan
    yang dinilai pakar. Dipakai untuk dicocokkan dengan hasil uji ablasi.
    """
    label = {"2025-Jul": "Juli 2025 (prediksi tepat)",
             "2024-Apr": "April 2024 (prediksi meleset)"}
    # Label bulan dipasang sebagai baris kepala yang membentang, bukan sebagai
    # kolom \multirow. Kolom label itu selebar sekitar 30 karakter dan terulang
    # pada tiap baris sehingga memaksa \resizebox mengecilkan font seluruh tabel
    # sampai sulit dibaca pada kertas A4.
    lines = [
        "% Auto-generated oleh src/analysis/xai_konsolidasi.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Tiga pendorong makroekonomi teratas menurut SHAP pada tiap "
        "model dan tiap bulan kontrol}",
        "\\label{tbl:xai-per-bulan}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|l|l|l|}", "\\hline",
        "\\textbf{Model} & \\textbf{Pendorong 1} & "
        "\\textbf{Pendorong 2} & \\textbf{Pendorong 3} \\\\ \\hline",
    ]
    for bulan in BULAN:
        models = [m for m in ["RF", "XGBoost", "BiLSTM", "LSTM"]
                  if f"rank_{m}_{bulan}" in df.columns]
        lines.append("\\multicolumn{4}{|l|}{\\textbf{"
                     + label.get(bulan, bulan) + "}} \\\\ \\hline")
        for m in models:
            sub = df.sort_values(f"rank_{m}_{bulan}").head(topn)
            sel = " & ".join(
                f"{r.fitur_tampil} ({_n(r[f'pangsa_{m}_{bulan}'])}\\%)"
                for _, r in sub.iterrows())
            lines.append(f"{m} & {sel} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}",
              "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def dari_csv(path=None):
    """Muat ulang hasil konsolidasi dari CSV yang sudah ada.

    Dipakai untuk meregenerasi tabel LaTeX tanpa menghitung ulang SHAP.
    Perhitungan ulang SHAP dihindari karena gambar dan tabel atribusi yang
    sudah dinilai pakar pada Lampiran E harus tetap persis sama, sedangkan
    estimator SHAP pada LSTM/BiLSTM bersifat stokastik.
    """
    p = Path(path or (OUT_DIR / "xai_konsolidasi.csv"))
    df = pd.read_csv(p)
    if "rank_se" not in df.columns:
        rc = [c for c in df.columns if c.startswith("rank_") and c != "rank_rata"]
        R = df[rc]
        df["rank_se"] = (R.std(axis=1, ddof=1) / np.sqrt(R.shape[1])).values
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--bulan", nargs="+", default=BULAN)
    ap.add_argument("--from-csv", action="store_true",
                    help="regenerasi tabel dari CSV lama, tanpa menghitung SHAP")
    args = ap.parse_args()

    if args.from_csv:
        kons = dari_csv()
        print("\n=== PERINGKAT KONSOLIDASI (dari CSV, tanpa hitung ulang SHAP) ===")
        print(kons[["fitur_tampil", "rank_rata", "rank_se", "pangsa_rata",
                    "n_top3", "n_penilaian"]].to_string(index=False))
        if args.write:
            kons.to_csv(OUT_DIR / "xai_konsolidasi.csv", index=False)
            write_konsolidasi_tex(kons)
            write_perbulan_tex(kons)
        return

    hasil_per_bulan = {}
    for b in args.bulan:
        print(f"\n=== KONTRIBUSI SHAP ({b}) ===")
        hasil_per_bulan[b] = kontribusi_semua(b, MODELS)

    pers = pangsa_persistensi(hasil_per_bulan)
    print("\n=== PANGSA PERSISTENSI PER MODEL ===")
    print(pers.to_string(index=False))

    kons = konsolidasi(hasil_per_bulan)
    print("\n=== PERINGKAT KONSOLIDASI PENDORONG MAKRO ===")
    print(kons[["fitur_tampil", "rank_rata", "pangsa_rata",
                "n_top3", "n_penilaian"]].to_string(index=False))

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pers.to_csv(OUT_DIR / "xai_pangsa_persistensi.csv", index=False)
        kons.to_csv(OUT_DIR / "xai_konsolidasi.csv", index=False)
        write_persistensi_tex(pers)
        write_konsolidasi_tex(kons)
        write_perbulan_tex(kons)
        print(f"\nCSV: {OUT_DIR}/xai_pangsa_persistensi.csv & xai_konsolidasi.csv")


if __name__ == "__main__":
    main()
