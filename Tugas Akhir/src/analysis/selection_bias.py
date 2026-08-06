"""Risiko seleksi: apa yang terjadi bila 384 kombinasi diperingkat menurut
RMSE VALIDASI, bukan RMSE UJI.

Latar belakang. Fase 1 (screening) memeringkat 384 kombinasi pra-pemrosesan
memakai rata-rata RMSE pada himpunan uji, dan peringkat itulah yang menyaring
kombinasi mana yang lolos ke Fase 3 dan Fase 4. Konsekuensinya himpunan uji
ikut menentukan konfigurasi yang akhirnya dilaporkan, sehingga RMSE uji
pemenang bukan lagi ukuran keluar-sampel yang murni.

Modul ini menjalankan ulang Fase 1 dengan metrik validasi ikut direkam
(kolom `RMSE_val` yang ditambahkan pada experiment_runner), lalu menjawab tiga
pertanyaan:

1. Bila 384 kombinasi diperingkat menurut RMSE validasi, di peringkat berapa
   E136 (kombinasi kanonis RF/BiLSTM) dan E112 (kombinasi kanonis XGBoost)?
2. Berapa RMSE uji kombinasi yang terbaik menurut validasi?
3. Seberapa besar selisih antara "juara menurut uji" dan "juara menurut
   validasi" bila keduanya diukur pada himpunan uji yang sama?

Keluaran ditulis ke results/experiments/analysis/ saja. Hasilnya TIDAK masuk
buku atas keputusan penulis dan berperan sebagai diagnostik untuk menjawab
pertanyaan sidang secara lisan.

CATATAN PENTING. Fase 1 dijalankan ulang ke direktori terpisah
(`results/experiments_seleksi`) supaya artefak kanonis
`results/experiments/screening/screening_results.csv` -- yang menjadi sumber
angka Fase 2 di Bab VI -- tidak tertimpa.

Jalankan: python -m src.analysis.selection_bias --run --write
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = Path("results/experiments/analysis")
RUN_DIR = Path("results/experiments_seleksi")
TEX = Path("../Tugas Akhir - Dokumen/tables/Tabel_Risiko_Seleksi.tex")

# Kombinasi kanonis yang dipakai model final (lihat best_results_registry.json).
KANONIS = {"E136": "RF, BiLSTM", "E112": "XGBoost"}


def run_screening(run_dir=RUN_DIR):
    """Jalankan ulang Fase 1 (384 kombinasi x RF & XGBoost) ke direktori terpisah."""
    from src.experiments.experiment_runner import FactorialExperimentRunner

    runner = FactorialExperimentRunner(output_dir=str(run_dir))
    runner.run_screening()
    return run_dir / "screening" / "screening_results.csv"


def peringkat(df):
    """Peringkat kombinasi menurut rata-rata RMSE uji dan RMSE validasi."""
    agg = (df.groupby("experiment_id")
             .agg(RMSE_uji=("RMSE", "mean"),
                  RMSE_val=("RMSE_val", "mean"))
             .reset_index())
    agg["rank_uji"] = agg["RMSE_uji"].rank(method="min").astype(int)
    agg["rank_val"] = agg["RMSE_val"].rank(method="min").astype(int)
    return agg.sort_values("rank_uji").reset_index(drop=True)


def ringkas(agg):
    """Rakit angka ringkas untuk keperluan diagnostik."""
    n = len(agg)
    juara_uji = agg.loc[agg["rank_uji"].idxmin()]
    juara_val = agg.loc[agg["rank_val"].idxmin()]

    out = {
        "n_kombinasi": int(n),
        "korelasi_spearman_rank": float(
            agg["RMSE_uji"].corr(agg["RMSE_val"], method="spearman")),
        "juara_uji": {
            "experiment_id": juara_uji["experiment_id"],
            "RMSE_uji": float(juara_uji["RMSE_uji"]),
            "RMSE_val": float(juara_uji["RMSE_val"]),
            "rank_val": int(juara_uji["rank_val"]),
        },
        "juara_val": {
            "experiment_id": juara_val["experiment_id"],
            "RMSE_uji": float(juara_val["RMSE_uji"]),
            "RMSE_val": float(juara_val["RMSE_val"]),
            "rank_uji": int(juara_val["rank_uji"]),
        },
        "kanonis": {},
    }
    for eid in KANONIS:
        sel = agg[agg["experiment_id"] == eid]
        if sel.empty:
            continue
        r = sel.iloc[0]
        out["kanonis"][eid] = {
            "model": KANONIS[eid],
            "RMSE_uji": float(r["RMSE_uji"]),
            "RMSE_val": float(r["RMSE_val"]),
            "rank_uji": int(r["rank_uji"]),
            "rank_val": int(r["rank_val"]),
        }
    # selisih RMSE uji antara juara menurut uji dan juara menurut validasi:
    # perkiraan kasar besar keuntungan yang diperoleh dari memilih di atas uji
    out["selisih_rmse_uji_juara"] = float(
        juara_val["RMSE_uji"] - juara_uji["RMSE_uji"])
    return out


def _n(x, dec=4):
    return f"{x:.{dec}f}".replace(".", ",")


def write_tex(agg, rk, path=TEX, topn=5):
    """Tabel: lima kombinasi teratas menurut tiap kriteria + posisi kombinasi kanonis."""
    lines = [
        "% Auto-generated oleh src/analysis/selection_bias.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Perbandingan pemeringkatan 384 kombinasi pra-pemrosesan "
        "menurut RMSE uji dan menurut RMSE validasi}",
        "\\label{tbl:risiko-seleksi}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|c|r|r|c|c|}", "\\hline",
        "\\textbf{Kelompok} & \\textbf{Kombinasi} & \\textbf{RMSE uji} & "
        "\\textbf{RMSE validasi} & \\textbf{Peringkat menurut uji} & "
        "\\textbf{Peringkat menurut validasi} \\\\ \\hline",
    ]

    top_uji = agg.nsmallest(topn, "RMSE_uji")
    for k, (_, r) in enumerate(top_uji.iterrows()):
        label = (f"\\multirow{{{topn}}}{{*}}{{Lima teratas menurut uji}}"
                 if k == 0 else "")
        lines.append(
            f"{label} & {r.experiment_id} & {_n(r.RMSE_uji)} & {_n(r.RMSE_val)} & "
            f"{int(r.rank_uji)} & {int(r.rank_val)} \\\\")
    lines.append("\\hline")

    top_val = agg.nsmallest(topn, "RMSE_val")
    for k, (_, r) in enumerate(top_val.iterrows()):
        label = (f"\\multirow{{{topn}}}{{*}}{{Lima teratas menurut validasi}}"
                 if k == 0 else "")
        lines.append(
            f"{label} & {r.experiment_id} & {_n(r.RMSE_uji)} & {_n(r.RMSE_val)} & "
            f"{int(r.rank_uji)} & {int(r.rank_val)} \\\\")
    lines.append("\\hline")

    kan = agg[agg["experiment_id"].isin(KANONIS)].sort_values("experiment_id")
    for k, (_, r) in enumerate(kan.iterrows()):
        label = (f"\\multirow{{{len(kan)}}}{{*}}{{Kombinasi kanonis}}"
                 if k == 0 else "")
        lines.append(
            f"{label} & {r.experiment_id} ({KANONIS[r.experiment_id]}) & "
            f"{_n(r.RMSE_uji)} & {_n(r.RMSE_val)} & "
            f"{int(r.rank_uji)} & {int(r.rank_val)} \\\\")
    lines += ["\\hline", "\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true",
                    help="jalankan ulang Fase 1 ke direktori terpisah")
    ap.add_argument("--write", action="store_true", help="tulis CSV + tabel LaTeX")
    args = ap.parse_args()

    csv_path = RUN_DIR / "screening" / "screening_results.csv"
    if args.run or not csv_path.exists():
        csv_path = run_screening()

    df = pd.read_csv(csv_path)
    df = df[np.isfinite(df["RMSE"]) & np.isfinite(df["RMSE_val"])]
    agg = peringkat(df)
    rk = ringkas(agg)

    print("\n=== RINGKASAN RISIKO SELEKSI ===")
    print(json.dumps(rk, indent=2, ensure_ascii=False))
    print("\n=== SEPULUH TERATAS MENURUT VALIDASI ===")
    print(agg.nsmallest(10, "RMSE_val").to_string(index=False))

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        agg.to_csv(OUT_DIR / "selection_bias_ranking.csv", index=False)
        (OUT_DIR / "selection_bias_summary.json").write_text(
            json.dumps(rk, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nCSV: {OUT_DIR}/selection_bias_ranking.csv")


if __name__ == "__main__":
    main()
