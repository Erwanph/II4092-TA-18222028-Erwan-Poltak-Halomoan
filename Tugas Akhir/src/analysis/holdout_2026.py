"""Kinerja keempat model kanonis pada holdout Januari--Mei 2026.

Latar belakang. Lima observasi terbaru (Januari--Mei 2026) tidak masuk ke
pembagian latih/validasi/uji. Berkas data/processed/dataset_2026.csv memuat
nilai BI-Rate aktualnya sehingga kinerja model kanonis pada bulan-bulan itu
tetap dapat dihitung sebagai pemeriksaan tambahan.

Perhitungan memakai model kanonis yang sama persis dengan seluruh angka Bab VI
(src.evaluation_core): Random Forest dan XGBoost dilatih ulang secara
deterministik, sedangkan LSTM dan BiLSTM memuat bobot ensembel tiga-seed yang
sudah dibekukan. Tidak ada model yang ditala ulang dan tidak ada konfigurasi
kanonis yang berubah.

Catatan penting. Holdout hanya berisi lima observasi dengan satu bulan-perubahan
(Mei 2026) sehingga seluruh metriknya bersifat indikatif, bukan penaksir
kinerja yang dapat diandalkan.

Hasilnya TIDAK masuk buku: lima observasi 2026 dinyatakan tidak masuk pembagian
latih/validasi/uji dan tidak dipakai pada evaluasi Bab VI. Modul ini disimpan
sebagai diagnostik untuk menjawab pertanyaan sidang secara lisan.

Jalankan: python -m src.analysis.holdout_2026 --write
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.evaluation_core import prep, train_predict, SPECS, MODEL_ORDER, STEP
from src.modeling.metrics import (
    calculate_rmse, calculate_mape, wilson_interval,
)

OUT_DIR = Path("results/experiments/analysis")
TEX = Path("../Tugas Akhir - Dokumen/tables/Tabel_Holdout_2026.tex")


def evaluasi():
    """Metrik holdout tiap model + rincian prediksi per bulan."""
    baris, rinci = [], []
    for name in MODEL_ORDER:
        d = prep(SPECS[name]["exp"])
        pred, pred_snap, _ = train_predict(name, d)
        m = d["split"] == "holdout"
        idx = np.where(m)[0]
        y = d["y"][idx]
        ps, pr = pred_snap[idx], pred[idx]
        ok = ~np.isnan(ps)
        if not ok.any():
            continue
        idx, y, ps, pr = idx[ok], y[ok], ps[ok], pr[ok]

        y_prev = d["y"][idx - 1]
        arah_asli = np.sign(np.round((y - y_prev) / STEP))
        arah_pred = np.sign(np.round((ps - y_prev) / STEP))
        benar = int((arah_asli == arah_pred).sum())
        lo, hi = wilson_interval(benar, len(y))

        baris.append({
            "model": name,
            "n": int(len(y)),
            "RMSE": calculate_rmse(y, pr),
            "MAPE": calculate_mape(y, pr),
            "hit_arah": benar / len(y),
            "hit_benar": benar,
            "wilson_lo": lo, "wilson_hi": hi,
            "n_ubah": int((arah_asli != 0).sum()),
            "tepat_persis": float(np.mean(np.abs(ps - y) < 1e-9)),
        })
        for k, i in enumerate(idx):
            rinci.append({"model": name, "bulan": d["dates"][i],
                          "aktual": y[k], "prediksi": pr[k],
                          "prediksi_snap": ps[k],
                          "galat": pr[k] - y[k]})
    return pd.DataFrame(baris), pd.DataFrame(rinci)


def _n(x, dec=4):
    return f"{x:.{dec}f}".replace(".", ",")


def write_tex(df, rinci, path=TEX):
    bulan = list(dict.fromkeys(rinci["bulan"]))
    lines = [
        "% Auto-generated oleh src/analysis/holdout_2026.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Kinerja model kanonis pada holdout Januari--Mei 2026 "
        "($n = 5$, indikatif)}",
        "\\label{tbl:holdout-2026}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|r|r|c|" + "c|" * len(bulan) + "}", "\\hline",
        "\\textbf{Model} & \\textbf{RMSE} & \\textbf{MAPE} & "
        "\\textbf{Arah benar} & "
        + " & ".join(f"\\textbf{{{b}}}" for b in bulan) + " \\\\ \\hline",
    ]
    aktual = {r.bulan: r.aktual for _, r in rinci.iterrows()}
    for _, r in df.iterrows():
        sub = rinci[rinci.model == r["model"]].set_index("bulan")
        sel = " & ".join(_n(sub.loc[b, "prediksi_snap"], 2) if b in sub.index
                         else "--" for b in bulan)
        lines.append(
            f"{r['model']} & {_n(r['RMSE'])} & {_n(r['MAPE'], 2)}\\% & "
            f"{int(r['hit_benar'])}/{int(r['n'])} & {sel} \\\\")
        lines.append("\\hline")
    lines.append("Aktual & -- & -- & -- & "
                 + " & ".join(_n(aktual[b], 2) for b in bulan) + " \\\\")
    lines += ["\\hline", "\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    df, rinci = evaluasi()
    print("\n=== METRIK HOLDOUT 2026 ===")
    print(df.to_string(index=False))
    print("\n=== PREDIKSI PER BULAN ===")
    print(rinci.to_string(index=False))

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "holdout_2026.csv", index=False)
        rinci.to_csv(OUT_DIR / "holdout_2026_rinci.csv", index=False)
        print(f"\nCSV: {OUT_DIR}/holdout_2026.csv")


if __name__ == "__main__":
    main()
