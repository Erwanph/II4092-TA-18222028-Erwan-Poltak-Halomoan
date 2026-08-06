"""Selang kepercayaan Wilson untuk hit-rate arah pada himpunan uji.

Latar belakang. Klaim pembeda utama tugas akhir ini bersandar pada akurasi
arah di bulan-perubahan: BiLSTM dan LSTM benar pada 6 dari 9 bulan (67%),
sedangkan Random Forest hanya 3 dari 9 (33%). Sembilan observasi merupakan
dasar yang sangat tipis, dan tanpa ukuran ketidakpastian selisih 67% berbanding
33% mudah dibaca lebih tegas daripada yang sebenarnya didukung data.

Modul ini menghitung selang kepercayaan Wilson 95% untuk seluruh hit-rate pada
Tabel akurasi arah, yaitu bulan-perubahan (n = 9), bulan-tahan (n = 27), dan
keseluruhan (n = 36). Selang Wilson dipilih karena tetap terkurung di dalam
[0, 1] pada n kecil, tidak seperti selang normal biasa.

Sumber angka: results/experiments/analysis/naive_benchmark.csv, yaitu artefak
yang sama dengan yang dipakai Tabel akurasi arah. Modul ini tidak melatih ulang
model apa pun.

Jalankan: python -m src.analysis.directional_ci --write
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.metrics import wilson_interval

SRC = Path("results/experiments/analysis/naive_benchmark.csv")
OUT_DIR = Path("results/experiments/analysis")
TEX = Path("../Tugas Akhir - Dokumen/tables/Tabel_Selang_Wilson.tex")

MODEL_LABEL = {"RF": "Random Forest", "BiLSTM": "BiLSTM",
               "LSTM": "LSTM", "XGBoost": "XGBoost"}
URUT = ["RF", "BiLSTM", "LSTM", "XGBoost"]
N_TOTAL, N_CHANGE = 36, 9


def hitung(src=SRC):
    """Tabel k/n dan selang Wilson tiap model untuk ketiga segmen."""
    df = pd.read_csv(src).set_index("model")
    rows = []
    for m in URUT:
        r = df.loc[m]
        k_all = int(round(r["hit_rate_all"] * N_TOTAL))
        k_chg = int(round(r["hit_rate_change"] * N_CHANGE))
        k_hold = k_all - k_chg
        n_hold = N_TOTAL - N_CHANGE
        for segmen, k, n in (("Bulan-perubahan", k_chg, N_CHANGE),
                             ("Bulan-tahan", k_hold, n_hold),
                             ("Keseluruhan", k_all, N_TOTAL)):
            lo, hi = wilson_interval(k, n)
            rows.append({"model": m, "label": MODEL_LABEL[m], "segmen": segmen,
                         "k": k, "n": n, "hit_rate": k / n,
                         "wilson_lo": lo, "wilson_hi": hi,
                         "lebar": hi - lo})
    return pd.DataFrame(rows)


def tumpang_tindih(df, segmen="Bulan-perubahan"):
    """Pasangan model yang selang Wilson-nya bertumpang tindih pada satu segmen."""
    sub = df[df.segmen == segmen].set_index("model")
    out = []
    for a, b in [(x, y) for i, x in enumerate(URUT) for y in URUT[i + 1:]]:
        ra, rb = sub.loc[a], sub.loc[b]
        overlap = (ra.wilson_lo <= rb.wilson_hi) and (rb.wilson_lo <= ra.wilson_hi)
        out.append({"a": a, "b": b, "tumpang_tindih": bool(overlap)})
    return pd.DataFrame(out)


def _p(x):
    return f"{x * 100:.0f}\\%"


def write_tex(df, path=TEX):
    lines = [
        "% Auto-generated oleh src/analysis/directional_ci.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Selang kepercayaan Wilson 95\\% untuk hit-rate arah pada "
        "himpunan uji}",
        "\\label{tbl:selang-wilson}",
        "\\begin{tabular}{|l|l|c|c|c|}", "\\hline",
        "\\textbf{Model} & \\textbf{Segmen} & \\textbf{Tepat/Total} & "
        "\\textbf{Hit-rate} & \\textbf{Selang Wilson 95\\%} \\\\", "\\hline",
    ]
    for m in URUT:
        sub = df[df.model == m]
        for k, (_, r) in enumerate(sub.iterrows()):
            nama = f"\\multirow{{3}}{{*}}{{{r['label']}}}" if k == 0 else ""
            lines.append(
                f"{nama} & {r['segmen']} & {int(r['k'])}/{int(r['n'])} & "
                f"{_p(r['hit_rate'])} & {_p(r['wilson_lo'])}--{_p(r['wilson_hi'])} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    df = hitung()
    print("\n=== SELANG WILSON 95% HIT-RATE ARAH ===")
    print(df.to_string(index=False))
    print("\n=== TUMPANG TINDIH SELANG (bulan-perubahan) ===")
    print(tumpang_tindih(df).to_string(index=False))

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "directional_wilson_ci.csv", index=False)
        write_tex(df)
        print(f"\nCSV: {OUT_DIR}/directional_wilson_ci.csv")


if __name__ == "__main__":
    main()
