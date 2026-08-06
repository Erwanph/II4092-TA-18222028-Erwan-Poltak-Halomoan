"""Konsistensi atribusi XAI antarmodel setelah persistensi dikeluarkan.

Atribusi SHAP keempat model tampak sangat berbeda pada bulan yang sama: model
berbasis pohon dipimpin lag BI-Rate dengan kontribusi besar, sedangkan model
\textit{deep learning} tidak memunculkan lag sama sekali dan besaran
kontribusinya jauh lebih kecil. Perbedaan itu bukan pertanda salah satu model
keliru, melainkan akibat perbedaan target: model residual meramal $y_{t-1}$ +
delta sehingga SHAP-nya hanya menjelaskan koreksi delta, sementara model pohon
meramal level sehingga SHAP-nya memuat seluruh level.

Konsekuensinya, nilai SHAP antarkelompok model TIDAK sebanding secara langsung.
Modul ini melakukan perbandingan yang sahih dengan dua penyesuaian: persistensi
(lag BI-Rate) dikeluarkan dari perbandingan, lalu kontribusi sisanya
dinormalisasi menjadi pangsa relatif. Dengan begitu pertanyaan yang diuji menjadi
tepat, yaitu apakah keempat model sepakat mengenai pendorong makroekonomi mana
yang paling berperan.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.evaluation_core import prep, train_predict, shap_instance, SPECS, DISPLAY

OUT_DIR = Path("results/experiments/analysis")
TEX_PATH = Path("../Tugas Akhir - Dokumen/tables/Tabel_Konsistensi_XAI.tex")

MODELS = ["RF", "XGBoost", "LSTM", "BiLSTM"]
PERSISTENSI = {"BI_Rate_Pct_lag1", "BI_Rate_Pct_lag2", "BI_Rate_Pct_lag3"}
BULAN_KONTROL = "2025-Jul"


def kontribusi_semua(bulan=BULAN_KONTROL, models=MODELS):
    """Kembalikan {model: {fitur: kontribusi SHAP}} pada satu bulan kontrol."""
    hasil = {}
    for name in models:
        d = prep(SPECS[name]["exp"])
        idx = np.where(d["dates"] == bulan)[0]
        if not len(idx):
            raise ValueError(f"Bulan {bulan} tidak ada pada timeline {name}.")
        row = int(idx[0])
        _, _, art = train_predict(name, d)
        contrib = shap_instance(name, d, art, row)
        if contrib is None:
            raise RuntimeError(f"SHAP gagal untuk {name}.")
        hasil[name] = contrib
        print(f"[{name:7s}] {len(contrib)} fitur; "
              f"teratas: {list(contrib)[0]} ({list(contrib.values())[0]:+.4f})")
    return hasil


def bangun_tabel(hasil):
    """Pangsa relatif dan peringkat pendorong makro (persistensi dikeluarkan)."""
    bersama = set.intersection(*(set(c) for c in hasil.values())) - PERSISTENSI
    bersama = sorted(bersama)

    baris = {}
    for name, contrib in hasil.items():
        v = np.array([abs(contrib[f]) for f in bersama], float)
        total = v.sum() or 1.0
        pangsa = v / total * 100
        # peringkat 1 = pangsa terbesar
        urut = (-pangsa).argsort()
        rank = np.empty(len(bersama), int)
        rank[urut] = np.arange(1, len(bersama) + 1)
        baris[name] = {"pangsa": pangsa, "rank": rank,
                       "arah": np.array([np.sign(contrib[f]) for f in bersama])}

    df = pd.DataFrame({"fitur": bersama,
                       "fitur_tampil": [DISPLAY.get(f, f) for f in bersama]})
    for name in hasil:
        df[f"pangsa_{name}"] = baris[name]["pangsa"]
        df[f"rank_{name}"] = baris[name]["rank"]
        df[f"arah_{name}"] = baris[name]["arah"]
    df["rank_rata"] = df[[f"rank_{m}" for m in hasil]].mean(axis=1)
    return df.sort_values("rank_rata").reset_index(drop=True), baris, bersama


def statistik_kesepakatan(baris, bersama):
    """Korelasi peringkat Spearman antarpasangan model + tumpang tindih top-3."""
    names = list(baris)
    pasangan, top3 = [], {}
    for n in names:
        urut = np.argsort(baris[n]["rank"])
        top3[n] = {bersama[i] for i in urut[:3]}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rho, p = stats.spearmanr(baris[a]["rank"], baris[b]["rank"])
            pasangan.append({"model_a": a, "model_b": b,
                             "spearman_rho": float(rho), "p_value": float(p),
                             "tumpang_tindih_top3": len(top3[a] & top3[b])})
    return pd.DataFrame(pasangan), top3


def uji_kestabilan(bulan=BULAN_KONTROL, models=MODELS, n_ulang=5):
    """Ukur kestabilan estimasi SHAP antarpengulangan pada satu bulan yang sama.

    TreeExplainer bersifat eksak sehingga hasilnya identik setiap kali dijalankan.
    Sebaliknya, KernelExplainer yang dipakai untuk model residual membangkitkan
    sampel perturbasi secara acak, sehingga estimasinya dapat berubah
    antarpengulangan. Fungsi ini menjalankan ekstraksi SHAP berulang kali dan
    melaporkan seberapa sering fitur teratas berubah serta simpangan baku pangsa
    kontribusinya.
    """
    rows = []
    for name in models:
        d = prep(SPECS[name]["exp"])
        row = int(np.where(d["dates"] == bulan)[0][0])
        _, _, art = train_predict(name, d)

        teratas, pangsa_per_fitur = [], {}
        for _ in range(n_ulang):
            c = shap_instance(name, d, art, row)
            if c is None:
                continue
            makro = {k: abs(v) for k, v in c.items() if k not in PERSISTENSI}
            total = sum(makro.values()) or 1.0
            teratas.append(max(makro, key=makro.get))
            for k, v in makro.items():
                pangsa_per_fitur.setdefault(k, []).append(v / total * 100)

        sd = {k: float(np.std(v)) for k, v in pangsa_per_fitur.items()}
        rows.append({
            "model": name, "n_ulang": len(teratas),
            "n_fitur_teratas_berbeda": len(set(teratas)),
            "fitur_teratas": " | ".join(dict.fromkeys(teratas)),
            "sd_pangsa_maks": max(sd.values()) if sd else np.nan,
            "sd_pangsa_rata": float(np.mean(list(sd.values()))) if sd else np.nan,
        })
        print(f"[{name:7s}] {len(teratas)} ulangan; fitur teratas berbeda: "
              f"{len(set(teratas))}; SB pangsa maks {max(sd.values()):.2f} poin "
              f"({' | '.join(dict.fromkeys(teratas))})")
    return pd.DataFrame(rows)


def _n(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",").replace("-", "$-$")


def write_tex(df, pasangan, path=TEX_PATH, topn=8):
    rho_min = pasangan["spearman_rho"].min()
    rho_max = pasangan["spearman_rho"].max()
    lines = [
        "% Auto-generated oleh src/analysis/xai_consistency.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Pangsa relatif kontribusi SHAP pendorong makroekonomi antarmodel "
        "pada Juli 2025, setelah persistensi BI-Rate dikeluarkan dari perbandingan}",
        "\\label{tbl:konsistensi-xai}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|r|r|r|r|c|}", "\\hline",
        "\\textbf{Pendorong makroekonomi} & \\textbf{RF} & \\textbf{XGBoost} & "
        "\\textbf{LSTM} & \\textbf{BiLSTM} & \\textbf{Peringkat rata-rata} "
        "\\\\ \\hline",
    ]
    for _, r in df.head(topn).iterrows():
        lines.append(
            f"{r.fitur_tampil} & {_n(r.pangsa_RF)}\\% & {_n(r.pangsa_XGBoost)}\\% & "
            f"{_n(r.pangsa_LSTM)}\\% & {_n(r.pangsa_BiLSTM)}\\% & "
            f"{_n(r.rank_rata, 2)} \\\\")
        lines.append("\\hline")
    # Rentang korelasi Spearman tidak dicetak sebagai catatan di bawah tabel;
    # angkanya dibahas sebagai argumen pada paragraf Bab VI.
    print(f"Rentang rho Spearman antarpasangan model: "
          f"{rho_min:.2f} sampai {rho_max:.2f}")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--bulan", default=BULAN_KONTROL)
    ap.add_argument("--kestabilan", type=int, default=0,
                    help="jumlah pengulangan uji kestabilan SHAP (0 = lewati)")
    args = ap.parse_args()

    print(f"\n=== KONTRIBUSI SHAP PENUH ({args.bulan}) ===")
    hasil = kontribusi_semua(args.bulan)

    df, baris, bersama = bangun_tabel(hasil)
    print(f"\n=== PANGSA PENDORONG MAKRO ({len(bersama)} fitur bersama) ===")
    kol = ["fitur_tampil"] + [f"pangsa_{m}" for m in MODELS] + ["rank_rata"]
    print(df[kol].head(10).to_string(index=False))

    pasangan, top3 = statistik_kesepakatan(baris, bersama)
    print("\n=== KESEPAKATAN ANTARMODEL ===")
    print(pasangan.to_string(index=False))
    print("\nTiga pendorong teratas tiap model:")
    for n, s in top3.items():
        print(f"  {n:8s}: {sorted(DISPLAY.get(f, f) for f in s)}")

    if args.kestabilan:
        print(f"\n=== KESTABILAN ESTIMASI SHAP ({args.kestabilan} pengulangan) ===")
        stab = uji_kestabilan(args.bulan, n_ulang=args.kestabilan)
        if args.write:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            stab.to_csv(OUT_DIR / "xai_stability.csv", index=False)
            print(f"CSV: {OUT_DIR}/xai_stability.csv")

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = args.bulan.replace("-", "_")
        df.to_csv(OUT_DIR / f"xai_consistency_{tag}.csv", index=False)
        pasangan.to_csv(OUT_DIR / f"xai_agreement_{tag}.csv", index=False)
        if args.bulan == BULAN_KONTROL:
            write_tex(df, pasangan)
        print(f"CSV: {OUT_DIR}/xai_consistency_{tag}.csv")


if __name__ == "__main__":
    main()
