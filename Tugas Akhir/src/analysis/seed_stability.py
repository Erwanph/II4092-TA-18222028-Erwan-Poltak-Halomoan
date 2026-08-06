"""Kestabilan antarseed konfigurasi L2 pada Tabel VI.24.

Latar belakang. Uji ablasi regularisasi melaporkan LSTM dengan L2 = 0,01
mencatat RMSE uji 0,119 -- lebih rendah daripada seluruh model kanonis di buku,
termasuk Random Forest (0,1434). Angka itu dihitung dari satu ensembel tiga
seed. Pertanyaan yang wajar: apakah keunggulan tersebut bertahan bila seed
diganti, atau hanya kebetulan satu ensembel yang beruntung.

Modul ini melatih sepuluh seed untuk tiap kombinasi (model x tingkat L2), lalu
melaporkan dua hal:

1. Sebaran RMSE uji per seed tunggal -- ukuran derau pelatihan mentah.
2. Sebaran RMSE uji seluruh ensembel tiga seed yang dapat dibentuk dari
   sepuluh seed itu (C(10,3) = 120 ensembel). Ini setara dengan cara angka
   di Tabel VI.24 dihasilkan, jadi dari sinilah pertanyaan "bertahan atau
   tidak" dijawab.

Split dan pipeline mengikuti src.evaluation_core supaya sebanding dengan
angka kanonis Bab VI.

Jalankan: python -m src.analysis.seed_stability --write
"""
import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.modeling.data_prep import create_sequences
from src.modeling.metrics import calculate_rmse
from src.evaluation_core import prep as canon_prep, SPECS
from src.analysis.regularization_ablation import DL

OUT_DIR = Path("results/experiments/analysis")
TEX = Path("../Tugas Akhir - Dokumen/tables/Tabel_Stabilitas_Seed.tex")

SEEDS = list(range(42, 52))          # sepuluh seed
L2_LEVELS = [0.0, 0.01]              # kanonis vs kandidat pada Tabel VI.24
ENSEMBLE_SIZE = 3                    # sama dengan ensembel kanonis


def prediksi_per_seed(kind, p, l2, seeds=SEEDS):
    """Prediksi uji tiap seed untuk satu konfigurasi (model, L2)."""
    from src.modeling.deep_learning import LSTMModel, BiLSTMModel

    d = canon_prep(SPECS[kind]["exp"])
    X, y = d["X"], d["y"]
    vi = d["val_idx"]
    seq = p["sequence_length"]
    Xseq, _ = create_sequences(X, y, seq)
    anc = y[seq - 1:-1]
    Xtr_s, ytr_s = create_sequences(X[:vi], y[:vi], seq)
    anc_tr = np.asarray(y[:vi], float)[seq - 1:-1]
    Cls = LSTMModel if kind == "LSTM" else BiLSTMModel

    te = d["split"] == "test"
    preds = {}
    for s in seeds:
        m = Cls(sequence_length=seq, n_features=Xseq.shape[2], units=p["units"],
                dropout=p["dropout"], learning_rate=p["learning_rate"],
                batch_size=p["batch_size"], epochs=p["epochs"],
                residual=True, seed=s, l2=l2)
        m.fit(Xtr_s, ytr_s, anchor_train=anc_tr)
        pf = np.full(len(y), np.nan)
        pf[seq:] = m.predict(Xseq, anchor=anc)
        preds[s] = pf
        print(f"    {kind} L2={l2:g} seed={s} "
              f"RMSE={calculate_rmse(y[te], pf[te]):.4f}")
    return preds, y, te


def ringkas(kind, l2, preds, y, te):
    """Statistik seed tunggal dan seluruh ensembel tiga seed."""
    per_seed = {s: calculate_rmse(y[te], p[te]) for s, p in preds.items()}
    ens = []
    for combo in itertools.combinations(sorted(preds), ENSEMBLE_SIZE):
        pf = np.mean([preds[s] for s in combo], axis=0)
        ens.append(calculate_rmse(y[te], pf[te]))
    ens = np.array(ens)
    v = np.array(list(per_seed.values()))
    return {
        "model": kind,
        "l2": l2,
        "n_seed": len(per_seed),
        "rmse_seed_min": v.min(), "rmse_seed_median": float(np.median(v)),
        "rmse_seed_max": v.max(), "rmse_seed_std": float(v.std(ddof=1)),
        "n_ensembel": len(ens),
        "rmse_ens_min": ens.min(), "rmse_ens_median": float(np.median(ens)),
        "rmse_ens_max": ens.max(), "rmse_ens_std": float(ens.std(ddof=1)),
        # seberapa sering ensembel tiga seed mengungguli RF kanonis (0,1434)
        "pangsa_ens_di_bawah_RF": float(np.mean(ens < 0.1434)) * 100,
    }


def _n(x, dec=4):
    return f"{x:.{dec}f}".replace(".", ",")


def write_tex(df, path=TEX):
    lines = [
        "% Auto-generated oleh src/analysis/seed_stability.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Sebaran RMSE uji konfigurasi regularisasi L2 atas sepuluh "
        "seed pelatihan}",
        "\\label{tbl:stabilitas-seed}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|c|r|r|r|r|r|}", "\\hline",
        "\\multirow{2}{*}{\\textbf{Model}} & \\multirow{2}{*}{\\textbf{L2}} & "
        "\\multicolumn{2}{c|}{\\textbf{Seed tunggal (10)}} & "
        "\\multicolumn{3}{c|}{\\textbf{Ensembel tiga seed (120)}} \\\\ \\cline{3-7}",
        " & & \\textbf{Median} & \\textbf{Simpangan} & \\textbf{Terendah} & "
        "\\textbf{Median} & \\textbf{Tertinggi} \\\\ \\hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']} & {str(r['l2']).replace('.', ',')} & "
            f"{_n(r['rmse_seed_median'])} & {_n(r['rmse_seed_std'])} & "
            f"{_n(r['rmse_ens_min'])} & {_n(r['rmse_ens_median'])} & "
            f"{_n(r['rmse_ens_max'])} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = []
    for kind, spec in DL.items():
        for l2 in L2_LEVELS:
            print(f"\n=== {kind} L2={l2:g} ===")
            preds, y, te = prediksi_per_seed(kind, spec["p"], l2)
            rows.append(ringkas(kind, l2, preds, y, te))

    df = pd.DataFrame(rows)
    print("\n=== RINGKASAN ===")
    print(df.to_string(index=False))

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_DIR / "seed_stability.csv", index=False)
        write_tex(df)
        print(f"\nCSV: {OUT_DIR}/seed_stability.csv")


if __name__ == "__main__":
    main()
