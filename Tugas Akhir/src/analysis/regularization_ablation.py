"""Uji regularisasi bertahap (longgar -> ketat): R2 train/test + RMSE uji.
Keluaran: CSV + tabel LaTeX Bab VI. Jalankan: python -m src.analysis.regularization_ablation
"""
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.modeling.data_prep import create_sequences
from src.modeling.metrics import calculate_rmse, r2_score
from src.modeling.machine_learning import XGBoostModel

ENSEMBLE_SEEDS = [42, 43, 44]
OUT = "results/experiments/analysis"
TEX = "../Tugas Akhir - Dokumen/tables/Tabel_Ablation_Regularisasi.tex"

DL = {
    "BiLSTM": {"exp": "E136", "p": {"units": 128, "dropout": 0.4682, "sequence_length": 3,
                                    "learning_rate": 0.001702, "batch_size": 16, "epochs": 100}},
    "LSTM": {"exp": "E144", "p": {"units": 32, "dropout": 0.4924, "sequence_length": 3,
                                  "learning_rate": 0.009188, "batch_size": 8, "epochs": 100}},
}
L2_LEVELS = [0.0, 1e-4, 1e-3, 1e-2]
XGB_LEVELS = [
    ("longgar", dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=3)),
    ("sedang", dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=3.0, min_child_weight=5)),
    ("ketat", dict(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.7,
                   colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, min_child_weight=8)),
    ("sangat ketat", dict(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.7,
                          colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=8.0, min_child_weight=10)),
]


# Memakai pipeline & split kanonis yang sama dengan src.evaluation_core
# (60/20/20 atas porsi historis) agar angka ablasi konsisten dengan tabel
# kinerja di Bab VI.
from src.evaluation_core import prep as canon_prep, SPECS


def dl_eval(kind, exp, p, l2):
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
    preds = []
    for s in ENSEMBLE_SEEDS:
        m = Cls(sequence_length=seq, n_features=Xseq.shape[2], units=p["units"],
                dropout=p["dropout"], learning_rate=p["learning_rate"],
                batch_size=p["batch_size"], epochs=p["epochs"], residual=True, seed=s, l2=l2)
        m.fit(Xtr_s, ytr_s, anchor_train=anc_tr)
        pf = np.full(len(y), np.nan)
        pf[seq:] = m.predict(Xseq, anchor=anc)
        preds.append(pf)
    pf = np.nanmean(preds, axis=0)
    valid = ~np.isnan(pf)
    tr = (d["split"] == "train") & valid
    te = (d["split"] == "test") & valid
    r2tr = r2_score(y[tr], pf[tr])
    r2te = r2_score(y[te], pf[te])
    return r2tr, r2te, calculate_rmse(y[te], pf[te])


def xgb_eval(p):
    d = canon_prep(SPECS["XGBoost"]["exp"])
    X, y = d["X"], d["y"]
    vi, ti = d["val_idx"], d["test_idx"]
    tr = d["split"] == "train"
    te = d["split"] == "test"
    m = XGBoostModel(random_state=42, **p)
    m.fit(X[:vi], y[:vi], X[vi:ti], y[vi:ti])   # early stopping pd validasi (sama dgn produksi)
    r2tr = r2_score(y[tr], m.predict(X[tr]))
    r2te = r2_score(y[te], m.predict(X[te]))
    return r2tr, r2te, calculate_rmse(y[te], m.predict(X[te]))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, spec in DL.items():
        for l2 in L2_LEVELS:
            r2tr, r2te, rmse = dl_eval(name, spec["exp"], spec["p"], l2)
            rows.append({"model": name, "regularisasi": "L2=" + f"{l2:g}".replace(".", ","),
                         "R2_train": r2tr, "R2_test": r2te,
                         "gap": r2tr - r2te, "RMSE_test": rmse})
            print(f"{name:7s} L2={l2:<7g} R2tr={r2tr:.3f} R2te={r2te:.3f} gap={r2tr-r2te:.3f} RMSE={rmse:.4f}")
    for lbl, p in XGB_LEVELS:
        r2tr, r2te, rmse = xgb_eval(p)
        rows.append({"model": "XGBoost", "regularisasi": lbl,
                     "R2_train": r2tr, "R2_test": r2te,
                     "gap": r2tr - r2te, "RMSE_test": rmse})
        print(f"XGBoost {lbl:<12s} R2tr={r2tr:.3f} R2te={r2te:.3f} gap={r2tr-r2te:.3f} RMSE={rmse:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/regularization_ablation.csv", index=False)

    # tabel LaTeX
    def fnum(x):
        return f"{x:.3f}".replace(".", ",")
    lines = [
        "% Auto-generated oleh src/analysis/regularization_ablation.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Pengaruh kekuatan regularisasi terhadap indikasi \\textit{overfitting}}",
        "\\label{tbl:ablation-regularisasi}",
        "\\begin{tabular}{|l|l|r|r|r|r|}", "\\hline",
        "\\textbf{Model} & \\textbf{Regularisasi} & \\textbf{$R^2$ Latih} & "
        "\\textbf{$R^2$ Uji} & \\textbf{Selisih} & \\textbf{RMSE Uji} \\\\ \\hline",
    ]
    for _, r in out.iterrows():
        lines.append(f"{r['model']} & {r['regularisasi']} & {fnum(r.R2_train)} & "
                     f"{fnum(r.R2_test)} & {fnum(r.gap)} & {fnum(r.RMSE_test)} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}"]
    with open(TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nCSV: {OUT}/regularization_ablation.csv")
    print(f"Tabel LaTeX: {TEX}")


if __name__ == "__main__":
    main()
