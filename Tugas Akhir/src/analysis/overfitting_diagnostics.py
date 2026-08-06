"""Diagnostik generalisasi: selisih latih-uji, anomali $R^2$ validasi, walk-forward.

Tiga hal yang dijawab modul ini.

1. Selisih kinerja latih/validasi/uji untuk KEEMPAT model evaluasi. Tabel
   regularisasi yang ada hanya memuat XGBoost, LSTM, dan BiLSTM; Random Forest,
   yang justru mencatat RMSE uji terbaik dan menjadi dasar seluruh analisis SHAP,
   belum pernah dilaporkan.

2. Sebab $R^2$ validasi bernilai negatif sementara $R^2$ uji tinggi. Dugaan awal
   adalah artefak variansi: bila target nyaris datar pada segmen validasi, $R^2$
   bisa negatif walau galatnya kecil. Pengukuran menolak dugaan itu - simpangan
   baku target pada validasi (0,562) justru LEBIH BESAR daripada pada uji
   (0,435), dan galat validasi memang besar (RMSE 0,760 pada RF, 0,969 pada
   XGBoost). Sebabnya adalah pergeseran rezim: validasi memuat pelonggaran COVID
   sampai 3,50% sementara target pada data latih tidak pernah di bawah 4,25%.
   Fungsi `run_regime_analysis` menguji mekanisme itu secara langsung.

3. Validasi bergulir (walk-forward / rolling origin) sebagai bukti generalisasi
   yang lebih kuat daripada satu pembagian tetap. Model dilatih ulang berulang
   dengan titik awal yang bergeser, selalu meramal satu bulan ke depan.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.evaluation_core import prep, SPECS, ENSEMBLE_SEEDS
from src.modeling.data_prep import create_sequences
from src.modeling.metrics import calculate_rmse, calculate_mape, r2_score
from src.modeling.machine_learning import RandomForestModel, XGBoostModel
from src.analysis.naive_benchmark import diebold_mariano

OUT_DIR = Path("results/experiments/analysis")
TEX_GENERALISASI = Path("../Tugas Akhir - Dokumen/tables/Tabel_Diagnostik_Generalisasi.tex")
TEX_WALKFWD = Path("../Tugas Akhir - Dokumen/tables/Tabel_Walk_Forward.tex")

MODELS = ["RF", "BiLSTM", "LSTM", "XGBoost"]
STEP = 0.25


def _predict_full(name, d, train_upto=None):
    """Prediksi seluruh timeline. `train_upto` menggeser batas data latih
    (dipakai walk-forward); None berarti memakai batas kanonis val_idx."""
    X, y = d["X"], d["y"]
    vi = d["val_idx"] if train_upto is None else train_upto
    p = SPECS[name]["p"]
    kind = SPECS[name]["kind"]

    if kind == "RF":
        m = RandomForestModel(**p)
        m.fit(X[:vi], y[:vi])
        return m.predict(X)
    if kind == "XGBoost":
        m = XGBoostModel(**p)
        if train_upto is None:
            # Skema kanonis: early stopping pada segmen validasi.
            m.fit(X[:vi], y[:vi], X[vi:d["test_idx"]], y[vi:d["test_idx"]])
        else:
            # Walk-forward: jendela early stopping WAJIB seluruhnya sebelum bulan
            # yang diramal, jika tidak model akan melihat masa depan.
            cut = max(1, vi - 12)
            m.fit(X[:cut], y[:cut], X[cut:vi], y[cut:vi])
        return m.predict(X)

    from src.modeling.deep_learning import LSTMModel, BiLSTMModel
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
                batch_size=p["batch_size"], epochs=p["epochs"],
                residual=True, seed=s)
        m.fit(Xtr_s, ytr_s, anchor_train=anc_tr)
        out = np.full(len(y), np.nan)
        out[seq:] = m.predict(Xseq, anchor=anc)
        preds.append(out)
    return np.nanmean(preds, axis=0)


def run_generalisasi(models=MODELS):
    """Metrik per segmen + variansi target per segmen (untuk menjelaskan $R^2$)."""
    rows = []
    for name in models:
        d = prep(SPECS[name]["exp"])
        y, split = d["y"], d["split"]
        pred = _predict_full(name, d)

        per = {}
        for seg in ("train", "val", "test"):
            m = (split == seg) & ~np.isnan(pred)
            ys, ps = y[m], pred[m]
            per[seg] = {
                "n": int(m.sum()), "RMSE": calculate_rmse(ys, ps),
                "MAPE": calculate_mape(ys, ps), "R2": r2_score(ys, ps),
                "sd_target": float(np.std(ys)),
                "rentang_target": float(ys.max() - ys.min()),
            }
        rows.append({
            "model": name,
            **{f"{k}_{seg}": per[seg][k] for seg in per for k in per[seg]},
            "R2_gap_train_test": per["train"]["R2"] - per["test"]["R2"],
            "RMSE_gap_train_test": per["test"]["RMSE"] - per["train"]["RMSE"],
        })
        print(f"[{name:7s}] "
              f"latih R2={per['train']['R2']:+.4f} RMSE={per['train']['RMSE']:.4f} sd={per['train']['sd_target']:.3f} | "
              f"val R2={per['val']['R2']:+.4f} RMSE={per['val']['RMSE']:.4f} sd={per['val']['sd_target']:.3f} | "
              f"uji R2={per['test']['R2']:+.4f} RMSE={per['test']['RMSE']:.4f} sd={per['test']['sd_target']:.3f}")
    return pd.DataFrame(rows)


def run_walk_forward(models=MODELS, n_origin=12, horizon=1):
    """Validasi bergulir: `n_origin` titik awal terakhir, ramalan 1 bulan ke depan.

    Pada setiap iterasi model dilatih ulang memakai seluruh data sebelum bulan
    target, lalu diminta meramal bulan itu. Ini meniru pemakaian nyata dan tidak
    pernah memakai informasi masa depan.
    """
    rows = []
    for name in models:
        d = prep(SPECS[name]["exp"])
        y, split = d["y"], d["split"]
        test_pos = np.where(split == "test")[0]
        origins = test_pos[-n_origin:]

        errs, e_rw = [], []
        for t in origins:
            pred = _predict_full(name, d, train_upto=t)
            if np.isnan(pred[t]):
                continue
            errs.append(y[t] - pred[t])
            e_rw.append(y[t] - y[t - 1])
        errs, e_rw = np.asarray(errs, float), np.asarray(e_rw, float)
        rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
        dm = diebold_mariano(errs, e_rw)
        rows.append({
            "model": name, "n_origin": len(errs), "horizon": horizon,
            "RMSE_walkforward": rms(errs), "MAE_walkforward": float(np.mean(np.abs(errs))),
            "RMSE_rw_walkforward": rms(e_rw),
            "Theil_U2_walkforward": rms(errs) / rms(e_rw),
            "DM_stat": dm["DM_stat"], "DM_p_value": dm["DM_p_value"],
        })
        print(f"[{name:7s}] walk-forward n={len(errs)} RMSE={rms(errs):.4f} "
              f"(RW {rms(e_rw):.4f}) U2={rms(errs)/rms(e_rw):.3f} "
              f"DM={dm['DM_stat']:+.3f} p={dm['DM_p_value']:.3f}")
    return pd.DataFrame(rows)


def run_regime_analysis(models=MODELS):
    """Uji kemampuan ekstrapolasi ke level kebijakan di luar pengalaman latih.

    Segmen validasi (Januari 2020-Desember 2022) memuat pelonggaran COVID sampai
    3,50% dan siklus pengetatan 2022, sedangkan target pada data latih tidak
    pernah turun di bawah 4,25%. Model berbasis pohon meramal rata-rata nilai
    daun sehingga secara struktural tidak dapat menghasilkan angka di luar
    rentang target latih; model residual meramal $y_{t-1}$ + delta sehingga
    levelnya dipasok jangkar, bukan dipelajari. Fungsi ini mengukur kedua
    perilaku itu secara langsung.
    """
    rows = []
    ref = prep(SPECS["RF"]["exp"])
    y_tr = ref["y"][ref["split"] == "train"]
    lo, hi = float(y_tr.min()), float(y_tr.max())

    for name in models:
        d = prep(SPECS[name]["exp"])
        pred = _predict_full(name, d)
        m = (d["split"] == "val") & ~np.isnan(pred)
        yv, pv = d["y"][m], pred[m]
        luar = yv < lo                     # bulan di luar rentang target latih
        rows.append({
            "model": name,
            "batas_bawah_latih": lo, "batas_atas_latih": hi,
            "pred_val_min": float(pv.min()), "pred_val_max": float(pv.max()),
            "aktual_val_min": float(yv.min()),
            "menembus_batas_bawah": bool(pv.min() < lo),
            "n_bulan_luar_rentang": int(luar.sum()),
            "bias_bulan_luar_rentang": float(np.mean(pv[luar] - yv[luar])) if luar.any() else np.nan,
            "RMSE_bulan_luar_rentang": calculate_rmse(yv[luar], pv[luar]) if luar.any() else np.nan,
        })
        print(f"[{name:7s}] pred val [{pv.min():.3f}; {pv.max():.3f}] "
              f"vs batas latih [{lo:.2f}; {hi:.2f}] | "
              f"{'MENEMBUS' if pv.min() < lo else 'TERKUNCI di atas'} batas bawah | "
              f"bias {luar.sum()} bulan luar rentang = {np.mean(pv[luar]-yv[luar]):+.3f}")
    return pd.DataFrame(rows)


def write_regime_tex(df, path=None):
    path = path or Path("../Tugas Akhir - Dokumen/tables/Tabel_Ekstrapolasi_Rezim.tex")
    lo = df["batas_bawah_latih"].iloc[0]
    lines = [
        "% Auto-generated oleh src/analysis/overfitting_diagnostics.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Kemampuan ekstrapolasi ke level kebijakan di luar rentang target "
        "data latih (segmen validasi, Januari 2020--Desember 2022)}",
        "\\label{tbl:ekstrapolasi-rezim}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|r|r|c|r|r|}", "\\hline",
        "\\textbf{Model} & \\textbf{Pred. min} & \\textbf{Pred. maks} & "
        "\\textbf{Menembus " + _n(lo, 2) + "\\%?} & \\textbf{Bias} & "
        "\\textbf{RMSE} \\\\ \\hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r.model} & {_n(r.pred_val_min, 3)} & {_n(r.pred_val_max, 3)} & "
            f"{'Ya' if r.menembus_batas_bawah else 'Tidak'} & "
            f"{_n(r.bias_bulan_luar_rentang, 3)} & "
            f"{_n(r.RMSE_bulan_luar_rentang, 3)} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def _n(x, dec=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{dec}f}".replace(".", ",").replace("-", "$-$")


def write_generalisasi_tex(df, path=TEX_GENERALISASI):
    lines = [
        "% Auto-generated oleh src/analysis/overfitting_diagnostics.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Diagnostik generalisasi per segmen data: galat, $R^2$, dan "
        "simpangan baku target}",
        "\\label{tbl:diagnostik-generalisasi}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|l|c|r|r|r|}", "\\hline",
        "\\textbf{Model} & \\textbf{Segmen} & \\textbf{$n$} & \\textbf{RMSE} & "
        "\\textbf{$R^2$} & \\textbf{SB target} \\\\ \\hline",
    ]
    label = {"train": "Latih", "val": "Validasi", "test": "Uji"}
    for _, r in df.iterrows():
        for k, seg in enumerate(("train", "val", "test")):
            first = f"\\multirow{{3}}{{*}}{{{r.model}}}" if k == 0 else ""
            lines.append(
                f"{first} & {label[seg]} & {int(r[f'n_{seg}'])} & "
                f"{_n(r[f'RMSE_{seg}'])} & {_n(r[f'R2_{seg}'])} & "
                f"{_n(r[f'sd_target_{seg}'], 3)} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}%", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def _rmse_kanonis():
    """RMSE uji pada pembagian tetap, diambil dari registry konfigurasi kanonis.

    Dipakai sebagai pembanding langsung terhadap RMSE validasi bergulir supaya
    kedua angka pada tabel memakai metrik yang sama.
    """
    import json
    p = Path("results/experiments/best_results_registry.json")
    if not p.exists():
        return {}
    reg = json.loads(p.read_text(encoding="utf-8"))
    return {m: v.get("RMSE") for m, v in reg.items()
            if isinstance(v, dict) and v.get("RMSE") is not None}


def write_walkforward_tex(df, path=TEX_WALKFWD):
    lines = [
        "% Auto-generated oleh src/analysis/overfitting_diagnostics.py",
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Validasi bergulir (\\textit{walk-forward}) satu bulan ke depan "
        "dengan pelatihan ulang pada setiap titik awal}",
        "\\label{tbl:walk-forward}",
        "\\begin{tabular}{|l|c|r|r|r|}", "\\hline",
        "\\textbf{Model} & \\textbf{Titik awal} & "
        "\\textbf{RMSE\\textsubscript{tetap}} & "
        "\\textbf{RMSE\\textsubscript{bergulir}} & "
        "$\\Delta$\\textbf{RMSE} \\\\ \\hline",
    ]
    kanonis = _rmse_kanonis()
    for _, r in df.iterrows():
        tetap = kanonis.get(r.model)
        d = r.RMSE_walkforward - tetap if tetap is not None else float("nan")
        lines.append(f"{r.model} & {int(r.n_origin)} & "
                     f"{_n(tetap) if tetap is not None else 'n/a'} & "
                     f"{_n(r.RMSE_walkforward)} & {_n(d)} \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabel LaTeX: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--skip-walkforward", action="store_true")
    ap.add_argument("--n-origin", type=int, default=12)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== DIAGNOSTIK GENERALISASI PER SEGMEN ===")
    gen = run_generalisasi()
    if args.write:
        gen.to_csv(OUT_DIR / "generalization_diagnostics.csv", index=False)
        write_generalisasi_tex(gen)
        print(f"CSV: {OUT_DIR/'generalization_diagnostics.csv'}")

    print("\n=== EKSTRAPOLASI KE REZIM DI LUAR RENTANG LATIH ===")
    reg = run_regime_analysis()
    if args.write:
        reg.to_csv(OUT_DIR / "regime_extrapolation.csv", index=False)
        write_regime_tex(reg)
        print(f"CSV: {OUT_DIR/'regime_extrapolation.csv'}")

    if not args.skip_walkforward:
        print(f"\n=== VALIDASI BERGULIR ({args.n_origin} titik awal) ===")
        wf = run_walk_forward(n_origin=args.n_origin)
        if args.write:
            wf.to_csv(OUT_DIR / "walk_forward.csv", index=False)
            write_walkforward_tex(wf)
            print(f"CSV: {OUT_DIR/'walk_forward.csv'}")


if __name__ == "__main__":
    main()
