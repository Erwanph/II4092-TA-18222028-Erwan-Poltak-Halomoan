"""Prediksi vs aktual BI-Rate sepanjang timeline (train/val/test) untuk
inspeksi: apakah model menangkap perubahan, terutama pada bulan-perubahan
ekstrem. Output: CSV penuh, CSV khusus bulan-berubah (semua model), dan plot.

Jalankan: python -m src.analysis.pred_vs_actual
"""
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.modeling.config import DATASET_PATH, TARGET_COLUMN
from src.modeling.data_prep import (
    load_dataset, handle_missing_values, time_series_split, scale_features,
    create_sequences,
)
from src.modeling.machine_learning import RandomForestModel, XGBoostModel
from src.experiments.feature_engineering import apply_feature_engineering
from src.experiments.scenario_config import (
    generate_experiment_grid, build_experiment_config, DEFAULT_HYPERPARAMS,
    ALL_FEATURES,
)

ENSEMBLE_SEEDS = [42, 43, 44]
STEP = 0.25
OUT = "results/experiments/analysis"

SPECS = {
    "BiLSTM": {"exp": "E136", "kind": "BiLSTM",
               "p": {"units": 128, "dropout": 0.4682, "sequence_length": 3,
                     "learning_rate": 0.001702, "batch_size": 16, "epochs": 100}},
    "LSTM": {"exp": "E144", "kind": "LSTM",
             "p": {"units": 32, "dropout": 0.4924, "sequence_length": 3,
                   "learning_rate": 0.009188, "batch_size": 8, "epochs": 100}},
    "RF": {"exp": "E136", "kind": "RF",
           "p": {"n_estimators": 50, "max_depth": None, "min_samples_split": 11,
                 "min_samples_leaf": 2, "max_features": 1.0, "random_state": 42}},
    "XGBoost": {"exp": "E112", "kind": "XGBoost",
                "p": dict(DEFAULT_HYPERPARAMS["XGBoost"], random_state=42)},
}
MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
          7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}


def prep(exp_id):
    cfg = build_experiment_config({c["id"]: c for c in generate_experiment_grid()}[exp_id])
    df = load_dataset(DATASET_PATH)
    df = handle_missing_values(df, method=cfg["missing_method"])
    fe = cfg.get("feature_engineering")
    new = []
    if fe:
        df, new = apply_feature_engineering(df, fe, TARGET_COLUMN)
    feat = [c for c in list(cfg.get("feature_columns", ALL_FEATURES)) + new if c in df.columns]
    df2 = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    # label tanggal + split (samakan dgn time_series_split: 60/20/20)
    n = len(df2)
    test_idx = int(n * 0.8)
    val_idx = int(test_idx * 0.75)
    split = np.array(["train"] * val_idx + ["val"] * (test_idx - val_idx) + ["test"] * (n - test_idx))
    dates = [f"{int(t)}-{MONTHS[int(b)]}" for b, t in zip(df2["Bulan"], df2["Tahun"])]
    Xtr, Xva, Xte, ytr, yva, yte = time_series_split(df, TARGET_COLUMN, feat)
    Xtr_s, Xva_s, Xte_s, _ = scale_features(Xtr, Xva, Xte, cfg["scaler"])
    Xfull = np.vstack([Xtr_s, Xva_s, Xte_s])
    yfull = np.concatenate([ytr, yva, yte]).astype(float)
    return Xfull, yfull, np.array(dates), split, (Xtr_s, Xva_s, ytr, yva)


def predict_full(name, spec):
    Xfull, yfull, dates, split, (Xtr, Xva, ytr, yva) = prep(spec["exp"])
    n = len(yfull)
    p = spec["p"]
    pred = np.full(n, np.nan)
    if spec["kind"] in ("RF", "XGBoost"):
        if spec["kind"] == "RF":
            m = RandomForestModel(**p); m.fit(Xtr, ytr)
        else:
            m = XGBoostModel(**p); m.fit(Xtr, ytr, Xva, yva)
        pred = m.predict(Xfull)
    else:
        from src.modeling.deep_learning import LSTMModel, BiLSTMModel
        seq = p["sequence_length"]
        Xseq, _ = create_sequences(Xfull, yfull, seq)
        anc = yfull[seq - 1:-1]
        Xtr_s2, ytr_s2 = create_sequences(Xtr, ytr, seq)
        anc_tr = np.asarray(ytr, float)[seq - 1:-1]
        Cls = LSTMModel if spec["kind"] == "LSTM" else BiLSTMModel
        outs = []
        for s in ENSEMBLE_SEEDS:
            mm = Cls(sequence_length=seq, n_features=Xfull.shape[1], units=p["units"],
                     dropout=p["dropout"], learning_rate=p["learning_rate"],
                     batch_size=p["batch_size"], epochs=p["epochs"], residual=True, seed=s)
            mm.fit(Xtr_s2, ytr_s2, anchor_train=anc_tr)
            outs.append(mm.predict(Xseq, anchor=anc))
        pred[seq:] = np.mean(outs, axis=0)
    # snap ke grid 0,25 (kebijakan BI)
    pred_snap = np.where(np.isnan(pred), np.nan, np.round(pred / STEP) * STEP)
    return yfull, dates, split, pred, pred_snap


def main():
    os.makedirs(OUT, exist_ok=True)
    # full timeline tiap model -> satu tabel gabungan
    base = None
    cols = {}
    for name, spec in SPECS.items():
        y, dates, split, pred, pred_snap = predict_full(name, spec)
        if base is None:
            actual = y
            prev = np.concatenate([[np.nan], y[:-1]])
            delta = y - prev
            base = pd.DataFrame({
                "tanggal": dates, "split": split, "aktual": actual,
                "delta_aktual": np.round(delta, 2),
                "is_change": (np.abs(np.round(delta / STEP)) >= 1),
            })
        cols[f"pred_{name}"] = np.round(pred, 4)
        cols[f"pred_{name}_snap"] = pred_snap
    for k, v in cols.items():
        base[k] = v

    base.to_csv(f"{OUT}/pred_vs_actual_full.csv", index=False)

    # --- ringkasan bulan-berubah (semua split) ---
    chg = base[base["is_change"]].copy()
    prev_actual = base["aktual"].shift(1)
    for name in SPECS:
        ps = base[f"pred_{name}_snap"]
        dir_true = np.sign(np.round((base["aktual"] - prev_actual) / STEP))
        dir_pred = np.sign(np.round((ps - prev_actual) / STEP))
        base[f"arah_benar_{name}"] = (dir_true == dir_pred)
    chg = base[base["is_change"]].copy()
    chg.to_csv(f"{OUT}/change_months.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("\n=== BULAN-BERUBAH BI-RATE (train/val/test) — prediksi (snap 0,25) ===")
    show = ["tanggal", "split", "aktual", "delta_aktual",
            "pred_BiLSTM_snap", "pred_LSTM_snap", "pred_RF_snap", "pred_XGBoost_snap"]
    print(chg[show].to_string(index=False))

    print("\n=== Hit-rate arah pada BULAN-BERUBAH per split ===")
    for sp in ("train", "val", "test"):
        sub = base[(base["split"] == sp) & (base["is_change"])]
        if len(sub) == 0:
            continue
        line = f"  {sp:5s} (n={len(sub):2d}): "
        for name in SPECS:
            hr = sub[f"arah_benar_{name}"].mean()
            line += f"{name}={hr:.0%}  "
        print(line)

    # --- plot timeline model terbaik (BiLSTM) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 4.5))
        x = np.arange(len(base))
        ax.plot(x, base["aktual"], color="black", lw=1.8, label="Aktual")
        ax.plot(x, base["pred_BiLSTM_snap"], color="tab:red", lw=1.3, ls="--", label="Prediksi BiLSTM (snap 0,25)")
        ti = (base["split"] == "test").idxmax()
        vi = (base["split"] == "val").idxmax()
        ax.axvspan(vi, ti, color="orange", alpha=0.08, label="Validasi")
        ax.axvspan(ti, len(base), color="green", alpha=0.08, label="Uji")
        for j in base.index[base["is_change"]]:
            ax.axvline(j, color="gray", alpha=0.25, lw=0.6)
        step = max(1, len(base) // 24)
        ax.set_xticks(x[::step]); ax.set_xticklabels(base["tanggal"][::step], rotation=90, fontsize=7)
        ax.set_ylabel("BI-Rate (%)"); ax.legend(fontsize=8, loc="upper right")
        ax.set_title("BI-Rate Aktual vs Prediksi BiLSTM (residual+ensemble) — garis abu = bulan-berubah")
        fig.tight_layout()
        img = "../Tugas Akhir - Dokumen/images/pred_vs_actual_timeline.png"
        fig.savefig(img, dpi=130)
        print(f"\nPlot timeline disimpan: {img}")
    except Exception as e:
        print(f"(plot dilewati: {e})")

    print(f"CSV: {OUT}/pred_vs_actual_full.csv & change_months.csv")


if __name__ == "__main__":
    main()
