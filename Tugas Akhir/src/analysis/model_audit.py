"""Audit model terbaik: matriks fitur final per konfigurasi + cek overfit
(R2 train/val/test). Jalankan: python -m src.analysis.model_audit
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
from src.modeling.metrics import calculate_rmse, r2_score
from src.modeling.machine_learning import RandomForestModel, XGBoostModel
from src.experiments.feature_engineering import apply_feature_engineering
from src.experiments.scenario_config import (
    generate_experiment_grid, build_experiment_config, DEFAULT_HYPERPARAMS,
    ALL_FEATURES,
)

ENSEMBLE_SEEDS = [42, 43, 44]

# Model terbaik per registry (exp_id, jenis, hyperparameter pemenang)
BEST = {
    "RF_E136": {"exp": "E136", "kind": "RF",
                "p": {"n_estimators": 50, "max_depth": None, "min_samples_split": 11,
                      "min_samples_leaf": 2, "max_features": 1.0, "random_state": 42}},
    "XGB_E112": {"exp": "E112", "kind": "XGBoost",
                 "p": dict(DEFAULT_HYPERPARAMS["XGBoost"], random_state=42)},
    "LSTM_E144": {"exp": "E144", "kind": "LSTM",
                  "p": {"units": 32, "dropout": 0.4924, "sequence_length": 3,
                        "learning_rate": 0.009188, "batch_size": 8, "epochs": 100}},
    "BiLSTM_E136": {"exp": "E136", "kind": "BiLSTM",
                    "p": {"units": 128, "dropout": 0.4682, "sequence_length": 3,
                          "learning_rate": 0.001702, "batch_size": 16, "epochs": 100}},
}


def build_df(config):
    df = load_dataset(DATASET_PATH)
    df = handle_missing_values(df, method=config["missing_method"])
    fe_config = config.get("feature_engineering")
    new_features = []
    if fe_config and isinstance(fe_config, dict):
        df, new_features = apply_feature_engineering(df, fe_config, TARGET_COLUMN)
    feature_cols = list(config.get("feature_columns", ALL_FEATURES))
    feature_cols = [c for c in feature_cols + new_features if c in df.columns]
    return df, feature_cols


def split_scale(df, feature_cols, scaler):
    Xtr, Xva, Xte, ytr, yva, yte = time_series_split(df, TARGET_COLUMN, feature_cols)
    Xtr_s, Xva_s, Xte_s, _ = scale_features(Xtr, Xva, Xte, scaler)
    return Xtr_s, Xva_s, Xte_s, ytr, yva, yte


def r2_split_tree(model, Xtr, Xva, Xte, ytr, yva, yte):
    return (r2_score(ytr, model.predict(Xtr)), r2_score(yva, model.predict(Xva)),
            r2_score(yte, model.predict(Xte)))


def eval_tree(kind, p, Xtr, Xva, Xte, ytr, yva, yte):
    if kind == "RF":
        m = RandomForestModel(**p); m.fit(Xtr, ytr)
    else:
        m = XGBoostModel(**p); m.fit(Xtr, ytr, Xva, yva)
    r2 = r2_split_tree(m, Xtr, Xva, Xte, ytr, yva, yte)
    rmse = calculate_rmse(yte, m.predict(Xte))
    return r2, rmse


def eval_nn(kind, p, Xtr, Xva, Xte, ytr, yva, yte):
    from src.modeling.deep_learning import LSTMModel, BiLSTMModel
    seq = p["sequence_length"]
    Xtrs, ytrs = create_sequences(Xtr, ytr, seq)
    Xvas, yvas = create_sequences(Xva, yva, seq)
    Xtes, ytes = create_sequences(Xte, yte, seq)
    yt, yv, yte_a = (np.asarray(a, float) for a in (ytr, yva, yte))
    anc_tr, anc_va, anc_te = yt[seq - 1:-1], yv[seq - 1:-1], yte_a[seq - 1:-1]
    Cls = LSTMModel if kind == "LSTM" else BiLSTMModel
    ptr, pte = [], []
    for s in ENSEMBLE_SEEDS:
        m = Cls(sequence_length=seq, n_features=Xtr.shape[1], units=p["units"],
                dropout=p["dropout"], learning_rate=p["learning_rate"],
                batch_size=p["batch_size"], epochs=p["epochs"], residual=True, seed=s)
        m.fit(Xtrs, ytrs, Xvas, yvas, anchor_train=anc_tr, anchor_val=anc_va)
        ptr.append(m.predict(Xtrs, anchor=anc_tr))
        pte.append(m.predict(Xtes, anchor=anc_te))
    pr_tr, pr_te = np.mean(ptr, axis=0), np.mean(pte, axis=0)
    r2 = (r2_score(ytrs, pr_tr), float("nan"), r2_score(ytes, pr_te))
    return r2, calculate_rmse(ytes, pr_te), (ytes, anc_te, len(ytes))


def main():
    grid = {c["id"]: c for c in generate_experiment_grid()}

    # --- BAGIAN 1: dataset per feature-set ---
    seen = {}
    for tag, spec in BEST.items():
        config = build_experiment_config(grid[spec["exp"]])
        key = config.get("selection_method"), tuple(config.get("feature_columns", ["ALL"]))
        fkey = "domain_itf" if config.get("feature_columns") else "all"
        if fkey in seen:
            continue
        seen[fkey] = True
        df, feat = build_df(config)
        Xdf = df[feat]
        print("\n" + "=" * 78)
        print(f"DATASET — feature-set '{fkey}' (dipakai: "
              f"{[t for t,s in BEST.items() if (build_experiment_config(grid[s['exp']]).get('feature_columns') is not None)==(fkey=='domain_itf')]})")
        print("=" * 78)
        print(f"Jumlah fitur: {len(feat)}  | baris: {len(Xdf)}")
        print("Kolom fitur:", feat)
        print("\n--- head(10) ---")
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(Xdf.head(10).to_string())
        print("\n--- info() ---")
        Xdf.info()

    # --- BAGIAN 2: cek overfit (R2 train/val/test) ---
    print("\n\n" + "=" * 78)
    print("AUDIT OVERFIT — R2 train/val/test")
    print("=" * 78)
    print(f"{'Model':12s} {'R2_train':>9s} {'R2_val':>8s} {'R2_test':>8s} "
          f"{'gap(tr-te)':>10s} {'RMSE':>7s}")
    for tag, spec in BEST.items():
        config = build_experiment_config(grid[spec["exp"]])
        df, feat = build_df(config)
        Xtr, Xva, Xte, ytr, yva, yte = split_scale(df, feat, config["scaler"])
        if spec["kind"] in ("RF", "XGBoost"):
            (r2tr, r2va, r2te), rmse = eval_tree(spec["kind"], spec["p"],
                                                 Xtr, Xva, Xte, ytr, yva, yte)
        else:
            (r2tr, r2va, r2te), rmse, (ytes, anc, _) = eval_nn(
                spec["kind"], spec["p"], Xtr, Xva, Xte, ytr, yva, yte)
        vstr = f"{r2va:8.3f}" if not np.isnan(r2va) else f"{'—':>8s}"
        print(f"{tag:12s} {r2tr:9.3f} {vstr} {r2te:8.3f} {r2tr-r2te:10.3f} "
              f"{rmse:7.4f}")


if __name__ == "__main__":
    main()
