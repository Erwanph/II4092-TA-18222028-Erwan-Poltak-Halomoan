"""Diagnostik NN residual-anchor + ensembel pada konfigurasi pemenang.
Jalankan: python -m src.analysis.nn_diagnostics
"""
import os
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.modeling.config import DATASET_PATH, TARGET_COLUMN
from src.modeling.data_prep import (
    load_dataset, handle_missing_values, time_series_split, scale_features,
    create_sequences,
)
from src.modeling.metrics import calculate_rmse, r2_score
from src.modeling.deep_learning import LSTMModel, BiLSTMModel
from src.experiments.feature_engineering import apply_feature_engineering
from src.experiments.scenario_config import (
    generate_experiment_grid, build_experiment_config, ALL_FEATURES,
)

ENSEMBLE_SEEDS = [42, 43, 44]

CONFIGS = {
    "E136_LSTM": {"params": {"units": 16, "dropout": 0.496, "sequence_length": 6,
                              "learning_rate": 0.00965, "batch_size": 8, "epochs": 100}},
    "E144_BiLSTM": {"params": {"units": 64, "dropout": 0.2, "sequence_length": 8,
                                "learning_rate": 0.001, "batch_size": 16, "epochs": 100}},
}


def _prepare(config):
    df = load_dataset(DATASET_PATH)
    df = handle_missing_values(df, method=config["missing_method"])
    fe_config = config.get("feature_engineering")
    new_features = []
    if fe_config and isinstance(fe_config, dict):
        df, new_features = apply_feature_engineering(df, fe_config, TARGET_COLUMN)
    feature_cols = list(config.get("feature_columns", ALL_FEATURES))
    feature_cols = [c for c in feature_cols + new_features if c in df.columns]
    X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
        df, TARGET_COLUMN, feature_cols)
    X_tr, X_va, X_te, _ = scale_features(X_train, X_val, X_test, config["scaler"])
    return X_tr, X_va, X_te, y_train, y_val, y_test


def _ensemble_eval(kind, X_tr, X_va, X_te, y_tr, y_va, y_te, p):
    seq = p["sequence_length"]
    Xtr, ytr = create_sequences(X_tr, y_tr, seq)
    Xva, yva = create_sequences(X_va, y_va, seq)
    Xte, yte = create_sequences(X_te, y_te, seq)
    yt, yv, yte_a = (np.asarray(a, float) for a in (y_tr, y_va, y_te))
    anc_tr, anc_va, anc_te = yt[seq - 1:-1], yv[seq - 1:-1], yte_a[seq - 1:-1]

    Cls = LSTMModel if kind == "LSTM" else BiLSTMModel
    preds_te, preds_tr = [], []
    for seed in ENSEMBLE_SEEDS:
        m = Cls(sequence_length=seq, n_features=X_tr.shape[1],
                units=p["units"], dropout=p["dropout"],
                learning_rate=p["learning_rate"], batch_size=p["batch_size"],
                epochs=p["epochs"], residual=True, seed=seed)
        m.fit(Xtr, ytr, Xva, yva, anchor_train=anc_tr, anchor_val=anc_va)
        preds_te.append(m.predict(Xte, anchor=anc_te))
        preds_tr.append(m.predict(Xtr, anchor=anc_tr))
    pr_te = np.mean(preds_te, axis=0)
    pr_tr = np.mean(preds_tr, axis=0)

    rmse_te = calculate_rmse(yte, pr_te)
    # variasi antar-seed (uji konsistensi)
    r2_per_seed = [r2_score(yte, p_) for p_ in preds_te]
    return {
        "n_test": len(yte), "test_std": float(np.std(yte)),
        "R2_train": r2_score(ytr, pr_tr), "R2_test_ensemble": r2_score(yte, pr_te),
        "RMSE_test": rmse_te,
        "R2_test_min": min(r2_per_seed), "R2_test_max": max(r2_per_seed),
    }


def run():
    grid = {c["id"]: c for c in generate_experiment_grid()}
    for tag, spec in CONFIGS.items():
        exp_id, kind = tag.split("_")
        config = build_experiment_config(grid[exp_id])
        data = _prepare(config)
        r = _ensemble_eval(kind, *data, spec["params"])
        print(f"\n===== {tag} (residual + ensemble {len(ENSEMBLE_SEEDS)} seed) =====")
        print(f"  test_win={r['n_test']}  test_std={r['test_std']:.3f}")
        print(f"  R2_train={r['R2_train']:.3f}  R2_test(ensemble)={r['R2_test_ensemble']:.3f}")
        print(f"  R2_test per-seed: [{r['R2_test_min']:.3f} .. {r['R2_test_max']:.3f}]  "
              f"(rentang konsistensi)")
        print(f"  RMSE_test={r['RMSE_test']:.4f}")


if __name__ == "__main__":
    run()
