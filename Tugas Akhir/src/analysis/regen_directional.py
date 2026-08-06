"""Regenerasi diagnostik akurasi arah RF/XGBoost (sumber Tabel VI.15).

Mereplikasi penyiapan data fase focused lalu melatih ulang kedua model pohon
(deterministik) sehingga test-RMSE harus cocok registry. Default hanya
memverifikasi CSV yang ada; pakai --write untuk menimpa.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.config import DATASET_PATH, TARGET_COLUMN
from src.modeling.data_prep import (
    load_dataset, handle_missing_values, time_series_split, scale_features,
)
from src.modeling.metrics import calculate_rmse, calculate_mape, r2_score
from src.modeling.machine_learning import RandomForestModel, XGBoostModel
from src.experiments.feature_engineering import (
    apply_feature_engineering, select_features_mutual_info,
)
from src.experiments.scenario_config import (
    generate_experiment_grid, build_experiment_config,
    DEFAULT_HYPERPARAMS, ALL_FEATURES,
)

STEP = 0.25  # BI-Rate hanya valid pada kelipatan 0,25%
OUT_PATH = Path("results/experiments/analysis/directional_diagnostics.csv")
TOL = 1e-9   # ambang kecocokan angka terhadap CSV lama

# Hyperparameter pemenang + acuan RMSE test (kanonis Windows). RF portabel
# lintas platform; XGBoost tidak (Mac menghasilkan 0,2207, bukan 0,1960) —
# lihat README bagian Reproduksibilitas.
MODEL_SPECS = {
    "E136": {"model": "RF", "registry_rmse": 0.14337659095244173,
             "params": {"n_estimators": 50, "max_depth": None, "min_samples_split": 11,
                        "min_samples_leaf": 2, "max_features": 1.0, "random_state": 42}},
    "E112": {"model": "XGBoost", "registry_rmse": 0.19601017688919078,
             "params": dict(DEFAULT_HYPERPARAMS.get("XGBoost", {}), random_state=42)},
}


def _prepare(config):
    """Replikasi persis penyiapan data fase focused (experiment_runner)."""
    df = load_dataset(DATASET_PATH)
    extra_kw = {}
    if config["missing_method"] == "knn":
        extra_kw["knn_neighbors"] = config.get("knn_neighbors", 5)
    df = handle_missing_values(df, method=config["missing_method"], **extra_kw)

    fe_config = config.get("feature_engineering")
    new_features = []
    if fe_config and isinstance(fe_config, dict):
        df, new_features = apply_feature_engineering(df, fe_config, TARGET_COLUMN)

    feature_cols = list(config.get("feature_columns", ALL_FEATURES))
    if config.get("selection_method") == "mutual_info":
        feature_cols, _ = select_features_mutual_info(
            df, TARGET_COLUMN, feature_cols, n_features=config.get("n_select", 7))
    feature_cols = [c for c in feature_cols + new_features if c in df.columns]

    X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
        df, TARGET_COLUMN, feature_cols)
    X_tr, X_va, X_te, _ = scale_features(
        X_train, X_val, X_test, scaler_type=config["scaler"])
    return X_tr, X_va, X_te, y_train, y_val, y_test


def _directional(y_true, y_pred, y_prev, step=STEP):
    """Hit-rate arah keseluruhan + khusus bulan-berubah/bulan-tahan + RMSE per jenis bulan.

    Galat memakai prediksi MENTAH; arah memakai tanda selisih yang dibulatkan ke grid.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_prev = np.asarray(y_prev, float)
    dir_true = np.sign(np.round((y_true - y_prev) / step))
    dir_pred = np.sign(np.round((y_pred - y_prev) / step))
    change = dir_true != 0
    hold = ~change
    return {
        "hit_rate_all": float(np.mean(dir_true == dir_pred)),
        "n_change": int(change.sum()), "n_hold": int(hold.sum()),
        "hit_rate_change": float(np.mean(dir_true[change] == dir_pred[change])) if change.any() else float("nan"),
        "hit_rate_hold": float(np.mean(dir_true[hold] == dir_pred[hold])) if hold.any() else float("nan"),
        "rmse_change": calculate_rmse(y_true[change], y_pred[change]) if change.any() else float("nan"),
        "rmse_hold": calculate_rmse(y_true[hold], y_pred[hold]) if hold.any() else float("nan"),
    }


def run(write=False, force=False):
    grid = {c["id"]: c for c in generate_experiment_grid()}
    rows = []
    for exp_id, spec in MODEL_SPECS.items():
        config = build_experiment_config(grid[exp_id])
        X_tr, X_va, X_te, y_train, y_val, y_test = _prepare(config)

        if spec["model"] == "RF":
            model = RandomForestModel(**spec["params"]); model.fit(X_tr, y_train)
        else:
            model = XGBoostModel(**spec["params"]); model.fit(X_tr, y_train, X_va, y_val)

        pred_tr, pred_va, pred_te = model.predict(X_tr), model.predict(X_va), model.predict(X_te)
        r2_tr, r2_va, r2_te = r2_score(y_train, pred_tr), r2_score(y_val, pred_va), r2_score(y_test, pred_te)
        rmse_te, mape_te = calculate_rmse(y_test, pred_te), calculate_mape(y_test, pred_te)

        # Nilai bulan-sebelumnya untuk tiap titik test (aktual, bukan prediksi berantai)
        prior = np.concatenate([np.asarray(y_train, float), np.asarray(y_val, float)])
        full = np.concatenate([prior, np.asarray(y_test, float)])
        y_prev = full[len(prior) - 1: len(full) - 1]
        d = _directional(y_test, pred_te, y_prev)

        rows.append({
            "exp_id": exp_id, "model": spec["model"],
            "RMSE_test": rmse_te, "RMSE_registry": spec["registry_rmse"], "MAPE_test": mape_te,
            "R2_train": r2_tr, "R2_val": r2_va, "R2_test": r2_te,
            "R2_gap_train_test": r2_tr - r2_te,
            "hit_rate_all": d["hit_rate_all"], "hit_rate_change": d["hit_rate_change"],
            "hit_rate_hold": d["hit_rate_hold"], "n_change": d["n_change"], "n_hold": d["n_hold"],
            "RMSE_change_months": d["rmse_change"], "RMSE_hold_months": d["rmse_hold"],
        })

    out = pd.DataFrame(rows)
    print("\n=== DIAGNOSTIK ARAH (regen, himpunan uji) ===")
    for _, r in out.iterrows():
        ok = "OK" if abs(r.RMSE_test - r.RMSE_registry) < 1e-3 else "!! TIDAK COCOK REGISTRY"
        nc = int(round(r.n_change * r.hit_rate_change))
        print(f"[{r.exp_id}/{r.model}] RMSE_test={r.RMSE_test:.4f} (vs registry {r.RMSE_registry:.4f} -> {ok})")
        print(f"   hit_all={r.hit_rate_all:.1%}  hit_change={r.hit_rate_change:.1%} ({nc}/{r.n_change})  "
              f"hit_hold={r.hit_rate_hold:.1%}  RMSE_change={r.RMSE_change_months:.4f}  RMSE_hold={r.RMSE_hold_months:.4f}")
    _verify_and_write(out, write, force)
    return out


def _verify_and_write(out, write, force=False):
    if OUT_PATH.exists():
        old = pd.read_csv(OUT_PATH).set_index("model")
        cur = out.set_index("model")
        cols = [c for c in cur.columns if c in old.columns and c != "exp_id"]
        diff = []
        for m in cur.index:
            for c in cols:
                a, b = cur.loc[m, c], old.loc[m, c]
                if isinstance(a, str) or isinstance(b, str):
                    continue
                if not (np.isnan(a) and np.isnan(b)) and abs(float(a) - float(b)) > TOL:
                    diff.append((m, c, float(b), float(a)))
        dropped = [c for c in old.columns if c not in cur.columns]
        if dropped:
            print(f"\nKolom lama yang DIHAPUS dari skema (mis. U2/random walk): {dropped}")
        if diff:
            print("\n!! ANGKA BERBEDA dari CSV lama. Rincian (lama -> regen):")
            for m, c, b, a in diff:
                print(f"   {m}.{c}: {b} -> {a}")
            if write and not force:
                print("TIDAK menimpa (butuh --force untuk mengadopsi nilai baru).")
                return
        else:
            print("\nSemua angka yang tumpang tindih IDENTIK dengan CSV lama (dalam toleransi).")
    else:
        print(f"\nCSV lama tak ada di {OUT_PATH}; melewati verifikasi.")

    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_PATH, index=False)
        print(f"Ditulis: {OUT_PATH}")
    else:
        print("Mode verifikasi saja (tanpa --write); CSV tidak diubah.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="tulis hasil regen ke CSV")
    ap.add_argument("--force", action="store_true",
                    help="timpa CSV walau angka berbeda (adopsi nilai baru)")
    args = ap.parse_args()
    run(write=args.write, force=args.force)
