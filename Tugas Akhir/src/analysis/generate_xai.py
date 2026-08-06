"""Regenerasi artefak XAI global (tabel SHAP + gambar summary/waterfall/LIME)
dari model tree-based terbaik di registry. Keluarannya di-input Bab VI.
Jalankan tiap registry berubah: python -m src.analysis.generate_xai
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from src.modeling.config import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from src.modeling.data_prep import handle_missing_values
from src.modeling.machine_learning import RandomForestModel, XGBoostModel
from src.experiments.feature_engineering import (
    apply_feature_engineering, select_features_mutual_info,
)
from src.experiments.scenario_config import build_experiment_config, DEFAULT_HYPERPARAMS
from src.experiments.best_registry import load_registry
from src.xai.explainers import SHAPExplainer, LIMEExplainer

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT.parent / "Tugas Akhir - Dokumen"
DOC_TABLES = DOC / "tables"
DOC_IMAGES = DOC / "images"
ANALYSIS = ROOT / "results" / "experiments" / "analysis"
DATA_2026 = DATASET_PATH.parent / "dataset_2026.csv"
REGISTRY_PATH = ROOT / "results" / "experiments" / "best_results_registry.json"
FALLBACK = {"id": "E062", "imputation": "ffill", "features": "domain_itf",
            "scaler": "standard", "feature_engineering": "lag"}
# Bulan kontrol untuk penjelasan lokal (tahun, bulan). Disamakan dengan bahan
# penilaian pakar pada Lampiran E agar seluruh penjelasan lokal di Bab VI
# merujuk observasi yang sama.
BULAN_KONTROL = (2025, 7)


def _parse_hp(hp):
    """Kembalikan dict hyperparameter dari registry (bisa dict atau string)."""
    if isinstance(hp, dict):
        return hp
    if isinstance(hp, str):
        import ast
        try:
            val = ast.literal_eval(hp)
            return val if isinstance(val, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def pick_model():
    """Model tree-based terbaik dari registry: (nama, combo, metrik, hyperparams)."""
    reg = load_registry(REGISTRY_PATH)
    cand = {m: reg[m] for m in ("XGBoost", "RF") if m in reg}
    if not cand:
        print("Registry kosong; memakai konfigurasi cadangan (jalankan pipeline untuk hasil final).")
        return "XGBoost", FALLBACK, None, {}
    name = min(cand, key=lambda m: cand[m].get("RMSE", float("inf")))
    info = cand[name]
    combo = {"id": info.get("experiment_id", "?"), **info.get("combination", {})}
    metrics = {k: info.get(k) for k in ("RMSE", "MAPE", "R2")}
    return name, combo, metrics, _parse_hp(info.get("hyperparams"))


def prepare(combo):
    """Gabung historis + holdout 2026, imputasi, rekayasa fitur."""
    hist = pd.read_csv(DATASET_PATH)
    hold = pd.read_csv(DATA_2026)
    df = pd.concat([hist, hold], ignore_index=True)

    num = [c for c in df.columns if c not in ("ID", "Bulan", "Tahun")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")

    config = build_experiment_config(combo)
    df = handle_missing_values(df, method=config["missing_method"],
                               knn_neighbors=config.get("knn_neighbors", 5))
    new_feats = []
    if config.get("feature_engineering"):
        df, new_feats = apply_feature_engineering(df, config["feature_engineering"], TARGET_COLUMN)

    feats = list(config.get("feature_columns", FEATURE_COLUMNS))
    if config.get("selection_method") == "mutual_info":
        feats, _ = select_features_mutual_info(df, TARGET_COLUMN, feats,
                                               n_features=config.get("n_select", 7))
    feats = [c for c in feats + new_feats if c in df.columns]
    return df, feats, config


def _build_model(name, hp):
    """Bangun model dengan hyperparameter hasil tuning (fallback ke default)."""
    base = dict(DEFAULT_HYPERPARAMS.get(name, {}))
    base.update(hp or {})
    if name == "RF":
        keys = ["n_estimators", "max_depth", "min_samples_split",
                "min_samples_leaf", "max_features"]
        return RandomForestModel(random_state=42,
                                 **{k: base[k] for k in keys if k in base})
    keys = ["n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_weight"]
    return XGBoostModel(random_state=42, **{k: base[k] for k in keys if k in base})

# Nama tampilan fitur (Indonesia) untuk tabel & gambar. Fitur tak terdaftar
# memakai nama mentahnya (di-escape) sebagai cadangan.
DISPLAY = {
    "BI_Rate_Pct_lag1": "BI-Rate t-1 (Persistensi)",
    "BI_Rate_Pct_lag2": "BI-Rate t-2",
    "BI_Rate_Pct_lag3": "BI-Rate t-3",
    "policy_regime": "Arah Kebijakan Suku Bunga (Mengetat/Melonggar)",
    "real_rate": "Suku Bunga Riil",
    "output_gap": "Output Gap",
    "infl_momentum_3": "Momentum Inflasi (3 Bulan)",
    "fx_pressure_3": "Tekanan Nilai Tukar (3 Bulan)",
    "ffr_change": "Perubahan Federal Funds Rate",
    "fed_cycle": "Siklus The Fed",
    "Inflation_YoY_Pct": "Inflasi Year-on-Year (%)",
    "Inflation_Gap_Pct": "Inflation Gap (%)",
    "GDP_Growth_YoY_Pct": "Pertumbuhan PDB YoY (%)",
    "USD_IDR_Monthly_Avg": "Kurs USD/IDR (Rata-rata Bulanan)",
    "M2_Triliun_Rp": "M2, Jumlah Uang Beredar (Triliun Rp)",
    "Credit_Growth_YoY_Pct": "Pertumbuhan Kredit YoY (%)",
    "Federal_Funds_Rate_Pct": "Federal Funds Rate (%)",
    "IHSG_End_of_Month": "IHSG (End of Month)",
    "Foreign_Reserves_Miliar_USD": "Cadangan Devisa (Miliar USD)",
    "Oil_Price_Brent_USD_per_Bbl": "Harga Minyak Brent (USD/Bbl)",
    "Gold_Price_USD_per_Oz": "Harga Emas (USD/Oz)",
    "VIX_Volatility_Index": "Indeks Volatilitas (VIX)",
}


def _disp(name):
    return DISPLAY.get(name, name)


def _num(x):
    return f"{x:.4f}".replace(".", ",")


def _esc(t):
    return str(t).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _write_shap_table(imp_df):
    """Tulis tabel peringkat global feature importance (di-input Bab VI)."""
    rows = ""
    for rank, (_, r) in enumerate(imp_df.iterrows(), start=1):
        rows += (f"{rank} & {_esc(_disp(r['feature']))} & "
                 f"{_num(r['mean_abs_shap'])} \\\\\n\\hline\n")
    tex = (
        "% Dihasilkan otomatis oleh src/analysis/generate_xai.py\n"
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Peringkat \\textit{Global Feature Importance} Berdasarkan SHAP}\n"
        "\\label{tbl:shap-importance}\n"
        "\\begin{tabular}{|c|p{7cm}|c|}\n\\hline\n"
        "\\textbf{Peringkat} & \\textbf{Fitur} & "
        "\\textbf{$\\overline{|\\text{SHAP}|}$} \\\\\n\\hline\n"
        f"{rows}"
        "\\end{tabular}\n\\end{table}\n"
    )
    (DOC_TABLES / "Tabel_SHAP_Importance.tex").write_text(tex, encoding="utf-8")


def main():
    DOC_TABLES.mkdir(parents=True, exist_ok=True)
    DOC_IMAGES.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    name, combo, _, tuned_hp = pick_model()
    if name not in ("RF", "XGBoost"):
        raise SystemExit("Model interpretasi harus tree-based (RF/XGBoost).")
    print(f"Model interpretasi XAI: {name} (kombinasi {combo.get('id')})")

    df, feats, _ = prepare(combo)
    is26 = df["Tahun"].astype(int) == 2026
    train, hold = df[~is26], df[is26]
    if hold.empty:
        raise SystemExit("Tidak ada baris 2026; jalankan data/build_datasets.py dulu.")

    # Model tree dilatih pada fitur MENTAH (skala ekonomi asli, A4).
    X_tr = train[feats].values.astype(float)
    X_ho = hold[feats].values.astype(float)
    y_tr = train[TARGET_COLUMN].values

    model = _build_model(name, tuned_hp)
    if name == "XGBoost":
        cut = int(len(X_tr) * 0.85)
        model.fit(X_tr[:cut], y_tr[:cut], X_tr[cut:], y_tr[cut:])
    else:
        model.fit(X_tr, y_tr)

    disp_names = [_disp(f) for f in feats]

    # --- SHAP global importance ---
    shap_exp = SHAPExplainer(model.model, model_type="tree")
    shap_exp.fit(X_tr)
    shap_exp.explain(X_tr)
    imp = shap_exp.get_global_importance(feats).reset_index(drop=True)
    imp_disp = imp.assign(feature_display=imp["feature"].map(_disp))
    imp_disp.to_csv(ANALYSIS / "shap_global_importance.csv", index=False)
    _write_shap_table(imp)
    print("\nPeringkat global mean(|SHAP|):")
    for rank, (_, r) in enumerate(imp.iterrows(), start=1):
        print(f"  {rank:2d}. {_disp(r['feature']):42s} {r['mean_abs_shap']:.4f}")

    # --- Gambar SHAP (summary global + waterfall instans bulan kontrol) ---
    # Instans lokal sengaja memakai bulan kontrol yang sama dengan bahan
    # penilaian pakar (Juli 2025), bukan bulan holdout, agar seluruh penjelasan
    # lokal di Bab VI merujuk observasi yang sama.
    shap_exp.plot_summary(X_tr, disp_names,
                          save_path=str(DOC_IMAGES / "shap_summary.png"))
    sel = (train["Tahun"].astype(int) == BULAN_KONTROL[0]) & \
          (train["Bulan"].astype(int) == BULAN_KONTROL[1])
    if not sel.any():
        raise SystemExit(f"Bulan kontrol {BULAN_KONTROL} tidak ada pada data.")
    x_lokal = train.loc[sel, feats].values.astype(float)[0]
    pred_lokal = float(model.predict(x_lokal.reshape(1, -1))[0])
    shap_exp.plot_waterfall(x_lokal, disp_names,
                            save_path=str(DOC_IMAGES / "shap_waterfall.png"))
    print(f"\nInstans lokal = {BULAN_KONTROL[1]:02d}/{BULAN_KONTROL[0]}, "
          f"prediksi mentah = {pred_lokal:.4f} "
          f"(snap 0,25 -> {round(pred_lokal * 4) / 4:.2f})")

    # --- LIME pada instans yang sama ---
    # LIMEExplainer sengaja dibangun ulang sebelum menggambar. explain_instance
    # memajukan RNG internal explainer, sehingga memakai satu objek untuk
    # menghitung R^2 lalu menggambar akan menghasilkan dua tarikan perturbasi
    # yang berbeda. Dengan dua objek berseed sama, angka dan gambar berasal
    # dari perturbasi yang identik.
    lime_res = LIMEExplainer(X_tr, disp_names).explain_instance(
        model.model.predict, x_lokal)
    LIMEExplainer(X_tr, disp_names).plot_explanation(
        model.model.predict, x_lokal,
        save_path=str(DOC_IMAGES / "lime_explanation.png"))
    print(f"LIME R^2 (model surrogat lokal) = {lime_res['r2_score']:.4f}")
    print("Kontribusi LIME teratas:")
    for f, v in sorted(lime_res["contributions"].items(),
                       key=lambda kv: abs(kv[1]), reverse=True)[:6]:
        print(f"  {f:52s} {v:+.4f}")

    # --- Sebaran R^2 surrogat LIME pada seluruh bulan uji ---
    # Angka R^2 satu observasi tidak dapat dibaca sendirian: kecocokan surrogat
    # linear lokal ternyata sangat bergantung pada letak observasinya. Sebaran
    # ini yang dilaporkan di Bab VI, bukan satu angka tunggal.
    uji = train[train["Tahun"].astype(int).between(2023, 2025)]
    rows = []
    for _, r in uji.iterrows():
        x = r[feats].values.astype(float)
        skor = LIMEExplainer(X_tr, disp_names).explain_instance(
            model.model.predict, x)["r2_score"]
        rows.append({"tahun": int(r["Tahun"]), "bulan": int(r["Bulan"]),
                     "lime_r2": skor})
    sebaran = pd.DataFrame(rows)
    sebaran.to_csv(ANALYSIS / "lime_r2_per_bulan.csv", index=False)
    print(f"\nSebaran R^2 LIME atas {len(sebaran)} bulan uji: "
          f"min={sebaran.lime_r2.min():.4f} median={sebaran.lime_r2.median():.4f} "
          f"maks={sebaran.lime_r2.max():.4f} "
          f"(di atas 0,5: {(sebaran.lime_r2 > 0.5).sum()} bulan)")
    print("\nArtefak XAI selesai diperbarui:")
    print(f"  - {DOC_TABLES / 'Tabel_SHAP_Importance.tex'}")
    print(f"  - {ANALYSIS / 'shap_global_importance.csv'}")
    print(f"  - {DOC_IMAGES}\\shap_summary.png | shap_waterfall.png | lime_explanation.png")


if __name__ == "__main__":
    main()
