"""Konfigurasi eksperimen faktorial prediksi BI-Rate.

Ruang eksperimen: imputasi (4) x seleksi fitur (4) x scaler (3) x
feature engineering (8) = 384 kombinasi, dijalankan lima fase
(screening -> analisis faktor -> focused -> tuning -> final).
"""

from itertools import product as itertools_product
from src.modeling.config import FEATURE_COLUMNS

ALL_FEATURES = FEATURE_COLUMNS.copy()
ALL_MODELS = [
    "ARIMA", "VAR",
    "RF", "XGBoost",
    "LSTM", "BiLSTM",
]
SCREENING_MODELS = ["RF", "XGBoost"]

DEFAULT_HYPERPARAMS = {
    "ARIMA": {"order": (1, 1, 1)},
    "VAR": {"maxlags": 12, "ic": "aic"},
    "RF": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
    },
    "XGBoost": {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "random_state": 42,
    },
    "LSTM": {
        "units": 64,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 30,
        "sequence_length": 8,
    },
    "BiLSTM": {
        "units": 64,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 30,
        "sequence_length": 8,
    },
}


DOMAIN_ITF_FEATURES = [
    "Inflation_YoY_Pct", "Inflation_Gap_Pct", "GDP_Growth_YoY_Pct",
    "Federal_Funds_Rate_Pct", "USD_IDR_Monthly_Avg",
    "M2_Triliun_Rp", "Credit_Growth_YoY_Pct",
]

CORR_TOP6_FEATURES = [
    "Inflation_YoY_Pct", "Inflation_Gap_Pct", "Federal_Funds_Rate_Pct",
    "GDP_Growth_YoY_Pct", "USD_IDR_Monthly_Avg",
    "Oil_Price_Brent_USD_per_Bbl",
]


IMPUTATION_CANDIDATES = {
    "interpolate": {
        "label": "Interpolasi Linear",
        "config": {"missing_method": "interpolate"},
        "rationale": "Mempertahankan tren temporal antar titik data.",
    },
    "ffill": {
        "label": "Forward Fill",
        "config": {"missing_method": "ffill"},
        "rationale": "Propagasi kronologis — tidak menggunakan informasi masa depan.",
    },
    "mean": {
        "label": "Mean Imputation",
        "config": {"missing_method": "mean"},
        "rationale": "Metode paling sederhana — benchmark statistik.",
    },
    "knn": {
        "label": "KNN Imputation (k=5)",
        "config": {"missing_method": "knn", "knn_neighbors": 5},
        "rationale": "Multivariat — menggunakan hubungan antar fitur.",
    },
}

FEATURE_CANDIDATES = {
    "all": {
        "label": "Semua Fitur (12)",
        "config": {"columns": ALL_FEATURES},
        "rationale": "Baseline tanpa seleksi — model menentukan fitur penting.",
    },
    "domain_itf": {
        "label": "Domain Expert — ITF (7)",
        "config": {"columns": DOMAIN_ITF_FEATURES},
        "rationale": "Teori Inflation Targeting Framework Bank Indonesia.",
    },
    "corr_top6": {
        "label": "Top-6 Korelasi Pearson",
        "config": {"columns": CORR_TOP6_FEATURES},
        "rationale": "Korelasi linear tertinggi terhadap BI-Rate.",
    },
    "mutual_info": {
        "label": "Mutual Info Top-7",
        "config": {
            "columns": ALL_FEATURES,
            "selection_method": "mutual_info",
            "n_select": 7,
        },
        "rationale": "Seleksi non-linear — menangkap hubungan kompleks.",
    },
}

SCALER_CANDIDATES = {
    "standard": {
        "label": "StandardScaler (Z-Score)",
        "config": {"scaler": "standard"},
        "rationale": "Transformasi distribusi normal (mean=0, std=1).",
    },
    "minmax": {
        "label": "MinMaxScaler [0, 1]",
        "config": {"scaler": "minmax"},
        "rationale": "Mempertahankan distribusi asli, cocok untuk neural networks.",
    },
    "robust": {
        "label": "RobustScaler (IQR)",
        "config": {"scaler": "robust"},
        "rationale": "Tahan outlier (krisis ekonomi, pandemi).",
    },
}

FE_CANDIDATES = {
    "none": {
        "label": "Tanpa Feature Engineering",
        "config": None,
        "rationale": "Baseline — fitur original saja.",
    },
    "lag": {
        "label": "Lag Features (t-1, t-3, t-6)",
        "config": {
            "type": "lag",
            "lags": [1, 3, 6],
            "columns": ["Inflation_YoY_Pct", "Federal_Funds_Rate_Pct",
                         "USD_IDR_Monthly_Avg", "GDP_Growth_YoY_Pct"],
        },
        "rationale": "Menangkap delayed response kebijakan moneter.",
    },
    "rolling": {
        "label": "Rolling Mean & Std (3, 6)",
        "config": {
            "type": "rolling",
            "windows": [3, 6],
            "columns": ["Inflation_YoY_Pct", "USD_IDR_Monthly_Avg",
                         "IHSG_End_of_Month", "Oil_Price_Brent_USD_per_Bbl"],
        },
        "rationale": "Tren kuartalan dan semesteran + volatilitas.",
    },
    "lag_rolling": {
        "label": "Lag (1,3) + Rolling (6)",
        "config": {
            "type": "combined",
            "lag": {
                "lags": [1, 3],
                "columns": ["Inflation_YoY_Pct", "Federal_Funds_Rate_Pct"],
            },
            "rolling": {
                "windows": [6],
                "columns": ["USD_IDR_Monthly_Avg", "IHSG_End_of_Month"],
            },
        },
        "rationale": "Gabungan titik waktu spesifik + tren rata-rata.",
    },
    "diff": {
        "label": "First Difference",
        "config": {
            "type": "diff",
            "columns": ["Inflation_YoY_Pct", "USD_IDR_Monthly_Avg",
                         "M2_Triliun_Rp", "Credit_Growth_YoY_Pct",
                         "Federal_Funds_Rate_Pct"],
        },
        "rationale": "Mengubah level ke rate of change untuk stasioneritas.",
    },
    "interaction": {
        "label": "Fitur Interaksi Ekonomi",
        "config": {
            "type": "interaction",
            "pairs": [
                ("Inflation_YoY_Pct", "Federal_Funds_Rate_Pct"),
                ("GDP_Growth_YoY_Pct", "Inflation_YoY_Pct"),
                ("USD_IDR_Monthly_Avg", "Federal_Funds_Rate_Pct"),
            ],
        },
        "rationale": "Menangkap efek interaksi variabel makroekonomi.",
    },
    "calendar": {
        "label": "Fitur Kalender (Bulan, Kuartal)",
        "config": {
            "type": "calendar",
        },
        "rationale": "Menangkap pola musiman pada suku bunga.",
    },
    "policy_reaction": {
        "label": "Fungsi Reaksi Kebijakan (Taylor/ITF) + Persistensi",
        "config": {
            "type": "policy_reaction",
        },
        "rationale": ("Lag BI-Rate (smoothing), suku bunga riil, output gap, "
                      "momentum inflasi, tekanan kurs, dan arah Fed sebagai "
                      "operasionalisasi fungsi reaksi kebijakan moneter."),
    },
}

TUNING_CONFIGS = {
    "RF": {
        "tuning_method": "optuna",
        "n_trials": 20,
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int_or_none", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical",
                             "choices": ["sqrt", "log2", 0.5, 0.7, 1.0]},
        },
    },
    "XGBoost": {
        "tuning_method": "optuna",
        "n_trials": 20,
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.3},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {"type": "float_log", "low": 1e-8, "high": 10.0},
            "reg_lambda": {"type": "float_log", "low": 1e-8, "high": 10.0},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
        },
    },
    "LSTM": {
        "tuning_method": "optuna",
        "n_trials": 15,
        "search_space": {
            "units": {"type": "categorical", "choices": [16, 32, 64, 128]},
            "dropout": {"type": "float", "low": 0.1, "high": 0.5},
            "sequence_length": {"type": "categorical", "choices": [3, 5, 6, 8, 10]},
            "learning_rate": {"type": "float_log", "low": 1e-4, "high": 1e-2},
            "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
            "l2": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]},
            "l1": {"type": "categorical", "choices": [0.0, 1e-4]},
        },
    },
    "BiLSTM": {
        "tuning_method": "optuna",
        "n_trials": 15,
        "search_space": {
            "units": {"type": "categorical", "choices": [16, 32, 64, 128]},
            "dropout": {"type": "float", "low": 0.1, "high": 0.5},
            "sequence_length": {"type": "categorical", "choices": [3, 5, 6, 8, 10]},
            "learning_rate": {"type": "float_log", "low": 1e-4, "high": 1e-2},
            "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
            "l2": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]},
            "l1": {"type": "categorical", "choices": [0.0, 1e-4]},
        },
    },
    "ARIMA": {
        "tuning_method": "auto_arima",
        "max_p": 5, "max_d": 2, "max_q": 5,
        "seasonal": False,
        "information_criterion": "aic",
        "stepwise": True,
    },
    "VAR": {
        "tuning_method": "grid_search",
        "param_grid": {
            "maxlags": [4, 6, 8, 10, 12, 15],
            "ic": ["aic", "bic", "hqic"],
        },
    },
}


def generate_experiment_grid():
    """Bangkitkan grid faktorial penuh (384 kombinasi ber-ID E001..E384)."""
    experiments = []
    idx = 1

    for imp_key, feat_key, sc_key, fe_key in itertools_product(
        IMPUTATION_CANDIDATES.keys(),
        FEATURE_CANDIDATES.keys(),
        SCALER_CANDIDATES.keys(),
        FE_CANDIDATES.keys(),
    ):
        exp = {
            "id": f"E{idx:03d}",
            "imputation": imp_key,
            "features": feat_key,
            "scaler": sc_key,
            "feature_engineering": fe_key,
            "label": (
                f"{IMPUTATION_CANDIDATES[imp_key]['label']} + "
                f"{FEATURE_CANDIDATES[feat_key]['label']} + "
                f"{SCALER_CANDIDATES[sc_key]['label']} + "
                f"{FE_CANDIDATES[fe_key]['label']}"
            ),
        }
        experiments.append(exp)
        idx += 1

    return experiments


def build_experiment_config(combo):
    """Ubah satu kombinasi menjadi konfigurasi pipeline yang siap dijalankan."""
    imp = IMPUTATION_CANDIDATES[combo["imputation"]]
    feat = FEATURE_CANDIDATES[combo["features"]]
    sc = SCALER_CANDIDATES[combo["scaler"]]
    fe = FE_CANDIDATES[combo["feature_engineering"]]

    config = {
        "missing_method": imp["config"]["missing_method"],
        "feature_columns": feat["config"]["columns"],
        "scaler": sc["config"]["scaler"],
        "feature_engineering": fe["config"],
        "hyperparams": DEFAULT_HYPERPARAMS,
    }

    if "knn_neighbors" in imp["config"]:
        config["knn_neighbors"] = imp["config"]["knn_neighbors"]

    if feat["config"].get("selection_method") == "mutual_info":
        config["selection_method"] = "mutual_info"
        config["n_select"] = feat["config"].get("n_select", 7)

    config["_factors"] = {
        "imputation": combo["imputation"],
        "features": combo["features"],
        "scaler": combo["scaler"],
        "feature_engineering": combo["feature_engineering"],
    }

    return config


def get_experiment_space_summary():
    """Ringkasan ukuran ruang eksperimen."""
    n_imp = len(IMPUTATION_CANDIDATES)
    n_feat = len(FEATURE_CANDIDATES)
    n_sc = len(SCALER_CANDIDATES)
    n_fe = len(FE_CANDIDATES)
    total = n_imp * n_feat * n_sc * n_fe

    return {
        "imputation": n_imp,
        "features": n_feat,
        "scaler": n_sc,
        "feature_engineering": n_fe,
        "total_combinations": total,
    }


def list_all_experiments():
    """Cetak ringkasan ruang eksperimen."""
    summary = get_experiment_space_summary()

    print(f"\nExperiment Space - Factorial Design")
    print("-" * 50)

    print(f"\n  Imputation ({len(IMPUTATION_CANDIDATES)} candidates):")
    for k, v in IMPUTATION_CANDIDATES.items():
        print(f"    - {k:<15} {v['label']}")

    print(f"\n  Features ({len(FEATURE_CANDIDATES)} candidates):")
    for k, v in FEATURE_CANDIDATES.items():
        print(f"    - {k:<15} {v['label']}")

    print(f"\n  Scaler ({len(SCALER_CANDIDATES)} candidates):")
    for k, v in SCALER_CANDIDATES.items():
        print(f"    - {k:<15} {v['label']}")

    print(f"\n  Feature Engineering ({len(FE_CANDIDATES)} candidates):")
    for k, v in FE_CANDIDATES.items():
        print(f"    - {k:<15} {v['label']}")

    print(f"\n  Total combinations: {summary['total_combinations']}")
    print(f"  = {summary['imputation']} x {summary['features']} "
          f"x {summary['scaler']} x {summary['feature_engineering']}")

    print(f"\n  Phases:")
    print(f"    Phase 1 - Screening:  {summary['total_combinations']} combinations "
          f"x {len(SCREENING_MODELS)} models = {summary['total_combinations'] * len(SCREENING_MODELS)} runs")
    print(f"    Phase 2 - Analysis:   Factor analysis on screening results")
    print(f"    Phase 3 - Focused:    Top-K combinations x {len(ALL_MODELS)} models")
    print(f"    Phase 4 - Tuning:     Optuna/GridSearch on best combinations")
    print(f"    Phase 5 - Final:      Baseline vs Optimized comparison")
