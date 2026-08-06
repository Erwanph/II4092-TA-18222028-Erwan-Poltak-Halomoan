"""Fungsi-fungsi feature engineering untuk eksperimen prediksi BI-Rate."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def add_lag_features(df, columns, lags, target_col=None):
    """Tambah fitur lag (shift) untuk kolom terpilih; opsional lag target."""
    df_new = df.copy()
    new_features = []

    for col in columns:
        if col not in df_new.columns:
            continue
        for lag in lags:
            new_col = f"{col}_lag{lag}"
            df_new[new_col] = df_new[col].shift(lag)
            new_features.append(new_col)

    if target_col and target_col in df_new.columns:
        for lag in lags:
            new_col = f"{target_col}_lag{lag}"
            df_new[new_col] = df_new[target_col].shift(lag)
            new_features.append(new_col)

    # Drop rows with NaN created by shifting
    df_new = df_new.dropna().reset_index(drop=True)
    return df_new, new_features


def add_rolling_features(df, columns, windows):
    """Tambah rolling mean & std (tren jangka pendek + proksi volatilitas)."""
    df_new = df.copy()
    new_features = []

    for col in columns:
        if col not in df_new.columns:
            continue
        for w in windows:
            mean_col = f"{col}_roll_mean_{w}"
            std_col = f"{col}_roll_std_{w}"
            df_new[mean_col] = df_new[col].rolling(window=w, min_periods=1).mean()
            df_new[std_col] = df_new[col].rolling(window=w, min_periods=1).std()
            new_features.extend([mean_col, std_col])

    # dropna, bukan bfill — mengisi mundur berarti bocor informasi masa depan
    if new_features:
        df_new = df_new.dropna(subset=new_features).reset_index(drop=True)
    return df_new, new_features


def add_diff_features(df, columns):
    """Tambah fitur first-difference (perubahan antarbulan) untuk stasioneritas."""
    df_new = df.copy()
    new_features = []

    for col in columns:
        if col not in df_new.columns:
            continue
        new_col = f"{col}_diff"
        df_new[new_col] = df_new[col].diff()
        new_features.append(new_col)

    df_new = df_new.dropna().reset_index(drop=True)
    return df_new, new_features


def add_target_lag_features(df, target_col, lags):
    """Tambah lag target (komponen autoregresif untuk model ML/DL)."""
    df_new = df.copy()
    new_features = []

    for lag in lags:
        new_col = f"{target_col}_lag{lag}"
        df_new[new_col] = df_new[target_col].shift(lag)
        new_features.append(new_col)

    df_new = df_new.dropna().reset_index(drop=True)
    return df_new, new_features


def add_interaction_features(df, pairs):
    """Tambah fitur interaksi (perkalian) antarpasangan variabel."""
    df_new = df.copy()
    new_features = []

    for col_a, col_b in pairs:
        if col_a not in df_new.columns or col_b not in df_new.columns:
            continue
        new_col = f"{col_a}_x_{col_b}"
        df_new[new_col] = df_new[col_a] * df_new[col_b]
        new_features.append(new_col)

    return df_new, new_features


def select_features_mutual_info(df, target_col, feature_cols, n_features=7):
    """Pilih top-N fitur berdasarkan skor mutual information (nonlinier)."""
    X = df[feature_cols].values
    y = df[target_col].values

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]

    mi = mutual_info_regression(X_clean, y_clean, random_state=42)
    mi_df = pd.DataFrame({
        "feature": feature_cols,
        "mutual_info": mi,
    }).sort_values("mutual_info", ascending=False)

    selected = mi_df.head(n_features)["feature"].tolist()
    print(f"Mutual Information selected {n_features} features: {selected}")
    return selected, mi_df


def select_features_correlation(df, target_col, feature_cols, threshold=None, n_top=None):
    """Pilih fitur berdasarkan |korelasi Pearson| terhadap target."""
    corr = df[feature_cols + [target_col]].corr()[target_col].drop(target_col).abs()
    corr_df = pd.DataFrame({
        "feature": corr.index,
        "abs_correlation": corr.values,
    }).sort_values("abs_correlation", ascending=False)

    if n_top:
        selected = corr_df.head(n_top)["feature"].tolist()
    elif threshold:
        selected = corr_df[corr_df["abs_correlation"] >= threshold]["feature"].tolist()
    else:
        selected = feature_cols

    print(f"Correlation selected {len(selected)} features: {selected}")
    return selected, corr_df


def add_calendar_features(df, date_col=None):
    """Tambah fitur kalender: bulan, kuartal, indikator akhir tahun."""
    df_new = df.copy()
    new_features = []

    if "Bulan" in df_new.columns and "Tahun" in df_new.columns:
        # Bulan/Tahun berupa angka — rakit tanggal agar tak dibaca sebagai epoch
        dates = pd.to_datetime(
            {"year": df_new["Tahun"], "month": df_new["Bulan"], "day": 1},
            errors="coerce",
        )
    elif date_col and date_col in df_new.columns:
        dates = pd.to_datetime(df_new[date_col], errors="coerce")
    elif hasattr(df_new.index, "month"):
        dates = pd.to_datetime(df_new.index, errors="coerce")
    else:
        print("Warning: no date column found for calendar features")
        return df_new, new_features

    dates = pd.Series(dates, index=df_new.index)
    df_new["month"] = dates.dt.month
    df_new["quarter"] = dates.dt.quarter
    df_new["is_year_end"] = (dates.dt.month == 12).astype(int)
    new_features = ["month", "quarter", "is_year_end"]

    return df_new, new_features


def add_policy_reaction_features(df, target_col="BI_Rate_Pct"):
    """Fitur fungsi reaksi kebijakan (Taylor/ITF) + persistensi suku bunga.

    Semua fitur bebas informasi masa depan: lag target & policy_regime murni
    nilai lampau; sisanya memakai nilai prediktor bulan berjalan + lampau.
    """
    d = df.copy()
    nf = []

    def add(name, series):
        d[name] = series
        nf.append(name)

    # persistensi suku bunga (interest-rate smoothing)
    if target_col in d.columns:
        for lag in (1, 2, 3):
            add(f"{target_col}_lag{lag}", d[target_col].shift(lag))
        # arah kebijakan 6 bulan terakhir (mengetat/melonggar/tahan)
        add("policy_regime", np.sign(d[target_col].shift(1) - d[target_col].shift(7)))

    # suku bunga riil = i_{t-1} - inflasi
    if target_col in d.columns and "Inflation_YoY_Pct" in d.columns:
        add("real_rate", d[target_col].shift(1) - d["Inflation_YoY_Pct"])

    # proksi output gap = PDB - tren 12 bulan
    if "GDP_Growth_YoY_Pct" in d.columns:
        trend = d["GDP_Growth_YoY_Pct"].rolling(12, min_periods=3).mean()
        add("output_gap", d["GDP_Growth_YoY_Pct"] - trend)

    # momentum inflasi 3 bulan
    if "Inflation_YoY_Pct" in d.columns:
        add("infl_momentum_3", d["Inflation_YoY_Pct"] - d["Inflation_YoY_Pct"].shift(3))

    # tekanan kurs 3 bulan (%)
    if "USD_IDR_Monthly_Avg" in d.columns:
        add("fx_pressure_3", d["USD_IDR_Monthly_Avg"].pct_change(3) * 100)

    # arah kebijakan The Fed
    if "Federal_Funds_Rate_Pct" in d.columns:
        add("ffr_change", d["Federal_Funds_Rate_Pct"].diff())
        add("fed_cycle", np.sign(d["Federal_Funds_Rate_Pct"]
                                 - d["Federal_Funds_Rate_Pct"].shift(6)))

    d = d.dropna().reset_index(drop=True)
    return d, nf


def apply_feature_engineering(df, config, target_col="BI_Rate_Pct"):
    """Terapkan strategi feature engineering sesuai config eksperimen."""
    if config is None:
        return df, []

    fe_type = config.get("type")

    if fe_type == "lag":
        return add_lag_features(
            df, config["columns"], config["lags"], target_col=None
        )

    elif fe_type == "rolling":
        return add_rolling_features(df, config["columns"], config["windows"])

    elif fe_type == "diff":
        return add_diff_features(df, config["columns"])

    elif fe_type == "target_lag":
        return add_target_lag_features(df, target_col, config["lags"])

    elif fe_type == "policy_reaction":
        return add_policy_reaction_features(df, target_col)

    elif fe_type == "interaction":
        return add_interaction_features(df, config["pairs"])

    elif fe_type == "calendar":
        return add_calendar_features(df)

    elif fe_type == "combined":
        new_features_all = []
        df_new = df.copy()

        if "lag" in config:
            df_new, nf = add_lag_features(
                df_new, config["lag"]["columns"], config["lag"]["lags"]
            )
            new_features_all.extend(nf)

        if "rolling" in config:
            df_new, nf = add_rolling_features(
                df_new, config["rolling"]["columns"], config["rolling"]["windows"]
            )
            new_features_all.extend(nf)

        if "diff" in config:
            df_new, nf = add_diff_features(df_new, config["diff"]["columns"])
            new_features_all.extend(nf)

        if "interaction" in config:
            df_new, nf = add_interaction_features(df_new, config["interaction"]["pairs"])
            new_features_all.extend(nf)

        if "calendar" in config:
            df_new, nf = add_calendar_features(df_new)
            new_features_all.extend(nf)

        return df_new, new_features_all

    elif fe_type == "pca":
        return df, []

    elif fe_type == "mutual_info_selection":
        return df, []

    else:
        print(f"Unknown feature engineering type: {fe_type}")
        return df, []

