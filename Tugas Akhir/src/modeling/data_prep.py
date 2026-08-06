import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

try:
    from sklearn.impute import KNNImputer
except ImportError:
    KNNImputer = None


def load_dataset(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df, method="interpolate", **kwargs):
    """Imputasi kolom numerik: interpolate/spline/ffill/bfill/mean/median/knn."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    skip_cols = ["ID", "Bulan", "Tahun"]
    cols = [c for c in numeric_cols if c not in skip_cols]

    if method == "interpolate":
        df_clean[cols] = df_clean[cols].interpolate(method="linear", limit_direction="both")
    elif method == "spline":
        df_clean[cols] = df_clean[cols].interpolate(method="spline", order=3, limit_direction="both")
    elif method == "ffill":
        df_clean[cols] = df_clean[cols].ffill()
    elif method == "bfill":
        df_clean[cols] = df_clean[cols].bfill()
    elif method == "ffill_bfill":
        df_clean[cols] = df_clean[cols].ffill().bfill()
    elif method == "mean":
        df_clean[cols] = df_clean[cols].fillna(df_clean[cols].mean())
    elif method == "median":
        df_clean[cols] = df_clean[cols].fillna(df_clean[cols].median())
    elif method == "knn":
        if KNNImputer is None:
            print("KNNImputer not available, falling back to interpolate")
            df_clean[cols] = df_clean[cols].interpolate(method="linear", limit_direction="both")
        else:
            n_neighbors = kwargs.get("knn_neighbors", 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_clean[cols] = pd.DataFrame(
                imputer.fit_transform(df_clean[cols]),
                columns=cols, index=df_clean.index
            )
    else:
        print(f"Unknown method '{method}', returning original dataframe")
        return df_clean

    remaining = df_clean[cols].isnull().sum().sum()
    print(f"Missing values handled (method={method}). Remaining NaN: {remaining}")
    return df_clean


def time_series_split(df, target_col, feature_cols, test_size=0.2, val_size=0.25):
    """Split kronologis tanpa shuffle: train 60% -> val 20% -> test 20%."""
    df_clean = df.dropna(subset=[target_col])
    n = len(df_clean)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))

    X_train = df_clean[feature_cols].iloc[:val_idx].values
    X_val = df_clean[feature_cols].iloc[val_idx:test_idx].values
    X_test = df_clean[feature_cols].iloc[test_idx:].values

    y_train = df_clean[target_col].iloc[:val_idx].values
    y_val = df_clean[target_col].iloc[val_idx:test_idx].values
    y_test = df_clean[target_col].iloc[test_idx:].values

    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, scaler_type="standard"):
    """Skalakan fitur (standard/minmax/robust); fit hanya pada data latih."""
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        print(f"Unknown scaler '{scaler_type}', defaulting to StandardScaler")
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"Features scaled ({scaler_type})")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def create_sequences(X, y, seq_length):
    """Create 3D sequences for LSTM/BiLSTM input."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)