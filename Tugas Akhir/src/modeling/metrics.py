import numpy as np


def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error, penalti lebih besar untuk error besar."""
    try:
        return np.sqrt(np.mean((y_true - y_pred)**2))
    except Exception as e:
        print("Error calculating RMSE:", e)
        return None


def calculate_mape(y_true, y_pred):
    try:
        mask = y_true != 0
        if not mask.any():
            print("y_true is all zeros, cannot calculate MAPE")
            return None
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))*100
        return mape
    except Exception as e:
        print("Error calculating MAPE:", e)
        return None


def r2_score(y_true, y_pred):
    """R-squared, proporsi variansi target yang berhasil dijelaskan model."""
    try:
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
    except Exception as e:
        print("Error calculating R2 Score:", e)
        return None


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluasi model dengan 3 metrik: RMSE, MAPE, R²."""
    metrics = {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    print(f"\n--- {model_name} ---")
    print(f"  RMSE:  {metrics['RMSE']:.4f}")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")
    print(f"  R2:    {metrics['R2']:.4f}")
    return metrics


def evaluate_quiet(y_true, y_pred):
    """Sama seperti evaluate_model tetapi tanpa mencetak apa pun.

    Dipakai untuk metrik segmen validasi yang dihitung berdampingan dengan
    metrik uji, agar log fase screening tidak berlipat dua.
    """
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def wilson_interval(k, n, z=1.96):
    """Selang kepercayaan Wilson untuk proporsi k/n.

    Dipakai pada hit-rate arah yang dihitung atas jumlah observasi kecil
    (bulan-perubahan pada himpunan uji hanya sembilan). Selang normal
    (Wald) tidak sahih pada n sekecil itu karena batasnya bisa keluar dari
    [0, 1]; Wilson tetap terkurung di dalam rentang yang sah.

    Mengembalikan (batas_bawah, batas_atas) dalam proporsi, atau
    (None, None) bila n = 0.
    """
    if not n:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def snap_to_grid(values, step=0.25, lo=2.0, hi=13.0):
    """Bulatkan prediksi ke kelipatan kebijakan BI-Rate (default 0,25%).

    BI-Rate hanya valid pada kelipatan 0,25%, jadi prediksi kontinu di-snap ke
    grid kebijakan sebagai lapisan pasca-pemrosesan. Hasil dibatasi ke rentang
    kebijakan yang wajar [lo, hi].
    """
    arr = np.round(np.asarray(values, dtype=float) / step) * step
    return np.clip(arr, lo, hi)


def directional_metrics(y_true, y_pred, y_prev, step=0.25):
    """Metrik keputusan kebijakan: akurasi arah + error per jenis bulan.

    Arah = tanda(nilai - bulan sebelumnya), dibulatkan ke grid agar tahan noise.
    Memisahkan error pada bulan-perubahan (naik/turun) vs bulan-tahan.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)

    dir_true = np.sign(np.round((y_true - y_prev) / step))
    dir_pred = np.sign(np.round((y_pred - y_prev) / step))
    change = dir_true != 0
    return {
        "direction_hit_rate": float(np.mean(dir_true == dir_pred)),
        "n_change_months": int(change.sum()),
        "rmse_change_months": calculate_rmse(y_true[change], y_pred[change]) if change.any() else None,
        "rmse_hold_months": calculate_rmse(y_true[~change], y_pred[~change]) if (~change).any() else None,
    }


def compare_models(results):
    """
    Cetak tabel perbandingan semua model.
    results = {nama_model: {RMSE, MAPE, R2}}
    """
    print(f"\n{'Model':<20} {'RMSE':>8} {'MAPE':>8} {'R2':>8}")
    for name, m in results.items():
        mape_str = f"{m['MAPE']:.2f}%" if m.get('MAPE') is not None else "N/A"
        print(f"{name:<20} {m['RMSE']:>8.4f} {mape_str:>8} {m['R2']:>8.4f}")

    best = min(results, key=lambda x: results[x]["RMSE"])
    print(f"Model terbaik (RMSE terendah): {best}")
    return best
