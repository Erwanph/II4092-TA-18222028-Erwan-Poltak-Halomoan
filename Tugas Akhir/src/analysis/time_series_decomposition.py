"""Dekomposisi STL deret waktu (trend/seasonal/residual) untuk EDA Bab III."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from pathlib import Path


def run_decomposition(df, column, period=12, output_dir=None, robust=True):
    """Dekomposisi STL satu kolom; robust=True agar tahan krisis 2008/2020."""
    series = df[column].dropna()
    if len(series) < 2 * period:
        print(f"  Skipping {column}: data terlalu pendek ({len(series)} < {2*period})")
        return None

    stl = STL(series, period=period, robust=robust)
    result = stl.fit()

    stats = {
        "column": column,
        "n_obs": len(series),
        "trend_mean": float(result.trend.mean()),
        "trend_range": float(result.trend.max() - result.trend.min()),
        "seasonal_amplitude": float(result.seasonal.max() - result.seasonal.min()),
        "seasonal_ratio": float(result.seasonal.std() / series.std()) if series.std() > 0 else 0,
        "residual_std": float(result.resid.std()),
        "residual_ratio": float(result.resid.std() / series.std()) if series.std() > 0 else 0,
    }

    if output_dir:
        _plot_decomposition(result, series, column, output_dir)

    return {
        "trend": result.trend.values,
        "seasonal": result.seasonal.values,
        "residual": result.resid.values,
        "stats": stats,
    }


def _plot_decomposition(result, original, column, output_dir):
    """Simpan visualisasi 4-panel: original + trend + seasonal + residual."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(original.values, color="#2c3e50", linewidth=1.2)
    axes[0].set_title(f"{column} — Original", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Nilai")

    axes[1].plot(result.trend.values, color="#e74c3c", linewidth=1.4)
    axes[1].set_title("Trend (LOESS)", fontsize=11)
    axes[1].set_ylabel("Trend")

    axes[2].plot(result.seasonal.values, color="#3498db", linewidth=1)
    axes[2].set_title("Seasonal Component", fontsize=11)
    axes[2].set_ylabel("Seasonal")
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    axes[3].scatter(range(len(result.resid)), result.resid.values,
                    s=8, alpha=0.6, color="#27ae60")
    axes[3].set_title("Residual (Noise)", fontsize=11)
    axes[3].set_ylabel("Residual")
    axes[3].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[3].set_xlabel("Observasi (bulan)")

    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = column.replace(" ", "_").replace("/", "_")
    fig.savefig(output_dir / f"decomposition_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def decompose_all(df, columns, target_col, output_dir=None, period=12):
    """
    Jalankan dekomposisi untuk target dan sejumlah fitur sekaligus.

    Returns:
        dict dengan key = nama kolom, value = hasil dekomposisi
    """
    all_cols = [target_col] + [c for c in columns if c != target_col]
    results = {}

    print("\n--- Dekomposisi Time Series (STL) ---")
    for col in all_cols:
        if col not in df.columns:
            continue
        print(f"  Memproses: {col}")
        res = run_decomposition(df, col, period=period, output_dir=output_dir)
        if res:
            results[col] = res
            s = res["stats"]
            print(f"    Seasonal amplitude: {s['seasonal_amplitude']:.4f}, "
                  f"Residual ratio: {s['residual_ratio']:.2%}")

    if output_dir and results:
        _write_decomposition_report(results, target_col, output_dir)

    return results


def _write_decomposition_report(results, target_col, output_dir):
    """Tulis ringkasan dekomposisi ke file markdown."""
    output_dir = Path(output_dir)
    md_lines = ["# Hasil Dekomposisi Time Series (STL)\n"]
    md_lines.append("Metode: Seasonal-Trend Decomposition using LOESS (STL), "
                     "robust=True\n")
    md_lines.append(f"Jumlah variabel: {len(results)}\n\n")

    md_lines.append("| Variabel | Trend Range | Seasonal Amp. | "
                     "Residual Std | Seasonal Ratio | Residual Ratio |\n")
    md_lines.append("|----------|-------------|---------------|"
                     "-------------|----------------|----------------|\n")

    for col, res in results.items():
        s = res["stats"]
        marker = " **[TARGET]**" if col == target_col else ""
        md_lines.append(
            f"| {col}{marker} | {s['trend_range']:.4f} | "
            f"{s['seasonal_amplitude']:.4f} | {s['residual_std']:.4f} | "
            f"{s['seasonal_ratio']:.2%} | {s['residual_ratio']:.2%} |\n"
        )

    # Interpretasi untuk target
    if target_col in results:
        ts = results[target_col]["stats"]
        md_lines.append(f"\n## Interpretasi {target_col}\n\n")

        if ts["seasonal_ratio"] < 0.10:
            md_lines.append("- **Komponen seasonal sangat kecil** — "
                           "pola musiman tidak dominan pada BI-Rate. "
                           "Ini masuk akal karena keputusan suku bunga BI "
                           "lebih dipengaruhi kondisi makro daripada kalendar.\n")
        else:
            md_lines.append(f"- **Komponen seasonal terdeteksi** dengan rasio "
                           f"{ts['seasonal_ratio']:.2%} terhadap total variasi.\n")

        md_lines.append(f"- **Trend mendominasi** variasi BI-Rate "
                       f"(range: {ts['trend_range']:.2f} poin), "
                       f"mencerminkan kebijakan moneter jangka panjang.\n")

        md_lines.append(f"- **Residual** (noise) memiliki std={ts['residual_std']:.4f}, "
                       f"ratio={ts['residual_ratio']:.2%}. "
                       f"Bagian inilah yang menjadi target model machine learning untuk "
                       f"dipelajari.\n")

    with open(output_dir / "decomposition_report.md", "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"  Report disimpan: {output_dir / 'decomposition_report.md'}")


if __name__ == "__main__":
    from src.modeling.config import DATASET_PATH, TARGET_COLUMN
    from src.modeling.data_prep import load_dataset, handle_missing_values

    df = load_dataset(str(DATASET_PATH))
    df = handle_missing_values(df)

    # Pilih fitur utama yang relevan untuk didekomposisi
    key_features = [
        "Inflation_YoY_Pct", "Federal_Funds_Rate_Pct",
        "USD_IDR_Monthly_Avg", "GDP_Growth_YoY_Pct",
        "IHSG_End_of_Month", "Oil_Price_Brent_USD_per_Bbl",
    ]

    results = decompose_all(
        df, key_features, TARGET_COLUMN,
        output_dir="results/decomposition",
    )
