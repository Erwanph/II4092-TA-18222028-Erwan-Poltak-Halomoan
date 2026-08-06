"""Data Understanding (CRISP-DM fase 2) untuk prediksi BI-Rate.

Membaca dataset mentah (dataset_raw_integrated.csv, Juli 2005 - Mei 2026) dan
menghasilkan seluruh plot eksplorasi ke folder data_understanding/plots/.
Run: python data_understanding/generate_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Setup
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})
sns.set_style("whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Gunakan dataset RAW (belum preprocessing) untuk Data Understanding
# Dataset ini masih mengandung missing values asli (GDP, Credit)
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_raw_integrated.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load Data
print("Data Understanding - CRISP-DM fase 2")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Tahun"].astype(str) + "-" + df["Bulan"].astype(str) + "-01")
df = df.sort_values("Date").reset_index(drop=True)

feature_cols = [
    "Inflation_YoY_Pct", "Inflation_Gap_Pct", "GDP_Growth_YoY_Pct",
    "USD_IDR_Monthly_Avg", "M2_Triliun_Rp", "Credit_Triliun_Rp",
    "IHSG_End_of_Month", "Foreign_Reserves_Miliar_USD", "Federal_Funds_Rate_Pct",
    "Oil_Price_Brent_USD_per_Bbl", "Gold_Price_USD_per_Oz", "VIX_Volatility_Index",
]
target_col = "BI_Rate_Pct"
numeric_cols = feature_cols + [target_col]

# 2. Dataset Overview
print("\n[1] Dataset Overview")
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Period: {df['Date'].min().strftime('%B %Y')} - {df['Date'].max().strftime('%B %Y')}")
print(f"  Duration: {(df['Date'].max() - df['Date'].min()).days // 365} years")
print(f"\n  Columns:")
for c in df.columns:
    print(f"    - {c}: {df[c].dtype}")

# 3. Missing Values Analysis
print("\n[2] Missing Values Analysis")
missing = df[numeric_cols].isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Pct (%)": missing_pct})
# Tetap tampilkan semua bar meskipun 0
missing_df = missing_df.sort_values("Missing", ascending=False)

print(missing_df.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#e74c3c" if v > 0 else "#27ae60" for v in missing_df["Pct (%)"]]
bars = ax.barh(missing_df.index, missing_df["Pct (%)"], color=colors, edgecolor="white")
ax.set_xlabel("Missing (%)")
ax.set_xlim(0, max(missing_df["Pct (%)"].max() * 1.2, 5))

for bar, (idx, row) in zip(bars, missing_df.iterrows()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{int(row['Missing'])} ({row['Pct (%)']:.1f}%)", va="center", fontsize=9)

ax.invert_yaxis()
ax.set_title("Missing Values pada Dataset Mentah (Raw)")

plt.savefig(OUTPUT_DIR / "01_missing_values.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 01_missing_values.png")


# 4. Descriptive Statistics
print("\n[3] Descriptive Statistics")
desc = df[numeric_cols].describe().T
desc["skewness"] = df[numeric_cols].skew()
desc["kurtosis"] = df[numeric_cols].kurtosis()
print(desc[["count", "mean", "std", "min", "max", "skewness", "kurtosis"]].to_string())

# Save stats table as image - Enlarged, No Title
fig, ax = plt.subplots(figsize=(20, 8))
ax.axis("tight")
ax.axis("off")
stats_display = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness", "kurtosis"]].round(2)

# Shorten row labels for readability
short_labels = {
    "Inflation_YoY_Pct": "Inflation YoY",
    "Inflation_Gap_Pct": "Inflation Gap",
    "GDP_Growth_YoY_Pct": "GDP Growth YoY",
    "USD_IDR_Monthly_Avg": "USD/IDR",
    "M2_Triliun_Rp": "M2 (Triliun Rp)",
    "Credit_Triliun_Rp": "Credit (Triliun Rp)",
    "IHSG_End_of_Month": "IHSG",
    "Foreign_Reserves_Miliar_USD": "Foreign Reserves",
    "Federal_Funds_Rate_Pct": "FFR",
    "Oil_Price_Brent_USD_per_Bbl": "Oil Price (Brent)",
    "Gold_Price_USD_per_Oz": "Gold Price (USD)",
    "VIX_Volatility_Index": "VIX",
    "BI_Rate_Pct": "BI-Rate",
}
row_labels = [short_labels.get(c, c) for c in stats_display.index]

table = ax.table(
    cellText=stats_display.values,
    colLabels=stats_display.columns,
    rowLabels=row_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(14)  # Perbesar font
table.auto_set_column_width(range(len(stats_display.columns)))
table.scale(1.2, 2.5)   # Buat sel tabel lebih tinggi dan lebar

# Style header row
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2980b9")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == -1:
        cell.set_facecolor("#eaf2f8")
        cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "white")

plt.savefig(OUTPUT_DIR / "02_descriptive_statistics.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 02_descriptive_statistics.png")

# 5. Target Variable Analysis (BI-Rate)
print("\n[4] Target Variable: BI-Rate")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Time series
axes[0, 0].plot(df["Date"], df[target_col], color="#2980b9", linewidth=1.5)
axes[0, 0].set_title("BI-Rate Time Series (2005-2026)")
axes[0, 0].set_ylabel("BI-Rate (%)")
axes[0, 0].xaxis.set_major_locator(mdates.YearLocator(2))
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].axhline(y=df[target_col].mean(), color="red", linestyle="--", alpha=0.5, label=f"Mean: {df[target_col].mean():.2f}%")
axes[0, 0].legend()

# Distribution
axes[0, 1].hist(df[target_col].dropna(), bins=20, color="#2980b9", edgecolor="white", alpha=0.8)
axes[0, 1].axvline(df[target_col].mean(), color="red", linestyle="--", label=f"Mean: {df[target_col].mean():.2f}%")
axes[0, 1].axvline(df[target_col].median(), color="green", linestyle="--", label=f"Median: {df[target_col].median():.2f}%")
axes[0, 1].set_title("Distribusi BI-Rate")
axes[0, 1].set_xlabel("BI-Rate (%)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend()

# Box plot
axes[1, 0].boxplot(df[target_col].dropna(), vert=True, widths=0.5,
                   boxprops=dict(color="#2980b9"), medianprops=dict(color="red"))
axes[1, 0].set_title("Box Plot BI-Rate")
axes[1, 0].set_ylabel("BI-Rate (%)")

# Year-over-year average
yearly = df.groupby("Tahun")[target_col].mean()
axes[1, 1].bar(yearly.index, yearly.values, color="#2980b9", edgecolor="white")
axes[1, 1].set_title("Rata-rata BI-Rate per Tahun")
axes[1, 1].set_xlabel("Tahun")
axes[1, 1].set_ylabel("BI-Rate (%)")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_target_birate_analysis.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 03_target_birate_analysis.png")

# 6. Time Series Plots For All Features
print("\n[5] Feature Time Series")

fig, axes = plt.subplots(4, 3, figsize=(20, 20))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    data = df[["Date", col]].dropna()
    ax.plot(data["Date"], data[col], color="#2980b9", linewidth=1, alpha=0.8)
    ax.fill_between(data["Date"], data[col], alpha=0.1, color="#2980b9")
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout(h_pad=2.0, w_pad=1.5)
plt.savefig(OUTPUT_DIR / "05_feature_timeseries.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 04_feature_timeseries.png")

# 7. Distribution Of All Features
print("\n[6] Feature Distributions")

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=25, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {data.mean():.2f}")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

plt.tight_layout(h_pad=2.0, w_pad=1.5)
plt.savefig(OUTPUT_DIR / "06_feature_distributions.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 05_feature_distributions.png")

# 8. Correlation Analysis
print("\n[7] Correlation Analysis")

# Full correlation heatmap - improved readability with triangular mask
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="YlOrRd", vmax=1, vmin=-1, center=0,
            annot=True, fmt=".2f", square=True, linewidths=0.8,
            linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
            ax=ax, annot_kws={"size": 9, "fontweight": "bold"})

# Shorten tick labels
short_tick = [short_labels.get(c, c) for c in corr.columns]
ax.set_xticklabels(short_tick, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(short_tick, rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_correlation_heatmap.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 06_correlation_heatmap.png")

# Correlation with target - ALL features
target_corr = corr[target_col].drop(target_col).sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#e74c3c" if v < 0 else "#27ae60" for v in target_corr.values]
bars = ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="white", height=0.6)
ax.set_xlabel("Korelasi Pearson")
ax.axvline(x=0, color="black", linewidth=0.5)
ax.axvline(x=0.7, color="orange", linestyle="--", alpha=0.5, label="Threshold kuat (|r|=0.7)")
ax.axvline(x=-0.7, color="orange", linestyle="--", alpha=0.5)
for bar, val in zip(bars, target_corr.values):
    ax.text(val + 0.01 if val >= 0 else val - 0.06, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_correlation_with_target.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 07_correlation_with_target.png")

# 9. Scatter Plots: All Features vs BI-Rate
print("\n[8] Scatter Plots (All Features vs BI-Rate)")

fig, axes = plt.subplots(4, 3, figsize=(20, 22))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    data = df[[col, target_col]].dropna()
    ax.scatter(data[col], data[target_col], alpha=0.5, s=20, color="#2980b9", edgecolors="white", linewidth=0.3)
    # Trend line
    z = np.polyfit(data[col], data[target_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[col].min(), data[col].max(), 100)
    ax.plot(x_line, p(x_line), color="red", linestyle="--", linewidth=1.5)
    r = data[col].corr(data[target_col])
    ax.set_title(f"{col}\nr = {r:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("BI-Rate (%)", fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout(h_pad=2.0, w_pad=1.5)
plt.savefig(OUTPUT_DIR / "09_scatter_all_features.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 08_scatter_all_features.png")

# 10. Box Plots For All Features
print("\n[9] Box Plots")

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    data = df[col].dropna()
    bp = ax.boxplot(data, vert=True, widths=0.5,
                    boxprops=dict(color="#2980b9"), medianprops=dict(color="red"),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    ax.set_title(col, fontsize=10, fontweight="bold")
    # Count outliers
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
    ax.set_xlabel(f"Outliers: {outliers}", fontsize=8, color="red")
    ax.tick_params(labelsize=8)

plt.tight_layout(h_pad=2.0, w_pad=1.5)
plt.savefig(OUTPUT_DIR / "10_boxplots_outliers.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 09_boxplots_outliers.png")

# 11. Bi-rate vs All Indicators (Dual Axis)
print("\n[10] BI-Rate vs All Indicators (Dual Axis)")

fig, axes = plt.subplots(4, 3, figsize=(22, 22))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax1 = axes[i]
    data = df[["Date", col, target_col]].dropna()
    color1, color2 = "#2980b9", "#e74c3c"

    ax1.plot(data["Date"], data[target_col], color=color1, linewidth=1.5, label="BI-Rate")
    ax1.set_ylabel("BI-Rate (%)", color=color1, fontsize=8)
    ax1.tick_params(axis="y", labelcolor=color1, labelsize=7)

    ax2 = ax1.twinx()
    ax2.plot(data["Date"], data[col], color=color2, linewidth=1, alpha=0.7, label=col)
    ax2.set_ylabel(short_labels.get(col, col), color=color2, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=7)

    ax1.set_title(f"BI-Rate vs {short_labels.get(col, col)}", fontsize=10, fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", rotation=45, labelsize=7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

plt.tight_layout(h_pad=2.0, w_pad=1.5)
plt.savefig(OUTPUT_DIR / "11_birate_vs_indicators.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 10_birate_vs_indicators.png")

# 12. Stationarity Check (Rolling Statistics)
print("\n[11] Rolling Statistics (Stationarity Check)")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
check_cols = [target_col, "Inflation_YoY_Pct", "USD_IDR_Monthly_Avg"]

for i, col in enumerate(check_cols):
    ax = axes[i]
    data = df[["Date", col]].dropna()
    rolling_mean = data[col].rolling(window=12).mean()
    rolling_std = data[col].rolling(window=12).std()

    ax.plot(data["Date"], data[col], color="#2980b9", linewidth=1, alpha=0.6, label="Original")
    ax.plot(data["Date"], rolling_mean, color="#e74c3c", linewidth=2, label="Rolling Mean (12)")
    ax.plot(data["Date"], rolling_std, color="#27ae60", linewidth=2, label="Rolling Std (12)")
    ax.set_title(f"{col} - Rolling Statistics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_rolling_statistics.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 11_rolling_statistics.png")


# 14. Yearly Heatmap
print("\n[13] BI-Rate Yearly Heatmap")

pivot = df.pivot_table(values=target_col, index="Tahun", columns="Bulan", aggfunc="mean")
bulan_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
pivot.columns = [bulan_names[int(c)-1] for c in pivot.columns]

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5,
            cbar_kws={"label": "BI-Rate (%)"}, ax=ax, annot_kws={"size": 8})
ax.set_ylabel("Tahun")
ax.set_xlabel("Bulan")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_birate_heatmap.png", bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  Saved: 13_birate_heatmap.png")

# Summary
print("\nData Understanding selesai")
print(f"Total plots: 13")
print(f"Output: {OUTPUT_DIR}")
print(f"\nKey findings:")
print(f"  - Dataset: {df.shape[0]} observations, {df.shape[1]} columns")
print(f"  - Period: {df['Date'].min().strftime('%B %Y')} - {df['Date'].max().strftime('%B %Y')}")
print(f"  - Target (BI-Rate): mean={df[target_col].mean():.2f}%, range=[{df[target_col].min():.2f}%, {df[target_col].max():.2f}%]")
total_missing = df[numeric_cols].isnull().sum().sum()
print(f"  - Missing values: {total_missing} (raw dataset)")
print(f"  - Strongest positive correlation with BI-Rate: {target_corr.idxmax()} (r={target_corr.max():.3f})")
print(f"  - Strongest negative correlation with BI-Rate: {target_corr.idxmin()} (r={target_corr.min():.3f})")
