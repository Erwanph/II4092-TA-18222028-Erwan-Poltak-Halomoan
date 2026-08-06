# Hasil Dekomposisi Time Series (STL)
Metode: Seasonal-Trend Decomposition using LOESS (STL), robust=True
Jumlah variabel: 7

| Variabel | Trend Range | Seasonal Amp. | Residual Std | Seasonal Ratio | Residual Ratio |
|----------|-------------|---------------|-------------|----------------|----------------|
| BI_Rate_Pct **[TARGET]** | 9.5359 | 2.4514 | 0.6524 | 12.43% | 32.09% |
| Inflation_YoY_Pct | 19.1528 | 6.2782 | 1.7622 | 19.91% | 50.98% |
| Federal_Funds_Rate_Pct | 5.3759 | 0.9558 | 0.3794 | 5.48% | 19.05% |
| USD_IDR_Monthly_Avg | 7961.8222 | 746.5642 | 413.4455 | 4.85% | 15.93% |
| GDP_Growth_YoY_Pct | 9.8677 | 4.7254 | 1.3456 | 31.77% | 66.76% |
| IHSG_End_of_Month | 6606.5486 | 1900.3465 | 322.6559 | 9.42% | 16.70% |
| Oil_Price_Brent_USD_per_Bbl | 65.8521 | 18.4576 | 10.2020 | 12.41% | 44.08% |

## Interpretasi BI_Rate_Pct

- **Komponen seasonal terdeteksi** dengan rasio 12.43% terhadap total variasi.
- **Trend mendominasi** variasi BI-Rate (range: 9.54 poin), mencerminkan kebijakan moneter jangka panjang.
- **Residual** (noise) memiliki std=0.6524, ratio=32.09%. Bagian inilah yang menjadi target model machine learning untuk dipelajari.
