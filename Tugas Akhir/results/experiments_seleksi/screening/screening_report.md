# Laporan Fase 1: Hasil Screening

**Waktu Eksekusi:** 409.6 detik
**Total Kombinasi Eksperimen:** 384
**Model Evaluasi:** RF, XGBoost

Fase ini bertujuan untuk mengevaluasi secara cepat seluruh ruang kombinasi perlakuan data menggunakan algoritma yang efisien. Dari total 384 skenario, berikut adalah 20 kombinasi teratas yang mencatat error terendah.

## 20 Kombinasi Terbaik

| Peringkat | ID Skenario | Imputasi | Filter Fitur | Skala | Feature Engineering | Rata-rata RMSE |
|-----------|-------------|----------|--------------|-------|---------------------|----------------|
| 1 | E136 | ffill | domain_itf | minmax | policy_reaction | 0.2000 |
| 2 | E144 | ffill | domain_itf | robust | policy_reaction | 0.2092 |
| 3 | E128 | ffill | domain_itf | standard | policy_reaction | 0.2093 |
| 4 | E160 | ffill | corr_top6 | minmax | policy_reaction | 0.2102 |
| 5 | E168 | ffill | corr_top6 | robust | policy_reaction | 0.2186 |
| 6 | E184 | ffill | mutual_info | minmax | policy_reaction | 0.2211 |
| 7 | E152 | ffill | corr_top6 | standard | policy_reaction | 0.2212 |
| 8 | E112 | ffill | all | minmax | policy_reaction | 0.2247 |
| 9 | E192 | ffill | mutual_info | robust | policy_reaction | 0.2287 |
| 10 | E176 | ffill | mutual_info | standard | policy_reaction | 0.2296 |
| 11 | E104 | ffill | all | standard | policy_reaction | 0.2353 |
| 12 | E120 | ffill | all | robust | policy_reaction | 0.2358 |
| 13 | E173 | ffill | mutual_info | standard | diff | 0.3886 |
| 14 | E189 | ffill | mutual_info | robust | diff | 0.3887 |
| 15 | E181 | ffill | mutual_info | minmax | diff | 0.3888 |
| 16 | E109 | ffill | all | minmax | diff | 0.3905 |
| 17 | E101 | ffill | all | standard | diff | 0.3905 |
| 18 | E117 | ffill | all | robust | diff | 0.3905 |
| 19 | E186 | ffill | mutual_info | robust | lag | 0.4052 |
| 20 | E178 | ffill | mutual_info | minmax | lag | 0.4076 |

## Ringkasan Statistik

| Metrik | Nilai |
|--------|-------|
| Total skenario sukses | 384 |
| RMSE rata-rata terbaik | 0.2000 |
| RMSE rata-rata terburuk | 2.4443 |
| Median performa (RMSE) | 1.0450 |

---
*Laporan dibuat pada: 05 August 2026, 13:45 WIB*
