# Laporan Fase 2: Analisis Pengaruh Faktor

Tahap ini mengurai dampak spesifik tiap komponen perlakuan (imputasi, seleksi fitur, algoritma skala, dan rekayasa fitur) terhadap tingkat akurasi prediksi. Penilaian didasarkan pada selisih antara nilai konfigurasi terbaik dan terburuk pada setiap kategori.

## Tingkat Signifikansi Faktor (Diurutkan dari dampak paling tinggi)

| Peringkat | Komponen / Faktor | Rentang Dampak (RMSE) | Level Paling Optimal | Level Terburuk |
|-----------|------------------|----------------------|----------------------|----------------|
| 1 | feature_engineering | 0.7444 | policy_reaction | interaction |
| 2 | features | 0.6892 | mutual_info | corr_top6 |
| 3 | imputation | 0.4275 | ffill | knn |
| 4 | scaler | 0.0009 | robust | standard |

## Performa Rata-rata per Komponen

### Faktor Pilihan: Imputation

| imputation   |     mean |      std |      min |     max |   count |
|:-------------|---------:|---------:|---------:|--------:|--------:|
| ffill        | 0.842986 | 0.557897 | 0.193424 | 2.40303 |     192 |
| interpolate  | 1.24823  | 0.417493 | 0.715276 | 2.31321 |     192 |
| mean         | 1.24877  | 0.437699 | 0.683321 | 2.45139 |     192 |
| knn          | 1.27047  | 0.448363 | 0.723004 | 2.41531 |     192 |

### Faktor Pilihan: Features

| features    |     mean |      std |      min |     max |   count |
|:------------|---------:|---------:|---------:|--------:|--------:|
| mutual_info | 0.939758 | 0.247282 | 0.200161 | 1.21756 |     192 |
| all         | 0.944813 | 0.23745  | 0.220693 | 1.21164 |     192 |
| domain_itf  | 1.0969   | 0.338453 | 0.193424 | 1.67779 |     192 |
| corr_top6   | 1.62898  | 0.674325 | 0.206722 | 2.45139 |     192 |

### Faktor Pilihan: Scaler

| scaler   |    mean |      std |      min |     max |   count |
|:---------|--------:|---------:|---------:|--------:|--------:|
| robust   | 1.1521  | 0.5018   | 0.193424 | 2.45139 |     256 |
| minmax   | 1.1527  | 0.501472 | 0.194041 | 2.45139 |     256 |
| standard | 1.15305 | 0.500967 | 0.194041 | 2.45139 |     256 |

### Faktor Pilihan: Feature Engineering

| feature_engineering   |     mean |       std |      min |     max |   count |
|:----------------------|---------:|----------:|---------:|--------:|--------:|
| policy_reaction       | 0.755432 | 0.356853  | 0.193424 | 1.21164 |      96 |
| rolling               | 1.01397  | 0.0782363 | 0.883002 | 1.12748 |      96 |
| lag_rolling           | 1.02358  | 0.410249  | 0.385963 | 1.77333 |      96 |
| lag                   | 1.09018  | 0.486786  | 0.401868 | 1.99042 |      96 |
| diff                  | 1.13113  | 0.608112  | 0.365215 | 2.3605  |      96 |
| none                  | 1.34552  | 0.466012  | 0.891978 | 2.28468 |      96 |
| calendar              | 1.36121  | 0.455129  | 0.891046 | 2.38482 |      96 |
| interaction           | 1.49988  | 0.536854  | 1.04672  | 2.45139 |      96 |


## Analisis Efek Interaksi (Pairwise)

### Relasi imputation terhadap features

| imputation   |    all |   corr_top6 |   domain_itf |   mutual_info |
|:-------------|-------:|------------:|-------------:|--------------:|
| ffill        | 0.7025 |      1.1743 |       0.8134 |        0.6817 |
| interpolate  | 1.0274 |      1.7541 |       1.1828 |        1.0286 |
| knn          | 1.0282 |      1.8123 |       1.2119 |        1.0295 |
| mean         | 1.0212 |      1.7752 |       1.1795 |        1.0192 |

### Relasi imputation terhadap scaler

| imputation   |   minmax |   robust |   standard |
|:-------------|---------:|---------:|-----------:|
| ffill        |   0.8423 |   0.8430 |     0.8437 |
| interpolate  |   1.2486 |   1.2474 |     1.2486 |
| knn          |   1.2708 |   1.2698 |     1.2708 |
| mean         |   1.2490 |   1.2482 |     1.2491 |

### Relasi imputation terhadap feature_engineering

| imputation   |   calendar |   diff |   interaction |    lag |   lag_rolling |   none |   policy_reaction |   rolling |
|:-------------|-----------:|-------:|--------------:|-------:|--------------:|-------:|------------------:|----------:|
| ffill        |     1.3220 | 0.4420 |        1.4892 | 0.4652 |        0.4693 | 1.3331 |            0.2203 |    1.0028 |
| interpolate  |     1.3671 | 1.3420 |        1.4791 | 1.2955 |        1.2156 | 1.3335 |            0.9347 |    1.0183 |
| knn          |     1.4079 | 1.3875 |        1.5131 | 1.3126 |        1.2189 | 1.3650 |            0.9398 |    1.0191 |
| mean         |     1.3478 | 1.3530 |        1.5181 | 1.2874 |        1.1906 | 1.3504 |            0.9270 |    1.0158 |

### Relasi features terhadap scaler

| features    |   minmax |   robust |   standard |
|:------------|---------:|---------:|-----------:|
| all         |   0.9449 |   0.9443 |     0.9452 |
| corr_top6   |   1.6288 |   1.6288 |     1.6293 |
| domain_itf  |   1.0970 |   1.0963 |     1.0974 |
| mutual_info |   0.9401 |   0.9389 |     0.9403 |

### Relasi features terhadap feature_engineering

| features    |   calendar |   diff |   interaction |    lag |   lag_rolling |   none |   policy_reaction |   rolling |
|:------------|-----------:|-------:|--------------:|-------:|--------------:|-------:|------------------:|----------:|
| all         |     1.0630 | 0.8439 |        1.0936 | 0.8938 |        0.8531 | 1.0285 |            0.7930 |    0.9896 |
| corr_top6   |     2.1143 | 1.8063 |        2.3743 | 1.5346 |        1.3188 | 2.1154 |            0.7189 |    1.0492 |
| domain_itf  |     1.2091 | 0.9746 |        1.4140 | 1.0606 |        1.1053 | 1.2269 |            0.7663 |    1.0185 |
| mutual_info |     1.0584 | 0.8997 |        1.1176 | 0.8718 |        0.8172 | 1.0113 |            0.7435 |    0.9986 |

### Relasi scaler terhadap feature_engineering

| scaler   |   calendar |   diff |   interaction |    lag |   lag_rolling |   none |   policy_reaction |   rolling |
|:---------|-----------:|-------:|--------------:|-------:|--------------:|-------:|------------------:|----------:|
| minmax   |     1.3612 | 1.1310 |        1.4999 | 1.0905 |        1.0236 | 1.3454 |            0.7561 |    1.0139 |
| robust   |     1.3612 | 1.1311 |        1.4999 | 1.0896 |        1.0235 | 1.3455 |            0.7520 |    1.0139 |
| standard |     1.3612 | 1.1313 |        1.4999 | 1.0905 |        1.0236 | 1.3456 |            0.7582 |    1.0141 |


---
*Laporan dibuat pada: 03 August 2026, 09:46 WIB*
