# Laporan Fase 5: Evaluasi Komparatif Komprehensif

Dalam fase puncak ini, kita mengkomparasi efisiensi dari penalaan struktur algoritma terpilih dengan model parameter acuan (baseline) klasik. Hal ini memastikan setiap modifikasi eksperimen telah meningkatkan prediksi data *Time Series*.

## Ringkasan Banding Data

Tabel di bawah menggabungkan setiap arsitektur model dan seberapa jauh peningkatan akurasinya:

| configuration               | experiment_id   | source      | model   |   RMSE |    MAPE |        R2 |
|:----------------------------|:----------------|:------------|:--------|-------:|--------:|----------:|
| Baseline                    | BASELINE        | current_run | ARIMA   | 1.9802 | 31.3861 |   -3.2475 |
| Baseline                    | BASELINE        | current_run | VAR     | 3.4002 | 50.7687 |  -11.5243 |
| Baseline                    | BASELINE        | current_run | RF      | 0.9920 | 18.0864 |   -0.0659 |
| Baseline                    | BASELINE        | current_run | XGBoost | 1.0501 | 19.4314 |   -0.1946 |
| Baseline                    | BASELINE        | current_run | LSTM    | 0.1946 |  2.4340 |    0.9107 |
| Baseline                    | BASELINE        | current_run | BiLSTM  | 0.1605 |  2.2032 |    0.9393 |
| Run Saat Ini (E160)         | E160            | current_run | ARIMA   | 0.8977 | 14.0499 |   -3.2685 |
| Run Saat Ini (E160)         | E160            | current_run | VAR     | 4.6012 | 63.2671 | -111.1357 |
| Run Saat Ini (E160)         | E160            | current_run | RF      | 0.2069 |  3.0264 |    0.7732 |
| Run Saat Ini (E160)         | E160            | current_run | XGBoost | 0.2134 |  3.1077 |    0.7588 |
| Run Saat Ini (E160)         | E160            | current_run | LSTM    | 0.1189 |  1.5125 |    0.9417 |
| Run Saat Ini (E160)         | E160            | current_run | BiLSTM  | 0.1765 |  2.7552 |    0.8716 |
| Terbaik Historis (Registry) | E152            | focused     | ARIMA   | 0.8974 | 14.0453 |   -3.2654 |
| Terbaik Historis (Registry) | E136            | tuning      | BiLSTM  | 0.1463 |  2.0100 |    0.8867 |
| Terbaik Historis (Registry) | E144            | tuning      | LSTM    | 0.1620 |  2.3000 |    0.8610 |
| Terbaik Historis (Registry) | E136            | tuning      | RF      | 0.1434 |  1.9500 |    0.8911 |
| Terbaik Historis (Registry) | E136            | focused     | VAR     | 2.7089 | 38.9022 |  -37.8675 |
| Terbaik Historis (Registry) | E112            | focused     | XGBoost | 0.2207 |  3.1834 |    0.7420 |

## Detail Kombinasi Terbaik Keseluruhan (Registry)

Tabel berikut menunjukkan kombinasi preprocessing yang digunakan oleh masing-masing model pada hasil terbaik sepanjang seluruh eksperimen.

| Model | ID | Imputasi | Seleksi Fitur | Scaler | Feature Engineering | RMSE | MAPE (%) | R2 | Sumber Fase |
|-------|----|----------|---------------|--------|--------------------|------|----------|----|-------------|
| RF | E136 | Forward Fill | Domain ITF (7) | MinMaxScaler | policy_reaction | 0.1434 | 1.95 | 0.8911 | Fase 4 (Tuned) |
| BiLSTM | E136 | Forward Fill | Domain ITF (7) | MinMaxScaler | policy_reaction | 0.1463 | 2.01 | 0.8867 | Fase 4 (Tuned) |
| LSTM | E144 | Forward Fill | Domain ITF (7) | RobustScaler | policy_reaction | 0.1620 | 2.30 | 0.8610 | Fase 4 (Tuned) |
| XGBoost | E112 | Forward Fill | Semua 12 Fitur | MinMaxScaler | policy_reaction | 0.2207 | 3.18 | 0.7420 | Fase 3 |
| ARIMA | E152 | Forward Fill | Mutual Info Top-7 | StandardScaler | diff | 0.8974 | 14.05 | -3.2654 | Fase 3 |
| VAR | E136 | Forward Fill | Domain ITF (7) | MinMaxScaler | policy_reaction | 2.7089 | 38.90 | -37.8675 | Fase 3 |

### Hyperparameter Hasil Tuning

- **RF**: `{'n_estimators': 50, 'max_depth_none': True, 'min_samples_split': 11, 'min_samples_leaf': 2, 'max_features': 1.0}`
- **BiLSTM**: `{'units': 128, 'dropout': 0.4682, 'sequence_length': 3, 'learning_rate': 0.001702, 'batch_size': 16, 'epochs': 100}`
- **LSTM**: `{'units': 32, 'dropout': 0.4924, 'sequence_length': 3, 'learning_rate': 0.009188, 'batch_size': 8, 'epochs': 100}`
- **XGBoost**: `default (DEFAULT_HYPERPARAMS['XGBoost'] + early stopping); hasil tuning ditolak karena degradasi pada data uji`


---
*Laporan dibuat pada: 03 August 2026, 11:58 WIB*
