# Prediksi BI-Rate dengan XAI dan LLM

Tugas Akhir oleh **Erwan Poltak Halomoan** (18222028)  
Program Studi Sistem dan Teknologi Informasi  
Institut Teknologi Bandung

## Deskripsi

Sistem prediksi suku bunga acuan Bank Indonesia (BI-Rate) berbasis:
- **Machine Learning**: Random Forest, XGBoost
- **Deep Learning**: LSTM, Bi-LSTM
- **Explainable AI**: SHAP, LIME
- **Large Language Model**: GPT-4o untuk narasi interpretasi

## Struktur Folder

```
Tugas Akhir/
├── data/
│   ├── raw/
│   │   ├── external/     # Data makroekonomi mentah (CSV per variabel)
│   │   └── bps/          # Data BPS (M2, PDB)
│   └── processed/
│       └── dataset_final.csv   # Dataset gabungan siap pakai
│
├── docs/
│   ├── proposal/
│   │   ├── latex/        # File LaTeX proposal
│   │   ├── image/        # Gambar untuk proposal
│   │   └── table/        # Tabel LaTeX
│   └── references/       # Jurnal dan referensi PDF
│
├── src/
│   ├── experiments/      # Mesin eksperimen faktorial + registry model terbaik
│   ├── modeling/         # Model (ARIMA, VAR, RF, XGB, LSTM, Bi-LSTM) + data prep
│   ├── xai/              # SHAP, LIME, dan LLM narrator
│   ├── analysis/         # Dekomposisi STL (EDA pendukung Bab III)
│   ├── evaluation_core.py      # Inti pipeline evaluasi kanonis (split, model, SHAP/LIME)
│   ├── generate_narratives.py  # Narasi LLM (teks) per model per bulan evaluasi
│   └── generate_xai_charts.py  # Gambar SHAP/LIME/waterfall per model per bulan
│
├── run_experiments.py    # Entry-point pipeline eksperimen
├── models/               # Model terbaik per algoritma (auto-prune)
├── results/              # Output eksperimen & dokumen evaluasi
└── requirements.txt      # Dependencies
```

## Dataset

Dataset berisi 16 kolom dengan 251 observasi bulanan (Juli 2005 - Mei 2026):

| Kolom | Deskripsi |
|-------|-----------|
| ID | Identitas unik observasi |
| Bulan, Tahun | Periode waktu |
| Inflation_YoY_Pct | Inflasi year-on-year (%) |
| Inflation_Gap | Selisih inflasi aktual - target |
| GDP_Growth_YoY_Pct | Pertumbuhan PDB (%) |
| USD_IDR_Monthly_Avg | Nilai tukar rata-rata bulanan |
| M2_Triliun_Rp | Jumlah uang beredar |
| Credit_Triliun_Rp | Posisi kredit bank umum+BPR (level; %YoY untuk pemodelan) |
| IHSG_End_of_Month | Indeks harga saham |
| Foreign_Reserves_Miliar_USD | Cadangan devisa (nilai dalam **juta** USD; nama kolom mewarisi label lama) |
| Federal_Funds_Rate_Pct | Suku bunga Fed |
| Oil_Price_Brent_USD_per_Bbl | Harga minyak Brent |
| Gold_Price_USD_per_Oz | Harga emas |
| VIX_Volatility_Index | Indeks volatilitas |
| **BI_Rate_Pct** | Target: Suku bunga BI |

## Instalasi

```bash
pip install -r requirements.txt
```

## Pengumpulan Data

Dataset dibangun dari berkas sumber per-variabel di `data/raw/external/`. Satu kali
jalan menghasilkan ketiga berkas dataset:

```bash
python data/build_datasets.py
```

Keluaran:
- `dataset_raw_integrated.csv` — mentah penuh (Juli 2005–Mei 2026), untuk Bab III.
- `dataset_final.csv` — terproses, **historis s.d. Desember 2025** untuk pelatihan/pengujian model.
- `dataset_2026.csv` — **holdout Januari–Mei 2026** (data uji terkini), terpisah agar
  pipeline pemodelan tetap berjalan walau nilai 2026 belum lengkap.

Menambah bulan terbaru (mengisi nilai ke berkas sumber lalu regenerasi otomatis):

```bash
# 1. Edit nilai bulan baru di data/data_collection/add_recent_months.py
#    (atau langsung di berkas data/raw/external/*.csv)
# 2. Jalankan:
python data/data_collection/add_recent_months.py
```

Skrip menjalankan urutan: `fetch_global_markets.py` (auto-fetch 6 variabel pasar
global dari FRED & Yahoo Finance, hanya periode ≥2026) → `preprocessing/gdp.py` (PDB
kuartalan→bulanan) → `preprocessing/inflation_gap.py` (inflasi−target) →
`preprocessing/extract_credit_seki.py` (kredit bulanan dari berkas Excel BI-SEKI
Tabel I.4 → level + %YoY) → `build_datasets.py` (integrasi kedua dataset). Variabel
domestik (BI-Rate, inflasi, PDB, M2, cadangan devisa) diisi manual dari rilis resmi
BI/BPS di `add_recent_months.py`; kredit diperbarui dengan mengunduh ulang berkas
SEKI ke `data/raw/Kredit_Bank_Umum_SEKI.xls`.

**Prinsip data hilang:** nilai bulan yang belum dirilis sumber resmi (mis. PDB Q2,
M2/cadangan/kredit bulan terkini) **dibiarkan kosong (NaN)**, bukan diisi paksa —
imputasinya ditentukan secara ilmiah oleh faktor *imputation* pada eksperimen
faktorial. Ekstraktor BPS satu kali (`extract_gdp_yoy_quarterly.py`, `extract_m2.py`)
hanya dijalankan bila ada rilis data sumber baru.

## Alur Lengkap

```bash
# 1. Bangun dataset (historis s.d. 2025 + holdout 2026 terpisah)
python data/build_datasets.py

# 2. Jalankan eksperimen faktorial penuh pada data historis (s.d. 2025)
python run_experiments.py --pipeline

# 3. Narasi LLM (teks) untuk seluruh bulan evaluasi
python -m src.generate_narratives

# 4. Gambar interpretasi XAI (SHAP, LIME, waterfall) per model per bulan
python -m src.generate_xai_charts
```

Langkah 3 dan 4 (keduanya berat, melatih ulang model) memakai jalur evaluasi
kanonis `src/evaluation_core.py`: keempat model terbaik dilatih, dua bulan
kontrol dipilih otomatis (satu bulan seluruh model tepat, satu bulan seluruh
model meleset), lalu dihasilkan **berkas teks narasi** di
`results/experiments/evaluasi/narasi/` dan **gambar PNG** di
`results/experiments/evaluasi/xai_charts/`. Kedua keluaran itulah yang penulis
rangkai secara manual menjadi lembar evaluasi domain expert (Excel) dan
lampiran buku.

Folder `models/` otomatis hanya menyimpan model terbaik per algoritma; berkas
`.pkl`/`.keras` lama yang tergantikan oleh model yang lebih baik akan dihapus.

## Opsi Eksperimen

`run_experiments.py` menjalankan eksperimen faktorial. Selain `--pipeline`
(seluruh fase berurutan), tersedia opsi:

```bash
python run_experiments.py --screening      # hanya fase screening
python run_experiments.py --final          # hanya fase final
python run_experiments.py --report         # bangun ulang PDF laporan eksperimen
python run_experiments.py --pipeline --skip-dl   # lewati deep learning (lebih cepat)
python run_experiments.py --pipeline --top 6     # jumlah kombinasi terbaik yang dilanjutkan
```

## Reproduksibilitas dan Sumber Kebenaran Angka

Angka kinerja yang dilaporkan pada buku tugas akhir (RMSE 0,1434 / 0,1463 /
0,1620 / 0,2207 dst.) bersumber dari **jalur evaluasi kanonis** `src/evaluation_core.py` (dipakai
`generate_narratives` dan `generate_xai_charts`): split kronologis 60/20/20 yang tetap,
LSTM/BiLSTM dimuat dari **ensembel 3-seed yang dibekukan**
(`models/*_seed{42,43,44}.weights.h5`), serta RF/XGBoost dilatih ulang secara
deterministik (`random_state=42`). Registry
(`results/experiments/best_results_registry.json`) disinkronkan ke angka kanonis
ini (lihat kolom `note` tiap entri) dan setiap entri ditandai `"locked": true`,
sehingga menjalankan ulang pipeline tidak akan menimpanya (`update_if_better`
melewati entri terkunci).

Dua hal yang perlu diketahui pembaca repositori:

1. **Artefak fase di `results/experiments/{screening,focused,tuning,final}/`
   berasal dari re-run verifikasi** dengan parameter berbeda (mis. `--top 6`,
   bukan 10) dan, untuk LSTM/BiLSTM, dari ensembel yang tidak dipersistensi
   penuh — metrik NN pada artefak fase karena itu **tidak identik** dengan
   angka kanonis di buku. Yang mengikat adalah jalur kanonis di atas.
2. **XGBoost tidak portabel antar-OS**: algoritma `hist` + subsampling
   menghasilkan angka yang deterministik per platform tetapi berbeda antara
   Windows dan macOS (RMSE uji E112: 0,1960 di Windows, 0,2207 di macOS). RF,
   LSTM, dan BiLSTM reprodusibel lintas OS. Sejak **31 Juli 2026 seluruh angka
   pada buku diseragamkan ke lingkungan macOS**, termasuk Tabel top-10
   *screening*, analisis faktor Fase 2, tabel hasil model terbaik, tabel
   penalaan, dan seluruh tabel modul analisis. Konfigurasi kanonis XGBoost
   tetap **E112** meskipun pada macOS ia bukan lagi kombinasi XGBoost terbaik
   pada *screening* (E144 yang terbaik, 0,1934), sebab narasi LLM dan penilaian
   *domain expert* pada Lampiran E dihasilkan dari konfigurasi tersebut dan
   tidak dapat diulang.

### Konfigurasi LLM (untuk narasi XAI)

Atur di berkas `.env` pada root proyek (lihat `src/xai/llm_narrator.py`):

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_MODEL=gpt-4o
```

Model yang dipakai untuk seluruh narasi pada buku adalah **GPT-4o**. Klien memakai
antarmuka pustaka `openai` sehingga penyedia lain yang kompatibel juga bisa
dipakai dengan mengganti `LLM_BASE_URL` dan `LLM_MODEL`.

Bila LLM tidak tersedia, narasi otomatis mundur ke ringkasan deterministik
(tiga kontributor SHAP teratas).

## Model yang Diimplementasikan

### Tradisional
- ARIMA (Auto-Regressive Integrated Moving Average)
- VAR (Vector Autoregression)

### Machine Learning
- Random Forest Regressor
- XGBoost Regressor

### Deep Learning
- LSTM (Long Short-Term Memory)
- Bi-LSTM (Bidirectional LSTM)

## Metrik Evaluasi

- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R-squared
- Akurasi arah (hit-rate) pada bulan-perubahan vs bulan-tahan
- Theil's U2 dan MASE terhadap pembanding naif (*random walk*)
- Uji Pesaran-Timmermann atas ketepatan arah (dipakai di buku)
- Uji Diebold-Mariano dengan koreksi Harvey-Leybourne-Newbold — diagnostik
  pendukung di kode saja, tidak dipakai di buku karena sumber acuannya tidak ada
  pada daftar pustaka; nilainya tetap tersimpan di CSV

## Modul Analisis Pasca-Sidang (revisi Juli 2026)

Lima modul berikut ditambahkan untuk menjawab catatan penguji. Seluruhnya membaca
artefak kanonis sehingga angkanya konsisten dengan `best_results_registry.json`.

| Modul | Menjawab | Perintah |
|---|---|---|
| `src/analysis/naive_benchmark.py` | Apakah model punya *forecast skill* (U2, MASE, uji DM) | `python -m src.analysis.naive_benchmark --write` |
| `src/analysis/ablation_study.py` | Kontribusi tiap kelompok fitur dan tiap komponen sistem | `python -m src.analysis.ablation_study --write` |
| `src/analysis/ablation_study.py --only perfitur` | Ablasi satu-per-satu (*leave-one-out*): sumbangan tiap fitur secara individual terhadap galat uji | `python -m src.analysis.ablation_study --only perfitur --write` |
| `src/analysis/overfitting_diagnostics.py` | Selisih per segmen, sebab $R^2$ validasi negatif, ekstrapolasi rezim, validasi bergulir | `python -m src.analysis.overfitting_diagnostics --write --n-origin 36` |
| `src/analysis/error_analysis.py` | Sebaran galat per tahun dan kasus terburuk | `python -m src.analysis.error_analysis --write` |
| `src/analysis/xai_consistency.py` | Apakah atribusi XAI antarmodel sepakat, dan seberapa stabil estimasinya | `python -m src.analysis.xai_consistency --write --kestabilan 5` |
| `src/analysis/xai_konsolidasi.py` | Apa yang menggerakkan BI-Rate menurut GABUNGAN keempat model: pangsa persistensi per model + peringkat konsolidasi pendorong makro atas 4 model × 2 bulan kontrol | `python -m src.analysis.xai_konsolidasi --write` |

Catatan penting soal biaya komputasi: `overfitting_diagnostics` dengan
`--n-origin 36` melatih ulang model pada setiap titik awal, sehingga untuk LSTM
dan BiLSTM berarti 108 pelatihan per model dan memakan waktu sekitar satu jam di
CPU. Pakai `--skip-walkforward` bila hanya perlu diagnostik per segmen, atau
turunkan `--n-origin`.

`naive_benchmark` dan `error_analysis` tidak melatih model sama sekali (keduanya
membaca `pred_vs_actual_full.csv`), sedangkan `ablation_study`,
`overfitting_diagnostics`, `xai_consistency`, dan `xai_konsolidasi` melatih ulang.
Sejak 31 Juli 2026 seluruh modul dijalankan di macOS dan angkanya sudah seragam
dengan buku; bila suatu saat dijalankan ulang di Windows, baris XGBoost akan
bergeser ke 0,1960 (lihat butir 2 pada bagian Reproduksibilitas di atas).

## XAI

- **SHAP**: Global dan local feature importance
- **LIME**: Local interpretable explanations

## LLM Integration

GPT-4o (dikonfigurasi via `.env`) digunakan untuk menerjemahkan hasil analisis XAI
menjadi narasi kebijakan yang dapat dipahami oleh pembuat kebijakan Bank Indonesia.

## Lisensi

Tugas Akhir - Institut Teknologi Bandung, 2026
