# Pengembangan Sistem Prediksi BI-Rate Berbasis *Explainable Artificial Intelligence* (XAI) dan *Large Language Model* (LLM) dengan Indikator Makroekonomi

Tugas Akhir — Program Studi Sistem dan Teknologi Informasi
Sekolah Teknik Elektro dan Informatika, Institut Teknologi Bandung, 2026

**Penulis:** Erwan Poltak Halomoan (18222028)

---

## Ringkasan

Repositori ini memuat kode eksperimen dan artefak hasil percobaan untuk sistem
prediksi suku bunga acuan Bank Indonesia (BI-Rate).

Sistem yang dibangun terdiri dari tiga lapisan:

1. **Prediksi** — enam model dari dua kelompok, yaitu ekonometrika (ARIMA, VAR)
   dan *machine learning* (Random Forest, XGBoost, LSTM, BiLSTM), dipilih melalui
   eksperimen faktorial atas 384 kombinasi pra-pemrosesan.
2. **Interpretasi** — SHAP dan LIME untuk menguraikan kontribusi tiap indikator
   makroekonomi terhadap angka prediksi.
3. **Narasi** — GPT-4o menerjemahkan keluaran XAI menjadi laporan berbahasa
   Indonesia yang sepenuhnya berpijak pada data yang diberikan.

Dataset berisi 251 observasi bulanan (Juli 2005–Mei 2026) dengan 16 kolom
indikator domestik dan global.


---

## Struktur Repositori

```
II4092-TA-18222028-Erwan-Poltak-Halomoan/
├── Tugas Akhir/                    # Kode, data, dan artefak eksperimen
│   ├── data/                       # Pengumpulan, pra-pemrosesan, dan dataset
│   │   ├── raw/                    # Berkas sumber per variabel (CSV, Excel)
│   │   └── processed/              # dataset_final.csv, dataset_2026.csv
│   ├── src/
│   │   ├── modeling/               # Enam model + pra-pemrosesan + metrik
│   │   ├── experiments/            # Mesin eksperimen faktorial + registry
│   │   ├── xai/                    # SHAP, LIME, dan LLM narrator
│   │   ├── analysis/               # Modul analisis lanjutan
│   │   └── evaluation_core.py      # Jalur evaluasi kanonis
│   ├── models/                     # Bobot model terbaik per algoritma
│   ├── results/                    # Metrik, registry, grafik, narasi
│   ├── run_experiments.py          # Entry-point pipeline eksperimen
│   ├── requirements.txt
│   └── README.md                   # Dokumentasi rinci sisi kode
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Menjalankan Ulang Eksperimen

**Prasyarat:** Python 3.10. Versi lain belum diuji.

```bash
cd "Tugas Akhir"
pip install -r requirements.txt
```

Alur lengkap dari data mentah sampai keluaran XAI dan narasi:

```bash
python data/build_datasets.py          # 1. Bangun dataset (historis + holdout 2026)
python run_experiments.py --pipeline   # 2. Eksperimen faktorial lima fase
python -m src.generate_narratives      # 3. Narasi LLM per model per bulan
python -m src.generate_xai_charts      # 4. Gambar SHAP/LIME/waterfall
```

Modul analisis lanjutan (uji ablasi, diagnostik generalisasi, analisis galat,
konsistensi dan konsolidasi XAI) dijalankan terpisah. Daftar perintahnya ada pada
[`Tugas Akhir/README.md`](Tugas%20Akhir/README.md).

### Konfigurasi LLM

Modul narasi membaca kredensial dari berkas `.env` di dalam folder
`Tugas Akhir/`. Berkas ini tidak disertakan di repositori dan harus dibuat
sendiri:

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=<kunci Anda>
LLM_MODEL=gpt-4o
```

Bila LLM tidak tersedia, narasi mundur otomatis ke ringkasan deterministik dari
tiga kontributor SHAP teratas.

### Catatan reproduksibilitas

Angka kinerja bersumber dari jalur evaluasi kanonis
`src/evaluation_core.py` dengan pembagian data kronologis 60/20/20 yang tetap,
ensembel tiga *seed* yang dibekukan untuk LSTM/BiLSTM, serta `random_state=42`
untuk Random Forest dan XGBoost. Entri pada
`results/experiments/best_results_registry.json` ditandai `"locked": true`
sehingga menjalankan ulang pipeline tidak menimpanya.

XGBoost **tidak portabel antar sistem operasi** — seluruh angka diseragamkan ke
lingkungan macOS. Detail ada di [`Tugas Akhir/README.md`](Tugas%20Akhir/README.md).

---

## Lisensi

Lihat [LICENSE](LICENSE). Kode ini disusun untuk keperluan tugas akhir di
Institut Teknologi Bandung, 2026.
