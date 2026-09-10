# Prompt Revisi Paper IEEE — Sinkronisasi dengan Buku TA Final

> Dipakai untuk menyelaraskan `Paper IEEE - BIRate XAI-LLM/paper_birate_xai_llm.tex`
> (draf 16 Juli 2026) dengan buku TA final yang sudah diserahkan
> (`Tugas Akhir - Dokumen/TA.pdf`, 212 halaman, build 7 Agustus 2026).

---

## Peran dan bahasa

Berperan ganda sebagai penulis paper dan reviewer akademis. Jawab dan berdiskusi
dalam Bahasa Indonesia, tetapi **seluruh isi paper tetap ditulis dalam Bahasa
Inggris** sesuai format IEEE conference.

## Berkas

| Peran | Lokasi |
|---|---|
| Paper yang direvisi | `Paper IEEE - BIRate XAI-LLM/paper_birate_xai_llm.tex` |
| Template resmi | `Conference-LaTeX-template_10-17-19/` (IEEEtran.cls V1.8b) |
| Sumber kebenaran naratif | `Tugas Akhir - Dokumen/TA.pdf` dan berkas `.tex` babnya |
| Sumber kebenaran angka | `Tugas Akhir/results/experiments/analysis/` |
| Gambar siap pakai | `Tugas Akhir - Dokumen/images/` |

## Batasan keras

1. **Jangan mengarang angka.** Setiap angka harus tertelusur ke artefak di
   `results/experiments/analysis/` atau ke tabel di buku. Bila sebuah klaim
   tidak didukung artefak, laporkan, jangan diam-diam diubah.
2. **Paper tidak boleh bertentangan dengan buku.** Buku sudah diserahkan dan
   menjadi rujukan. Bila paper dan buku berbeda, paper yang menyesuaikan.
3. Build serial saja: `pdflatex → bibtex (bila perlu) → pdflatex → pdflatex`.
   Jangan menjalankan dua build bersamaan.
4. Batas halaman menyesuaikan konferensi tujuan. Draf saat ini 4 halaman.
   Bila penambahan konten melewati batas, potong dari bagian yang paling
   sedikit menambah nilai (Related Work dan Limitations dapat dipadatkan).

---

## 1. Koreksi angka yang salah (WAJIB, prioritas tertinggi)

### 1.1 XGBoost tidak portabel antar-OS

Draf memakai angka Windows. Seluruh buku sudah diseragamkan ke macOS.

| Lokasi | Draf sekarang | Harus menjadi |
|---|---|---|
| Table I, baris XGBoost | RMSE 0.1960 · MAPE 2.88 · R² 0.7965 | **RMSE 0.2207 · MAPE 3.18 · R² 0.7420** |

Tambahkan satu kalimat di Data and Methodology yang menyatakan seluruh angka
dilaporkan dari satu lingkungan (macOS), karena XGBoost menghasilkan nilai
berbeda pada Windows meskipun *seed* dan data identik.

### 1.2 Tabel rezim kebijakan salah kolom dan salah *hit rate*

Draf Table II menukar kolom *hold* dan *change* untuk XGBoost, mengosongkan dua
baris, dan salah menulis *hit rate* XGBoost. Ganti seluruh tabel dengan angka
kanonis berikut (sumber: `ablation_features.csv` baris "Konfigurasi penuh" dan
Tabel VI.11 buku).

| Model | RMSE (change) | RMSE (hold) | Hit (change) | Hit (hold) | Hit (all) |
|---|---|---|---|---|---|
| Random Forest | 0.214 | 0.110 | 33% (3/9) | 63% (17/27) | 56% |
| BiLSTM | 0.162 | 0.141 | 67% (6/9) | 63% (17/27) | 64% |
| LSTM | 0.179 | 0.156 | 67% (6/9) | 37% (10/27) | 44% |
| XGBoost | 0.241 | 0.213 | 67% (6/9) | 33% (9/27) | 42% |

Perbaiki pula kalimat di subbagian The Persistence Trap: *hit rate* XGBoost pada
bulan perubahan adalah **67%**, bukan 56%, dan *hit rate* keseluruhannya **42%**,
bukan 39%.

### 1.3 Simpangan baku antarpelatihan

Ganti `±0.012 (BiLSTM)` menjadi **`±0.009 (BiLSTM)`** di subbagian Models, dan
`(±0.012--0.015)` menjadi **`(±0.009--0.015)`** di Predictive Performance.
Sumber: `seed_stability.csv`, sepuluh *seed* dan seluruh ensembel tiga-*seed*
yang dapat dibentuk darinya. Perbaiki juga deskripsinya: bukan "sepuluh kali
pengulangan ensembel", melainkan sepuluh *seed* yang menghasilkan 120 ensembel
tiga-*seed*.

### 1.4 LIME dilaporkan terbalik

Ini koreksi paling serius. Draf menulis `local R² = 0.6847` dan "LIME agrees
locally" — buku menemukan sebaliknya.

Fakta menurut `lime_r2_per_bulan.csv` dan Bab VI buku:

- R² surrogat LIME pada Juli 2025 hanya **0.0022**
- Di seluruh 36 bulan uji, rentangnya **0.0022 sampai 0.6916** dengan **median 0.0247**
- Hanya tiga bulan (Oktober sampai Desember 2025) yang melewati 0.5
- LIME dan SHAP hanya sepakat pada satu hal, yaitu persistensi BI-Rate sebagai
  kontributor terkuat. LIME menempatkan Perubahan Federal Funds Rate di urutan
  kedua, sedangkan Pertumbuhan Kredit (pendorong makro teratas menurut SHAP)
  baru muncul di urutan keenam.

Tulis ulang bagian ini sesuai posisi buku: LIME berperan sebagai pemeriksa arah,
bukan sebagai penjelasan lokal yang setara otoritasnya dengan SHAP. Sebutkan
penyebabnya, yaitu bentuk fungsi Random Forest yang bertingkat membuat bidang
linear sulit mencocokkan ketetanggaan sebuah titik.

---

## 2. Keputusan struktural yang harus diikuti

### 2.1 Pembanding naif sudah dibuang dari buku

Sejak 4 Agustus, seluruh pembanding *random walk* dan uji Diebold-Mariano
**dihapus dari buku** atas keputusan penulis. Penggantinya adalah pemisahan
bulan-perubahan versus bulan-tahan.

Draf paper masih menyandarkan argumennya pada pembanding naif di empat tempat,
yaitu abstrak, kontribusi nomor 2, subbagian The Persistence Trap, dan
kesimpulan. Ada dua pilihan, dan penulis perlu memilih salah satu secara sadar:

1. **Selaraskan dengan buku.** Buang seluruh rujukan ke *naive persistence*, dan
   sandarkan argumen "akurasi agregat menyesatkan" sepenuhnya pada pemisahan
   jenis bulan serta uji ablasi. Ini pilihan yang konsisten dan direkomendasikan.
2. **Pertahankan di paper saja.** Boleh, karena kodenya masih ada
   (`naive_benchmark.csv`), tetapi paper lalu memuat analisis yang tidak ada di
   buku. Bila dipilih, angkanya harus diambil dari artefak, bukan ditulis ulang
   dari draf lama.

Ajukan pertanyaan ini ke penulis sebelum menyunting bagian tersebut.

### 2.2 Hasil kuat yang belum masuk paper

Buku memuat beberapa temuan yang lebih orisinal daripada perbandingan akurasi,
dan tidak satu pun ada di draf paper. Masukkan sebanyak yang muat.

**Prioritas 1 — Uji ablasi *leave-one-out*.** Ini temuan terkuat paper.
Sumber: `ablation_per_feature.csv`, `ablation_features.csv`, `ablation_components.csv`.

- Random Forest dan XGBoost bertumpu pada satu fitur saja. Melepas BI-Rate t−1
  menaikkan RMSE Random Forest dari 0.1434 menjadi **0.4087**, dan XGBoost dari
  0.2207 menjadi 0.2563.
- Kedua model *deep learning* berperilaku sebaliknya. Melepas BI-Rate t−1
  praktis tidak berpengaruh (LSTM 0.1620 ke 0.1689, BiLSTM 0.1463 ke 0.1378,
  keduanya di bawah ambang derau), sedangkan melepas Federal Funds Rate
  menaikkan RMSE LSTM menjadi **0.2805**. Penyebabnya penambatan residual.
- Varian hanya-*lag* lebih akurat secara agregat (RF 0.1174, BiLSTM 0.1172,
  LSTM 0.1205, XGBoost 0.1631) tetapi akurasi arah bulan-perubahannya runtuh
  dari 33–67% menjadi **0–22%**.
- Melepas seluruh indikator makro memperbaiki RMSE agregat tetapi menurunkan
  akurasi arah bulan-perubahan pada keempat model tanpa kecuali.
- Sepuluh dari 17 fitur justru menurunkan RMSE rata-rata bila dibuang
  (rata-rata konfigurasi penuh 0.1681). Laporkan ini apa adanya.
- Ablasi komponen: mematikan jangkar residual menaikkan RMSE BiLSTM dari 0.1463
  menjadi **0.4876** dan LSTM menjadi 0.3331. Ensembel tiga-*seed* hampir tidak
  mengubah RMSE tetapi menstabilkan keputusan arah.

**Prioritas 2 — Ketakbersepakatan atribusi antarmodel.**
Sumber: `xai_konsolidasi.csv`, `xai_pangsa_persistensi.csv`, `xai_stability.csv`.

- Pangsa persistensi sangat berbeda antararsitektur, yaitu Random Forest 75.5%,
  XGBoost 48.0%, BiLSTM 10.3%, LSTM 6.7%.
- Setelah persistensi dikeluarkan dan kontribusi sisanya dinormalisasi, keempat
  model tetap tidak sepakat mengenai urutan pendorong makro (Spearman −0.31
  sampai 0.51, tidak ada yang bermakna secara statistik).
- Atribusi Random Forest eksak dan reprodusibel penuh, sedangkan atribusi LSTM
  berubah antarpengulangan. Random Forest karena itu dipakai sebagai rujukan.
- Peringkat gabungan pendorong makro: Pertumbuhan Kredit, Suku Bunga Riil,
  M2, Arah Kebijakan Suku Bunga.
- Besaran SHAP antara model level dan model residual tidak dapat dibandingkan
  langsung. Jelaskan alasannya, sebab ini kekeliruan pembacaan yang wajar.

**Prioritas 3 — Validasi bergulir.** Sumber: `walk_forward.csv`.
Dengan pelatihan ulang di tiap titik awal, RMSE membaik menjadi RF 0.1267,
BiLSTM 0.1299, LSTM 0.1388, sedangkan XGBoost justru memburuk menjadi 0.2260.
Urutan antarmodel tidak berubah.

**Prioritas 4 — Ekstrapolasi rezim.** Sumber: `regime_extrapolation.csv`.
Kedua model berbasis pohon tidak dapat meramalkan di bawah batas bawah target
latih 4.25%, sedangkan kedua model *deep learning* menembusnya. Ini menjelaskan
R² validasi yang negatif pada model pohon.

**Prioritas 5 — Holdout 2026.** Lima bulan Januari sampai Mei 2026 tidak masuk
ketiga himpunan data. Galatnya lebih besar untuk seluruh model (RF 0.1887,
BiLSTM 0.2227, XGBoost 0.2570, LSTM 0.3064), dan satu-satunya bulan-perubahan
di sana (kenaikan Mei 2026 ke 5.25%) dilewatkan keempatnya. Ini bukti bias
searah yang berdiri sendiri.

---

## 3. Perbaikan sitasi

Audit buku menemukan beberapa masalah sitasi. Sebagian terbawa ke paper.

1. `\bibitem{yu2023}` menulis "X. Yu, Z. Chen, and Y. Lu". Nama yang benar
   adalah **X. Yu, Z. Chen, Y. Ling, S. Dong, Z. Liu, dan Y. Lu**. Perbaiki
   menjadi "X. Yu, Z. Chen, Y. Ling, et al."
2. `\bibitem{calik2025}` masih memuat TODO venue. Lengkapi.
3. `breiman2001`, `hochreiter1997`, dan `schuster1997` dipakai di paper tetapi
   tidak ada di daftar pustaka buku. Verifikasi metadata ketiganya secara
   mandiri sebelum finalisasi.
4. Periksa ulang setiap `\cite` dengan cara yang sama seperti audit buku, yaitu
   baca kalimat yang menyitasi lalu pastikan sumbernya memang mendukungnya.
   Khusus `chakraborty2017`, judul yang benar adalah "Machine learning at
   central banks".

---

## 4. TODO administratif yang masih terbuka

Empat TODO di berkas `.tex` belum tertutup, dan semuanya perlu keputusan penulis.

1. Alamat surel penulis (baris 26)
2. Alamat surel pembimbing dan konfirmasi kesediaan sebagai *co-author* (baris 33)
3. Konferensi tujuan serta batas halamannya
4. Urutan penulis

Tanyakan keempatnya ke penulis, jangan diisi sendiri.

---

## 5. Gaya penulisan

- Bahasa Inggris akademis yang wajar, bukan bahasa pemasaran. Hindari kesan
  tulisan mesin.
- Setiap klaim kuantitatif tertelusur ke tabel, gambar, atau artefak.
- Pertahankan nada jujur draf sekarang. Draf ini melaporkan hasil yang tidak
  menguntungkan secara terbuka, dan itu kekuatannya. Jangan diperhalus.
- Nama diri tidak dimiringkan (Random Forest, XGBoost, LSTM, BiLSTM, Optuna,
  GPT-4o). Istilah umum asing boleh dimiringkan.
- Desimal memakai titik karena papernya berbahasa Inggris. Jangan tercampur
  dengan gaya buku yang memakai koma.

---

## 6. Urutan kerja yang disarankan

1. Laporkan seluruh selisih paper terhadap buku sebelum menyunting apa pun.
2. Tanyakan keputusan pembanding naif (bagian 2.1) dan TODO administratif.
3. Kerjakan koreksi angka bagian 1, lalu bangun ulang dan periksa halaman.
4. Tambahkan hasil bagian 2 sesuai prioritas, sampai batas halaman tercapai.
5. Perbaiki sitasi bagian 3.
6. Build bersih, laporkan jumlah halaman dan sisa peringatan.
7. Jangan *commit* atau *push* tanpa persetujuan penulis.
