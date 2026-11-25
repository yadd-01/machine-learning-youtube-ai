# PRD: Analisis Komentar YouTube terhadap Perkembangan Artificial Intelligence (AI) Menggunakan Text Mining dan WordCloud

| Item | Keterangan |
|---|---|
| Mata kuliah | Machine Learning (tugas kelompok) |
| Versi dokumen | 1.0 |
| Status | Final, seluruh fase (1–9) selesai dan terdokumentasi |
| Bahasa data | Bahasa Indonesia |

---

## 1. Ringkasan

Proyek ini mengumpulkan komentar pengguna YouTube tentang perkembangan AI, membersihkan dan memprosesnya dengan teknik text mining, lalu memvisualisasikan kata yang paling sering muncul dalam bentuk WordCloud dan grafik frekuensi. Seluruh hasil harus berasal dari data yang benar-benar diambil, bukan data contoh.

## 2. Latar Belakang Masalah

Komentar YouTube berupa teks tidak terstruktur: penuh bahasa gaul, singkatan, emoji, URL, dan spam. Membaca ribuan komentar secara manual tidak efisien. Dibutuhkan pipeline otomatis yang mengumpulkan, membersihkan, dan merangkum komentar tersebut.

## 3. Tujuan

1. Mengumpulkan minimal 2.000 komentar YouTube dengan YouTube Data API v3.
2. Menyimpan hasil scraping sebagai dataset CSV (UTF-8).
3. Melakukan data cleaning dan text preprocessing Bahasa Indonesia.
4. Mengidentifikasi noise dan membandingkan data sebelum dan sesudah pengolahan.
5. Menghitung frekuensi kata dan membuat WordCloud (raw, cleaned, preprocessed, noise).
6. Menyusun workflow dan dokumentasi akhir yang dapat dipresentasikan.

## 4. Ruang Lingkup

### 4.1 Dalam lingkup
- Scraping komentar YouTube (resmi via API).
- EDA dataset mentah.
- Cleaning, normalisasi, stopword removal, stemming (Sastrawi).
- Noise analysis.
- Word frequency, WordCloud, bar chart Top 20.
- Workflow dan dokumentasi.

### 4.2 Di luar lingkup
- Sentiment analysis, klasifikasi, training model, prediksi.
- Scraping non-resmi (library atau HTML scraping).
- Deployment aplikasi web atau dashboard.

## 5. Pengguna dan Pemangku Kepentingan

| Peran | Kebutuhan |
|---|---|
| Anggota kelompok | Kode berjalan, terstruktur, mudah dijalankan ulang |
| Dosen | Bukti pemenuhan seluruh poin tugas (dataset, WordCloud, workflow, dokumentasi) |

## 6. Kebutuhan Fungsional

### 6.1 Pengumpulan Data (Phase 2)

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-01 | Menerima daftar URL atau Video ID YouTube dari file konfigurasi | Wajib |
| FR-02 | Mengekstrak Video ID dari berbagai format URL | Wajib |
| FR-03 | Mengambil komentar teratas (top-level) dengan pagination sampai habis atau target tercapai | Wajib |
| FR-04 | Menyimpan kolom: `video_id`, `comment_id`, `author`, `text`, `published_at`, `like_count` | Wajib |
| FR-05 | Menghindari duplikat berdasarkan `comment_id` | Wajib |
| FR-06 | Melanjutkan ke video berikutnya jika komentar video kurang | Wajib |
| FR-07 | Menangani error API (kuota habis, komentar dinonaktifkan, video tidak ditemukan, error jaringan) tanpa menghentikan seluruh proses | Wajib |
| FR-08 | Menampilkan jumlah komentar terkumpul per video dan total; tidak menyatakan target tercapai sebelum diperiksa | Wajib |
| FR-09 | Membaca API Key dari environment variable (`.env`), bukan ditulis di kode | Wajib |
| FR-10 | Output `dataset/raw_comments.csv` dengan encoding UTF-8 | Wajib |

### 6.2 EDA (Phase 3)

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-11 | Menghitung jumlah komentar, data kosong, dan duplikat | Wajib |
| FR-12 | Menghitung panjang karakter dan jumlah kata per komentar | Wajib |
| FR-13 | Menampilkan contoh komentar dan histogram distribusi panjang | Wajib |
| FR-14 | Mengidentifikasi indikasi noise (URL, mention, emoji, simbol berlebihan) | Wajib |

### 6.3 Data Cleaning (Phase 4)

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-15 | Case folding | Wajib |
| FR-16 | Hapus URL, mention, simbol hashtag | Wajib |
| FR-17 | Hapus emoji, tanda baca, karakter tidak perlu, angka tidak relevan | Wajib |
| FR-18 | Hapus spasi berlebih, tangani komentar kosong dan duplikat | Wajib |
| FR-19 | Tampilkan perbandingan teks sebelum dan sesudah cleaning | Wajib |
| FR-20 | Output `dataset/cleaned_comments.csv` | Wajib |

Catatan: kata tidak baku tidak boleh dihapus sembarangan. Penanganannya dilakukan di tahap normalisasi.

### 6.4 Text Preprocessing (Phase 5)

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-21 | Tokenization | Wajib |
| FR-22 | Normalisasi kata tidak baku memakai kamus yang bisa diedit manual (`normalization.py` atau file kamus terpisah) | Wajib |
| FR-23 | Stopword removal dengan daftar yang dapat disesuaikan; kata penting konteks AI tidak boleh terhapus | Wajib |
| FR-24 | Stemming memakai Sastrawi | Wajib |
| FR-25 | Contoh sebelum dan sesudah untuk setiap langkah | Wajib |
| FR-26 | Output `dataset/preprocessed_comments.csv` | Wajib |

### 6.5 Noise Analysis (Phase 6)

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-27 | Mendeteksi spam, komentar kosong, simbol berlebihan, URL tidak relevan, kata berulang tanpa informasi | Wajib |
| FR-28 | Membandingkan statistik sebelum dan sesudah noise removal | Wajib |
| FR-29 | Tidak menganggap semua kata informal sebagai noise | Wajib |
| FR-30 | Menghasilkan dataset noise dan WordCloud noise | Opsional |

### 6.6 Visualisasi (Phase 7)

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-31 | WordCloud raw, cleaned, preprocessed (PNG) | Wajib |
| FR-32 | WordCloud noise (PNG) | Opsional |
| FR-33 | Bar chart Top 20 kata (`top_20_words.png`) | Wajib |
| FR-34 | Semua visualisasi dibuat dari dataset hasil scraping nyata, lengkap dengan judul dan ukuran jelas | Wajib |

### 6.7 Workflow dan Dokumentasi (Phase 8-9)

| ID | Kebutuhan | Prioritas |
|---|---|---|
| FR-35 | Diagram workflow (`workflow/workflow.png`) | Wajib |
| FR-36 | Dokumentasi akhir 13 bagian (pendahuluan sampai saran) berbasis hasil nyata | Wajib |
| FR-37 | `README.md` berisi instalasi dan cara menjalankan dari awal sampai akhir | Wajib |

## 7. Kebutuhan Non-Fungsional

| ID | Kebutuhan |
|---|---|
| NFR-01 | Kode berjalan di Python 3.9+ (Google Colab atau VS Code) |
| NFR-02 | Kode modular, setiap modul punya satu tanggung jawab, dengan komentar penjelas |
| NFR-03 | Hasil dapat direproduksi: urutan menjalankan jelas, path relatif terhadap root proyek |
| NFR-04 | API Key tidak pernah masuk repository (`.env` di `.gitignore`, sediakan `.env.example`) |
| NFR-05 | Penggunaan kuota API efisien (`maxResults=100`, hentikan saat target tercapai) |
| NFR-06 | Encoding UTF-8 di semua file CSV |
| NFR-07 | Library stabil dan resmi; hindari scraping tidak resmi |

## 8. Spesifikasi Data

### 8.1 raw_comments.csv

| Kolom | Tipe | Keterangan |
|---|---|---|
| video_id | string | ID video sumber |
| comment_id | string | ID unik komentar (kunci deduplikasi) |
| author | string | Nama channel pengomentar |
| text | string | Isi komentar mentah |
| published_at | datetime | Waktu komentar dibuat |
| like_count | int | Jumlah like komentar |

### 8.2 cleaned_comments.csv
Kolom raw ditambah `text_cleaned`.

### 8.3 preprocessed_comments.csv
Kolom cleaned ditambah `tokens`, `tokens_normalized`, `tokens_no_stopword`, `text_preprocessed` (hasil stemming).

## 9. Kriteria Penerimaan

| No | Kriteria | Cara verifikasi |
|---|---|---|
| 1 | `raw_comments.csv` berisi ≥ 2.000 baris unik | Hitung baris dan `comment_id` unik, tampilkan di log |
| 2 | Tidak ada API Key di kode atau repository | Cek `.gitignore` dan pencarian string di kode |
| 3 | Cleaning dan preprocessing menghasilkan file output yang benar | Cek keberadaan file dan contoh sebelum/sesudah |
| 4 | Kamus normalisasi bisa diedit tanpa mengubah logika kode | Ubah satu entri kamus lalu jalankan ulang |
| 5 | 3 WordCloud wajib + Top 20 tersimpan di `results/` | Cek file PNG |
| 6 | Seluruh angka di dokumentasi sesuai output program | Bandingkan dengan log dan CSV |
| 7 | Workflow dan dokumentasi lengkap | Review oleh kelompok |

## 10. Deliverables

1. Dataset: `raw_comments.csv`, `cleaned_comments.csv`, `preprocessed_comments.csv`.
2. Kode: `scraping/`, `preprocessing/`, `analysis/`, `visualization/`.
3. Gambar: 4 WordCloud (satu opsional) dan `top_20_words.png`.
4. `workflow/workflow.png`.
5. `notebook/analysis.ipynb`.
6. `README.md` dan dokumentasi akhir.

## 11. Rencana Fase

| Fase | Aktivitas | Output utama | Syarat lanjut |
|---|---|---|---|
| 1 | Project planning | Rancangan, PRD, arsitektur | Dikonfirmasi |
| 2 | Scraping | `raw_comments.csv` | Jumlah ≥ 2.000 terverifikasi |
| 3 | EDA | Laporan dataset mentah | Hasil diperiksa |
| 4 | Cleaning | `cleaned_comments.csv` | Perbandingan sebelum/sesudah diperiksa |
| 5 | Preprocessing | `preprocessed_comments.csv` | Contoh tiap tahap diperiksa |
| 6 | Noise analysis | Analisis dan dataset noise | Diperiksa |
| 7 | WordCloud dan frekuensi | 5 gambar di `results/` | Diperiksa |
| 8 | Workflow | `workflow.png` | Diperiksa |
| 9 | Dokumentasi | Laporan akhir | Final |

Aturan kerja: tidak melompati fase sebelum hasil fase sebelumnya diperiksa.

## 12. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Kuota API harian habis | Scraping terhenti | Batasi request, simpan progres berkala, lanjutkan keesokan hari |
| Komentar video kurang dari target | Total < 2.000 | Siapkan 8-10 video cadangan, otomatis lanjut ke video berikutnya |
| Komentar dinonaktifkan di sebuah video | Video gagal diambil | Tangkap error, lewati, catat di log |
| Kata gaul salah dinormalisasi | Makna bergeser | Kamus manual, tinjau Top frekuensi sebelum finalisasi |
| Stemming Sastrawi terlalu agresif | Kata teknis AI rusak | Daftar pengecualian (misalnya "chatgpt", "ai") |
| Komentar didominasi Bahasa Inggris | Stopword dan stemming Indonesia kurang efektif | Pilih video berbahasa Indonesia; dokumentasikan batasan jika ada komentar campuran |
| Duplikat atau spam menurun jumlah setelah cleaning | Dataset bersih < 2.000 | Kumpulkan lebih dari 2.000 komentar mentah |

## 13. Asumsi

- Anggota kelompok memiliki akun Google untuk membuat API Key.
- Video yang dipilih bersifat publik dan komentarnya aktif.
- Data bersumber dari komentar publik dan dipakai untuk tujuan akademik.

## 14. Pertanyaan Terbuka

1. Daftar final video YouTube (topik, jumlah video, bahasa).
2. Apakah laporan akhir dikumpulkan sebagai PDF atau Word?
3. Pembagian peran anggota kelompok per fase.
