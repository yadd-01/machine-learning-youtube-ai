# Analisis Komentar YouTube tentang AI (Text Mining + WordCloud)

Tugas kelompok mata kuliah Machine Learning. Proyek ini mengambil komentar publik YouTube
bertopik perkembangan AI, membersihkannya, lalu memvisualisasikan kata yang paling sering
muncul (WordCloud dan bar chart Top 20). Seluruh hasil berasal dari data hasil scraping nyata.

Dokumen acuan:

- `PRD.md` — kebutuhan produk dan kriteria penerimaan
- `ARCHITECTURE.md` — rancangan teknis dan alur data
- `LAPORAN.md` — laporan akhir 13 bagian (pendahuluan sampai saran)
- `notebook/analysis.ipynb` — ringkasan hasil dalam bentuk notebook

## Status pengerjaan

| Fase | Aktivitas | Status |
|---|---|---|
| 1 | Project planning (PRD + arsitektur) | Selesai |
| 2 | Scraping komentar | Selesai (3.000 komentar dari 5 video) |
| 3 | EDA | Selesai |
| 4 | Cleaning | Selesai |
| 5 | Preprocessing (normalisasi, stopword, stemming) | Selesai |
| 6 | Noise analysis | Selesai |
| 7 | WordCloud & frekuensi kata | Selesai |
| 8 | Workflow | Selesai |
| 9 | Dokumentasi akhir | Selesai (`LAPORAN.md` + notebook) |

## Struktur folder

```
.
├── config/
│   └── videos.txt          # daftar URL/Video ID yang di-scrape
├── dataset/                # hasil CSV tiap tahap (dibuat otomatis)
├── scraping/
│   └── scraping.py         # Phase 2: ambil komentar via YouTube Data API v3
├── preprocessing/
│   ├── cleaning.py         # Phase 4: case folding, hapus noise, output cleaned_comments.csv
│   ├── normalization.py    # Phase 5: pemuat kamus normalisasi
│   ├── normalization_dict.csv  # kamus kata tidak baku -> baku (editable)
│   ├── stopwords_id.txt    # daftar stopword (editable)
│   ├── stem_exceptions.csv # kata yang tidak boleh di-stem (editable)
│   ├── protected_words.py  # istilah yang tidak boleh dibuang/di-stem
│   └── preprocessing.py    # Phase 5: tokenize, normalisasi, stopword, stemming
├── analysis/
│   ├── eda.py              # Phase 3: statistik dataset mentah + histogram
│   ├── noise_analysis.py   # Phase 6: label noise + dataset noise_comments.csv
│   └── word_frequency.py   # Phase 7: frekuensi kata per tahap
├── visualization/
│   └── make_wordcloud.py   # Phase 7: 4 WordCloud + bar chart Top 20
├── workflow/
│   ├── make_workflow.py    # Phase 8: diagram alur pipeline (PNG)
│   └── workflow.png
├── results/                # PNG WordCloud, bar chart, tabel frekuensi, ringkasan
├── notebook/
│   └── analysis.ipynb      # Phase 9: ringkasan hasil + visualisasi
├── requirements.txt
├── .env.example
├── .gitignore
├── PRD.md
├── ARCHITECTURE.md
├── LAPORAN.md              # Phase 9: laporan akhir 13 bagian
└── README.md
```

## Prasyarat

- Python 3.9 atau lebih baru (sudah diuji pada Python 3.13)
- Akun Google untuk membuat API Key
- Koneksi internet

## Membuat YouTube Data API Key

1. Buka <https://console.cloud.google.com/> dan login dengan akun Google.
2. Buat project baru (misalnya `ml-youtube-ai`).
3. Masuk ke **APIs & Services → Library**, cari **YouTube Data API v3**, lalu klik **Enable**.
4. Masuk ke **APIs & Services → Credentials → Create credentials → API key**.
5. Salin API Key yang muncul.
6. (Disarankan) Klik **Edit API key** → **API restrictions** → pilih **YouTube Data API v3** saja,
   lalu **Save**. Ini membatasi key agar tidak bisa dipakai untuk layanan lain.
7. Buat file `.env` di root proyek:

   ```bash
   # Windows PowerShell
   Copy-Item .env.example .env
   ```

   Buka `.env` dan isi:

   ```
   YOUTUBE_API_KEY=AIza...key_anda
   TARGET_COMMENTS=3000
   ```

   Target minimum tugas adalah 2.000 komentar; proyek ini memakai 3.000 agar
   dataset lebih besar dan tetap memenuhi syarat setelah cleaning.

> `.env` sudah masuk `.gitignore`. Jangan pernah menulis API Key di kode, notebook, atau commit.

## Instalasi

```bash
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktifkan (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# 3. Install dependensi
pip install -r requirements.txt
```

## Menjalankan scraping (Phase 2)

Cek dulu daftar video tanpa memanggil API:

```bash
python scraping/scraping.py --dry-run
```

Jalankan scraping:

```bash
python scraping/scraping.py
```

Perilaku penting:

- Komentar diambil dengan `maxResults=100` dan pagination, diurutkan berdasarkan relevansi.
- Duplikat dibuang berdasarkan `comment_id`, termasuk terhadap hasil run sebelumnya.
- Progres disimpan ke `dataset/raw_comments.csv` **setiap selesai satu video**, jadi aman
  dihentikan kapan saja.
- Jika kuota API habis, skrip berhenti dengan rapi (exit code `2`) dan bisa dilanjutkan
  besok dengan perintah yang sama.
- Video dengan komentar dinonaktifkan atau tidak ditemukan akan dilewati dan dicatat di log.
- Di akhir, skrip memverifikasi jumlah `comment_id` unik sebelum menyatakan target tercapai.

Opsi lain:

```bash
python scraping/scraping.py --target 2500          # target berbeda
python scraping/scraping.py --videos config/lain.txt
python scraping/scraping.py --fresh                # mulai dari nol, hapus CSV lama
```

Kode keluar: `0` target tercapai, `1` kesalahan konfigurasi/API, `2` kuota habis,
`3` target belum tercapai (tambahkan video cadangan di `config/videos.txt`).

### Mengubah daftar video

Edit `config/videos.txt`: satu URL atau Video ID per baris. Baris berawalan `#` diabaikan.
Format yang didukung: `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`, atau ID polos.
Jika total komentar belum mencapai target (3.000 pada proyek ini), aktifkan video cadangan di bagian bawah file
(hapus tanda `#` di awal baris) lalu jalankan ulang perintah scraping.

### Output

`dataset/raw_comments.csv` (UTF-8) dengan kolom:

| Kolom | Keterangan |
|---|---|
| `video_id` | ID video sumber |
| `comment_id` | ID unik komentar (kunci deduplikasi) |
| `author` | Nama channel pengomentar |
| `text` | Isi komentar mentah |
| `published_at` | Waktu komentar dibuat (ISO 8601) |
| `like_count` | Jumlah like komentar |

Log lengkap tersimpan di `scraping.log` (tidak dikomit).

## Menjalankan tahap berikutnya

EDA (Phase 3):

```bash
python analysis/eda.py
```

Menghasilkan ringkasan di konsol, `results/eda_summary.txt`, dan
`results/eda_length_distribution.png`.

Cleaning (Phase 4):

```bash
python preprocessing/cleaning.py
```

Menghasilkan `dataset/cleaned_comments.csv` (kolom raw + `text_cleaned`),
`results/cleaning_summary.txt`, serta contoh sebelum/sesudah di konsol.

Preprocessing (Phase 5):

```bash
python preprocessing/preprocessing.py
```

Menghasilkan `dataset/preprocessed_comments.csv` (kolom cleaned + `tokens`,
`tokens_normalized`, `tokens_no_stopword`, `text_preprocessed`) dan
`results/preprocessing_summary.txt`.

- Kamus normalisasi: `preprocessing/normalization_dict.csv` (edit manual, satu token per baris).
- Stopword: `preprocessing/stopwords_id.txt` (edit manual, satu kata per baris).
- Pengecualian stemming: `preprocessing/stem_exceptions.csv` (satu token per baris).
  Dipakai untuk kata yang maknanya rusak bila di-stem Sastrawi, misalnya
  `perkembangan` (bukan `kembang`), `perusahaan` (bukan `usaha`), dan
  `setuju`/`tujuan` (bukan `tuju`).
- Kata dasar `rasa` sengaja dijadikan stopword. Bentuk turunannya
  (`terasa`, `merasa`, `rasakan`, `rasanya`) dipertahankan apa adanya agar
  makna kalimat tidak hilang.
- Istilah AI yang tidak boleh dibuang/di-stem: `PROTECTED_WORDS` di `preprocessing/protected_words.py`.
- Stemming hanya untuk kata unik, diparalelkan, dan di-cache ke
  `preprocessing/stem_cache.json`. Run pertama bisa 2–3 menit; run berikutnya hampir instan.
  Hapus file cache jika kamus/aturan berubah dan ingin menghitung ulang dari nol.

Noise analysis (Phase 6):

```bash
python analysis/noise_analysis.py
```

Menghasilkan `dataset/noise_comments.csv` (komentar berlabel noise beserta jenisnya)
dan `results/noise_summary.txt` (jumlah per jenis noise + perbandingan sebelum/sesudah).
Dataset utama tidak diubah oleh tahap ini.

Tahap visualisasi (Phase 7):

```bash
python analysis/word_frequency.py
python visualization/make_wordcloud.py
```

- `word_frequency.py` menghasilkan `results/word_frequencies.csv` (kolom
  `stage, word, count` untuk tahap raw/cleaned/preprocessed/noise),
  `results/top_20_words.csv`, dan `results/word_frequency_summary.txt`.
- `make_wordcloud.py` menghasilkan `results/wordcloud_raw.png`,
  `wordcloud_cleaned.png`, `wordcloud_preprocessed.png`, `wordcloud_noise.png`,
  `top_20_words.png`, dan `results/wordcloud_summary.txt`.
- WordCloud memakai 2400×1500 px, maksimal 200 kata, hanya kata yang muncul ≥ 2 kali,
  dan `random_state=42` agar hasil reproducible.
- WordCloud noise memakai `text_cleaned` sebagai fallback untuk komentar yang gugur
  saat cleaning, sehingga jumlah komentarnya tetap 314.

Nama file sengaja `make_wordcloud.py`, bukan `wordcloud.py`, agar tidak menimpa
library `wordcloud` saat proses impor.

Workflow (Phase 8):

```bash
python workflow/make_workflow.py
```

Menghasilkan `workflow/workflow.png` (2.400 × 1.500 px): alur scraping → EDA →
cleaning → preprocessing → noise analysis → frekuensi kata → WordCloud/Top 20 →
dokumentasi. Angka pada diagram dibaca langsung dari dataset dan file
konfigurasi, jadi otomatis ikut berubah bila data diperbarui.

Dokumentasi akhir (Phase 9):

- `LAPORAN.md` — laporan 13 bagian: pendahuluan, landasan teori, metodologi,
  pengumpulan data, EDA, cleaning, preprocessing, noise analysis, frekuensi kata,
  visualisasi, workflow, pembahasan, serta kesimpulan dan saran. Seluruh angka
  diambil dari output program dan file di `dataset/` serta `results/`.
- `notebook/analysis.ipynb` — notebook ringkasan yang membaca dataset dan hasil
  analisis, lalu menampilkan statistik, tabel Top 20, WordCloud, dan workflow.
  Notebook tidak melakukan scraping atau stemming ulang.

## Troubleshooting

| Gejala | Penyebab dan solusi |
|---|---|
| `YOUTUBE_API_KEY belum diisi` | Belum ada `.env` atau nilainya masih placeholder. Ikuti bagian API Key di atas. |
| `Kuota API habis` | Kuota harian 10.000 unit habis. Tunggu reset (tengah malam waktu Pasifik) lalu jalankan perintah yang sama; progres otomatis dilanjutkan. |
| `komentar tidak dapat diakses` | Video menonaktifkan komentar. Scraper melewatinya; ganti dengan video lain di `config/videos.txt`. |
| Target belum tercapai | Tambahkan video cadangan di `config/videos.txt`, lalu jalankan ulang tanpa `--fresh`. |
| `File daftar video tidak ditemukan` | Jalankan dari root proyek atau isi opsi `--videos` dengan path yang benar. |

## Catatan metodologis

- Hanya komentar top-level yang diambil (balasan tidak termasuk), sesuai `ARCHITECTURE.md`.
- Data bersumber dari komentar publik dan dipakai untuk tujuan akademik.
- Kuota YouTube Data API v3 gratis 10.000 unit/hari; satu halaman komentar berbiaya 1 unit,
  sehingga pengumpulan 3.000 komentar hanya membutuhkan sekitar 30 unit.
