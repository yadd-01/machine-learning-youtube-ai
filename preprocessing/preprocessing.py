"""Text preprocessing komentar (Phase 5).

Menjawab FR-21 s.d. FR-26 dengan urutan sesuai ARCHITECTURE.md 4.5:
    1. Tokenization
    2. Normalisasi kata tidak baku (kamus manual, FR-22)
    3. Stopword removal (FR-23), dengan daftar PROTECTED_WORDS
    4. Stemming Sastrawi (FR-24) memakai cache per kata

Catatan performa: Sastrawi lambat pada ribuan kata unik. Karena itu stemming
hanya dijalankan untuk kata unik (bukan per kemunculan), diparalelkan dengan
ProcessPoolExecutor, dan hasilnya di-cache ke preprocessing/stem_cache.json
sehingga run berikutnya hampir instan.

Kata yang tidak boleh di-stem karena Sastrawi merusak maknanya (mis.
perusahaan -> usaha, perkembangan -> kembang) diatur di
preprocessing/stem_exceptions.csv dan dipetakan langsung ke bentuk akhir.

Input : dataset/cleaned_comments.csv
Output: dataset/preprocessed_comments.csv
        (kolom cleaned + tokens, tokens_normalized, tokens_no_stopword, text_preprocessed)
        ringkasan di konsol + results/preprocessing_summary.txt

Jalankan dari root proyek:
    python preprocessing/preprocessing.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd

try:  # saat dijalankan sebagai skrip, folder ini ada di sys.path[0]
    from normalization import load_dictionary, normalize_tokens
    from protected_words import PROTECTED_WORDS
except ImportError:  # jaga-jaga saat diimpor dari root proyek
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from normalization import load_dictionary, normalize_tokens
    from protected_words import PROTECTED_WORDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_CSV = PROJECT_ROOT / "dataset" / "cleaned_comments.csv"
PREPROCESSED_CSV = PROJECT_ROOT / "dataset" / "preprocessed_comments.csv"
STOPWORDS_PATH = PROJECT_ROOT / "preprocessing" / "stopwords_id.txt"
STEM_EXCEPTIONS_PATH = PROJECT_ROOT / "preprocessing" / "stem_exceptions.csv"
STEM_CACHE_PATH = PROJECT_ROOT / "preprocessing" / "stem_cache.json"
SUMMARY_PATH = PROJECT_ROOT / "results" / "preprocessing_summary.txt"

MAX_WORKERS = 12          # dibatasi agar tidak menghabiskan RAM/CPU berlebihan
MAX_STEM_LENGTH = 40      # token lebih panjang dari ini (spam karakter) tidak di-stem

# PROTECTED_WORDS diimpor dari protected_words.py (dipakai juga oleh noise_analysis.py).

logger = logging.getLogger("preprocessing")

_STEMMER = None


# ---------------------------------------------------------------------------
# Stemmer: cache, paralelisasi, dan worker
# ---------------------------------------------------------------------------
def _get_stemmer():
    """Buat stemmer Sastrawi sekali per proses (lazy)."""
    global _STEMMER
    if _STEMMER is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        _STEMMER = StemmerFactory().create_stemmer()
    return _STEMMER


def _init_stemmer_worker() -> None:
    """Initializer untuk setiap proses worker."""
    _get_stemmer()


def _stem_chunk(words: list[str]) -> dict[str, str]:
    """Stem satu potongan kata (dijalankan di proses worker)."""
    stemmer = _get_stemmer()
    return {word: stemmer.stem(word) for word in words}


def load_stem_cache(path: Path = STEM_CACHE_PATH) -> dict[str, str]:
    """Muat cache hasil stemming dari run sebelumnya (jika ada)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Cache stemming tidak terbaca (%s); diabaikan.", error)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def save_stem_cache(cache: dict[str, str], path: Path = STEM_CACHE_PATH) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


def stem_missing_tokens(tokens: set[str], cache: dict[str, str]) -> None:
    """Isi cache untuk kata yang belum pernah di-stem.

    Strategi: sequential untuk sedikit kata, paralel untuk banyak kata.
    Jika proses paralel gagal (misalnya lingkungan melarang subprocess),
    otomatis jatuh ke mode sequential.
    """
    missing = sorted(token for token in tokens if token not in cache)
    if not missing:
        logger.info("Semua kata sudah ada di cache stemming.")
        return

    workers = min(os.cpu_count() or 1, MAX_WORKERS)
    if workers < 2 or len(missing) < 200:
        stemmer = _get_stemmer()
        for word in missing:
            cache[word] = stemmer.stem(word)
        logger.info("Stemming sequential: %d kata.", len(missing))
        return

    chunks = [chunk for chunk in (missing[i::workers * 2] for i in range(workers * 2)) if chunk]
    logger.info("Stemming paralel: %d kata dengan %d worker.", len(missing), workers)

    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_stemmer_worker) as executor:
            futures = [executor.submit(_stem_chunk, chunk) for chunk in chunks]
            for done, future in enumerate(as_completed(futures), start=1):
                cache.update(future.result())
                logger.info("  bagian %d/%d selesai (%d kata di cache).", done, len(chunks), len(cache))
    except Exception as error:  # noqa: BLE001 - fallback apa pun penyebabnya
        logger.warning("Stemming paralel gagal (%s); melanjutkan sequential.", error)
        stemmer = _get_stemmer()
        for word in missing:
            if word not in cache:
                cache[word] = stemmer.stem(word)

    save_stem_cache(cache)


# ---------------------------------------------------------------------------
# I/O dan pemrosesan teks
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(console)


def load_stopwords(path: Path = STOPWORDS_PATH) -> set[str]:
    """Muat daftar stopword dari file teks (satu kata per baris)."""
    if not path.exists():
        raise FileNotFoundError(f"Daftar stopword tidak ditemukan: {path}")

    words = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lower()
        if not line or line.startswith("#"):
            continue
        words.add(line)

    if not words:
        raise ValueError(f"{path} kosong; isi minimal satu stopword.")
    return words


def load_stem_exceptions(path: Path = STEM_EXCEPTIONS_PATH) -> dict[str, str]:
    """Muat pengecualian stemming (kata -> bentuk akhir, tanpa di-stem).

    Raises:
        FileNotFoundError: file tidak ada.
        ValueError: kolom tidak lengkap atau entri tidak valid.
    """
    if not path.exists():
        raise FileNotFoundError(f"File pengecualian stemming tidak ditemukan: {path}")

    df = pd.read_csv(
        path,
        comment="#",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    missing = {"kata", "hasil"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} kehilangan kolom: {sorted(missing)}")

    exceptions: dict[str, str] = {}
    invalid: list[str] = []
    conflicts: list[str] = []
    for row in df.itertuples(index=False):
        key = str(row.kata).strip().lower()
        value = str(row.hasil).strip().lower()
        if not key and not value:
            continue
        if not key or not value or " " in key or " " in value:
            invalid.append(f"{key!r} -> {value!r}")
            continue
        if key in exceptions and exceptions[key] != value:
            conflicts.append(f"{key}: {exceptions[key]!r} vs {value!r}")
            continue
        exceptions[key] = value

    if invalid:
        raise ValueError(f"Pengecualian stemming tidak valid (harus satu token): {invalid[:5]}")
    if conflicts:
        raise ValueError(f"Pengecualian stemming punya pemetaan ganda: {conflicts[:5]}")
    return exceptions


def prepare_tokens(
    texts: Iterable[str], dictionary: dict[str, str], stopwords: set[str]
) -> list[tuple[list[str], list[str], list[str]]]:
    """Tokenization + normalisasi + stopword removal (tanpa stemming)."""
    prepared = []
    for text in texts:
        tokens = text.split()  # FR-21 (teks sudah bersih: tanpa tanda baca)
        normalized = normalize_tokens(tokens, dictionary)  # FR-22
        kept = [
            token
            for token in normalized
            if token in PROTECTED_WORDS or (token not in stopwords and len(token) > 1)
        ]  # FR-23
        prepared.append((tokens, normalized, kept))
    return prepared


def stem_token(token: str, cache: dict[str, str], exceptions: dict[str, str]) -> str:
    """Ambil hasil stemming dari cache; kata terlindungi/pengecualian apa adanya.

    Pengecualian diperiksa lebih dulu supaya Sastrawi tidak sempat melebur kata
    yang maknanya berbeda (mis. "perusahaan" -> "usaha").
    """
    if token in exceptions:
        return exceptions[token]
    if token in PROTECTED_WORDS:
        return token
    if len(token) > MAX_STEM_LENGTH:
        return token
    if token not in cache:
        cache[token] = _get_stemmer().stem(token)
    return cache[token]


def to_text(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def truncate(tokens: list[str] | str, limit: int = 18) -> str:
    if isinstance(tokens, str):
        tokens = tokens.split()
    shown = tokens[:limit]
    suffix = " ..." if len(tokens) > limit else ""
    return " | ".join(shown) + suffix


def main() -> int:
    setup_logging()
    lines: list[str] = []

    def say(message: str = "") -> None:
        lines.append(message)
        logger.info(message)

    try:
        df = pd.read_csv(
            CLEANED_CSV,
            encoding="utf-8",
            dtype={"video_id": str, "comment_id": str},
            keep_default_na=False,
        )
        dictionary = load_dictionary()
        stopwords = load_stopwords()
        exceptions = load_stem_exceptions()
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    if "text_cleaned" not in df.columns:
        logger.error("%s belum punya kolom text_cleaned. Jalankan dulu cleaning.", CLEANED_CSV)
        return 1

    df["text_cleaned"] = df["text_cleaned"].astype(str)
    df = df[df["text_cleaned"].str.strip().ne("")].copy()
    if df.empty:
        logger.error("Tidak ada teks untuk diproses setelah baris kosong dibuang.")
        return 1

    say("=" * 70)
    say("TEXT PREPROCESSING (PHASE 5)")
    say("=" * 70)
    say(f"Input            : {CLEANED_CSV.relative_to(PROJECT_ROOT)}")
    say(f"Jumlah komentar  : {len(df):,}".replace(",", "."))
    say(f"Kamus normalisasi: {len(dictionary)} entri")
    say(f"Stopword         : {len(stopwords)} kata")
    say(f"Kata terlindungi : {len(PROTECTED_WORDS)} istilah")
    say(f"Pengecualian stem: {len(exceptions)} kata")
    say()

    prepared = prepare_tokens(df["text_cleaned"], dictionary, stopwords)

    # Stemming hanya untuk kata unik, sekali saja.
    cache = load_stem_cache()
    cache_before = len(cache)
    unique_tokens = {
        token
        for _, _, kept in prepared
        for token in kept
        if token not in PROTECTED_WORDS
        and token not in exceptions
        and len(token) <= MAX_STEM_LENGTH
    }
    stem_missing_tokens(unique_tokens, cache)

    records = []
    total_normalized = 0
    total_dropped = 0
    total_dropped_after_stem = 0
    for tokens, normalized, kept in prepared:
        total_normalized += sum(1 for before, after in zip(tokens, normalized) if before != after)
        total_dropped += len(normalized) - len(kept)

        # Stemming lalu pembersihan akhir: stemming bisa memunculkan bentuk baku
        # yang ternyata stopword (mis. "menggunakan" -> "guna") atau kata tidak
        # baku dari batang kata (mis. "gaknya" -> "gak").
        stemmed = [stem_token(token, cache, exceptions) for token in kept]
        final = [
            token
            for token in normalize_tokens(stemmed, dictionary)
            if token in PROTECTED_WORDS or (token not in stopwords and len(token) > 1)
        ]
        total_dropped_after_stem += len(stemmed) - len(final)

        records.append(
            {
                "tokens": to_text(tokens),
                "tokens_normalized": to_text(normalized),
                "tokens_no_stopword": to_text(kept),
                "text_preprocessed": to_text(final),
            }
        )

    processed = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, processed], axis=1)

    say("-" * 70)
    say("1. STATISTIK PER TAHAP (FR-25)")
    say("-" * 70)
    stages = {
        "Token awal": df["tokens"].str.split(),
        "Setelah normalisasi": df["tokens_normalized"].str.split(),
        "Setelah stopword": df["tokens_no_stopword"].str.split(),
        "Setelah stemming": df["text_preprocessed"].str.split(),
    }
    say(f"  {'Tahap':<24}{'Total token':>14}{'Rata/komentar':>16}{'Kosakata':>12}")
    for name, series in stages.items():
        total = int(series.map(len).sum())
        avg = total / len(df)
        vocab = len({token for tokens in series for token in tokens})
        say(f"  {name:<24}{total:>14}{avg:>16.1f}{vocab:>12}")
    say()
    say(f"  Token yang diubah kamus normalisasi : {total_normalized}")
    say(f"  Token dibuang (stopword/1 huruf)   : {total_dropped}")
    say(f"  Token dibuang setelah stemming     : {total_dropped_after_stem}")
    say(f"  Cache stemming                     : {cache_before} kata lama + {len(cache) - cache_before} kata baru")
    say()

    final_series = df["text_preprocessed"].str.split()
    empty_final = int(final_series.map(len).eq(0).sum())
    say(f"  Komentar tanpa kata tersisa        : {empty_final} ({empty_final / len(df) * 100:.2f}%)")
    say("  Catatan: komentar seperti ini tetap disimpan; noise analysis (Phase 6) yang menandainya.")
    say()

    say("-" * 70)
    say("2. 10 KATA TERBANYAK SEBELUM vs SESUDAH STOPWORD REMOVAL")
    say("-" * 70)
    before = Counter(token for tokens in stages["Setelah normalisasi"] for token in tokens)
    after = Counter(token for tokens in final_series for token in tokens)
    say(f"  {'Sebelum':<28}{'Sesudah stemming':<22}")
    for (word_a, count_a), (word_b, count_b) in zip(before.most_common(10), after.most_common(10)):
        say(f"  {word_a + ' (' + str(count_a) + ')':<28}{word_b + ' (' + str(count_b) + ')':<22}")
    say()

    say("-" * 70)
    say("3. 20 KATA TERBANYAK HASIL AKHIR (bahan WordCloud Phase 7)")
    say("-" * 70)
    for index, (word, count) in enumerate(after.most_common(20), start=1):
        say(f"  {index:>2}. {word:<18}{count}")
    say()

    say("-" * 70)
    say("4. CONTOH PROSES PER TAHAP (FR-25)")
    say("-" * 70)
    examples = df[final_series.map(len) >= 5].head(3)
    for index, row in enumerate(examples.itertuples(index=False), start=1):
        say(f"Contoh {index}:")
        say(f"  Token          : {truncate(row.tokens)}")
        say(f"  Normalisasi    : {truncate(row.tokens_normalized)}")
        say(f"  Tanpa stopword : {truncate(row.tokens_no_stopword)}")
        say(f"  Stemming       : {truncate(row.text_preprocessed)}")
        say()

    # Pemeriksaan akhir.
    sisa_stopword = sum(
        1
        for tokens in final_series
        for token in tokens
        if token in stopwords and token not in PROTECTED_WORDS
    )
    sisa_non_huruf = int(df["text_preprocessed"].str.contains(r"[^a-z\s]").sum())
    sisa_kamus = sum(1 for tokens in final_series for token in tokens if token in dictionary)
    say("-" * 70)
    say("5. PEMERIKSAAN AKHIR")
    say("-" * 70)
    say(f"  Sisa stopword        : {sisa_stopword}")
    say(f"  Sisa karakter aneh   : {sisa_non_huruf}")
    say(f"  Sisa kata tidak baku : {sisa_kamus} (muncul dari batang kata, mis. 'gaknya' -> 'gak')")
    say(f"  Ukuran cache stemming: {len(cache)} kata unik")
    say("=" * 70)

    PREPROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREPROCESSED_CSV, index=False, encoding="utf-8")
    say(f"Output disimpan di : {PREPROCESSED_CSV.relative_to(PROJECT_ROOT)}")
    say(f"Kolom              : {', '.join(df.columns)}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Ringkasan disimpan di: %s", SUMMARY_PATH.relative_to(PROJECT_ROOT))

    if sisa_stopword:
        logger.error("Masih ada stopword di hasil akhir; periksa logika filter.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
