"""Noise analysis komentar YouTube (Phase 6).

Menjawab FR-27 s.d. FR-30:
    - Mendeteksi spam, komentar kosong, simbol berlebihan, URL tidak relevan,
      karakter/kata berulang tanpa informasi.
    - Membandingkan statistik sebelum dan sesudah noise removal.
    - TIDAK menganggap kata informal sebagai noise (FR-29).

Pendekatan: label noise bersifat struktural (bukan berdasarkan gaya bahasa).
Analisis dijalankan pada SEMUA komentar mentah, termasuk yang gugur saat
cleaning (kosong/duplikat), agar jumlahnya transparan.

Input : dataset/raw_comments.csv
        dataset/cleaned_comments.csv (untuk validasi silang)
        dataset/preprocessed_comments.csv (untuk kata informatif)
Output: dataset/noise_comments.csv
        ringkasan di konsol + results/noise_summary.txt

Jalankan dari root proyek:
    python analysis/noise_analysis.py
"""

from __future__ import annotations

import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from cleaning import (  # noqa: E402  (butuh sys.path di atas)
    EMOJI_RE,
    HASHTAG_RE,
    MENTION_RE,
    URL_RE,
    clean_text,
)
from protected_words import PROTECTED_WORDS  # noqa: E402

RAW_CSV = PROJECT_ROOT / "dataset" / "raw_comments.csv"
CLEANED_CSV = PROJECT_ROOT / "dataset" / "cleaned_comments.csv"
PREPROCESSED_CSV = PROJECT_ROOT / "dataset" / "preprocessed_comments.csv"
NOISE_CSV = PROJECT_ROOT / "dataset" / "noise_comments.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "noise_summary.txt"

# ---------------------------------------------------------------------------
# Aturan deteksi noise
# ---------------------------------------------------------------------------
REPEATED_CHAR_RE = re.compile(r"([A-Za-z!?.,~\-_*])\1{4,}")
SYMBOL_RE = re.compile(r"[!-/:-@\[-`{-~]")  # tanda baca/simbol ASCII
PHONE_RE = re.compile(r"(?:\+?62|0)8\d{7,}")

SPAM_PATTERNS = {
    "ajakan ke channel lain": re.compile(
        r"(\b(subrek|back\s?sub)\b.*\b(channel|chanel|youtube)\b)"
        r"|(\b(mampir|cek|kunjungi)\b.*\b(channel|chanel)\b)",
        re.IGNORECASE,
    ),
    "jualan eksplisit": re.compile(
        r"(open\s*(order|po)|hubungi\s*(wa|whatsapp)|wa\.me|t\.me|bit\.ly|link di (bio|deskripsi))",
        re.IGNORECASE,
    ),
    "judi/pinjol": re.compile(
        r"\b(slot gacor|judi online|pinjol|casino)\b",
        re.IGNORECASE,
    ),
}

# Label yang dihitung sebagai noise (bukan sekadar indikator).
NOISE_LABELS = [
    "kosong setelah cleaning",
    "duplikat teks",
    "tanpa kata informatif",
    "terlalu pendek (<= 1 kata informatif)",
    "spam/promosi",
    "karakter berulang",
    "kata berulang",
    "simbol berlebihan",
]

logger = logging.getLogger("noise")


def setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(console)


def load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan tahap sebelumnya lebih dulu."
        )
    df = pd.read_csv(
        path,
        encoding="utf-8",
        dtype={"video_id": str, "comment_id": str},
        keep_default_na=False,
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} kehilangan kolom: {sorted(missing)}")
    return df


def symbol_ratio(text: str) -> float:
    """Proporsi simbol/tanda baca/emoji terhadap panjang teks."""
    if not text:
        return 0.0
    symbols = len(SYMBOL_RE.findall(text)) + len(EMOJI_RE.findall(text))
    return symbols / len(text)


def repeated_word(tokens: list[str]) -> bool:
    # Kata topik (ai, robot, ...) sengaja tidak dihitung: komentar tentang AI
    # wajar menyebut "ai" berkali-kali.
    content = [token for token in tokens if token not in PROTECTED_WORDS and len(token) > 2]
    if len(content) >= 4:
        top_count = max(Counter(content).values(), default=0)
        if top_count >= 4 and top_count / len(content) >= 0.4:
            return True
    if len(tokens) >= 6:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.35:
            return True
    return False


def detect(
    text: str,
    text_cleaned: str,
    informative: list[str],
    informative_known: bool,
    duplicate: bool,
) -> tuple[list[str], list[str]]:
    """Kembalikan (label noise, indikator non-noise)."""
    labels: list[str] = []
    indicators: list[str] = []

    if not text_cleaned:
        labels.append("kosong setelah cleaning")
    if duplicate:
        labels.append("duplikat teks")
    if informative_known and text_cleaned and not informative:
        labels.append("tanpa kata informatif")
    if informative_known and len(informative) == 1:
        labels.append("terlalu pendek (<= 1 kata informatif)")

    if REPEATED_CHAR_RE.search(text):
        labels.append("karakter berulang")
    if len(text) >= 5 and symbol_ratio(text) >= 0.35:
        labels.append("simbol berlebihan")
    if repeated_word(text_cleaned.split()):
        labels.append("kata berulang")

    for reason, pattern in SPAM_PATTERNS.items():
        if pattern.search(text):
            labels.append("spam/promosi")
            indicators.append(f"pola {reason}")
            break
    if PHONE_RE.search(text):
        if "spam/promosi" not in labels:
            labels.append("spam/promosi")
        indicators.append("nomor telepon")

    if URL_RE.search(text):
        indicators.append("mengandung URL")
    if MENTION_RE.search(text):
        indicators.append("mengandung mention")
    if HASHTAG_RE.search(text):
        indicators.append("mengandung hashtag")
    if EMOJI_RE.search(text):
        indicators.append("mengandung emoji")
    if re.search(r"\d", text):
        indicators.append("mengandung angka")
    letters = [char for char in text if char.isalpha()]
    if len(letters) >= 10 and sum(char.isupper() for char in letters) / len(letters) >= 0.7:
        indicators.append("mayoritas kapital")

    return labels, indicators


def main() -> int:
    setup_logging()
    lines: list[str] = []

    def say(message: str = "") -> None:
        lines.append(message)
        logger.info(message)

    try:
        raw = load_csv(RAW_CSV, {"video_id", "comment_id", "text"})
        cleaned = load_csv(CLEANED_CSV, {"comment_id", "text_cleaned"})
        preprocessed = load_csv(PREPROCESSED_CSV, {"comment_id", "text_preprocessed"})
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    cleaned_map = dict(zip(cleaned["comment_id"], cleaned["text_cleaned"]))
    preprocessed_map = dict(zip(preprocessed["comment_id"], preprocessed["text_preprocessed"]))

    # Hitung ulang cleaning untuk SEMUA komentar mentah agar baris yang gugur
    # saat cleaning tetap terhitung dalam analisis noise.
    raw["text"] = raw["text"].astype(str)
    raw["text_cleaned"] = raw["text"].map(clean_text)
    raw["text_preprocessed"] = raw["comment_id"].map(preprocessed_map).fillna("")

    mismatch = 0
    for comment_id, value in zip(raw["comment_id"], raw["text_cleaned"]):
        if comment_id in cleaned_map and cleaned_map[comment_id] != value:
            mismatch += 1
    if mismatch:
        logger.warning("%d teks tidak sama dengan cleaned_comments.csv (cek aturan cleaning).", mismatch)

    duplicate_mask = raw["text_cleaned"].duplicated(keep="first") & raw["text_cleaned"].ne("")

    records = []
    for row, duplicate in zip(raw.itertuples(index=False), duplicate_mask):
        informative = str(row.text_preprocessed).split()
        informative_known = row.comment_id in preprocessed_map and bool(row.text_cleaned)
        labels, indicators = detect(
            row.text, row.text_cleaned, informative, informative_known, bool(duplicate)
        )
        records.append(
            {
                "comment_id": row.comment_id,
                "video_id": row.video_id,
                "text": row.text,
                "text_cleaned": row.text_cleaned,
                "text_preprocessed": row.text_preprocessed,
                "noise_type": "; ".join(labels),
                "indikator": "; ".join(indicators),
                "jumlah_label": len(labels),
            }
        )

    labeled = pd.DataFrame(records)
    labeled["is_noise"] = labeled["jumlah_label"] > 0

    total = len(labeled)
    noise_rows = labeled[labeled["is_noise"]].copy()
    clean_rows = labeled[~labeled["is_noise"]]

    say("=" * 70)
    say("NOISE ANALYSIS (PHASE 6)")
    say("=" * 70)
    say(f"Input            : {RAW_CSV.relative_to(PROJECT_ROOT)}")
    say(f"Total komentar   : {total:,}".replace(",", "."))
    say(f"Komentar noise   : {len(noise_rows):,} ({len(noise_rows) / total * 100:.1f}%)".replace(",", "."))
    say(f"Komentar bersih  : {len(clean_rows):,} ({len(clean_rows) / total * 100:.1f}%)".replace(",", "."))
    say()
    say("Catatan: label noise bersifat struktural. Kata informal seperti 'gak',")
    say("'bgt', atau 'gue' TIDAK dianggap noise (FR-29).")
    say()

    say("-" * 70)
    say("1. JUMLAH PER JENIS NOISE (FR-27)")
    say("-" * 70)
    say(f"  {'Jenis noise':<40}{'Jumlah':>8}{'% total':>10}")
    for label in NOISE_LABELS:
        count = int(labeled["noise_type"].str.contains(re.escape(label)).sum())
        say(f"  {label:<40}{count:>8}{count / total * 100:>9.1f}%")
    if not labeled["noise_type"].str.contains("spam/promosi").any():
        say("  Tidak ditemukan spam promosi pada dataset ini (indikasi data relatif bersih).")
    say("  (Satu komentar bisa punya lebih dari satu label, jadi total > jumlah noise unik.)")
    say()

    say("-" * 70)
    say("2. SEBELUM vs SESUDAH NOISE REMOVAL (FR-28)")
    say("-" * 70)
    raw_words_before = labeled["text"].str.split().str.len()
    raw_words_after = clean_rows["text"].str.split().str.len()
    tokens_before = [token for text in labeled["text_preprocessed"] for token in str(text).split()]
    tokens_after = [token for text in clean_rows["text_preprocessed"] for token in str(text).split()]

    say(f"  {'Metrik':<32}{'Sebelum':>12}{'Sesudah':>12}")
    say(f"  {'Jumlah komentar':<32}{total:>12}{len(clean_rows):>12}")
    say(f"  {'Rata-rata kata mentah':<32}{raw_words_before.mean():>12.1f}{raw_words_after.mean():>12.1f}")
    say(f"  {'Total kata informatif':<32}{len(tokens_before):>12}{len(tokens_after):>12}")
    say(f"  {'Kosakata informatif':<32}{len(set(tokens_before)):>12}{len(set(tokens_after)):>12}")
    say(f"  {'Rata-rata kata informatif':<32}{len(tokens_before) / total:>12.1f}{len(tokens_after) / max(len(clean_rows), 1):>12.1f}")
    say()

    say("-" * 70)
    say("3. CONTOH NOISE PER JENIS")
    say("-" * 70)
    for label in NOISE_LABELS:
        subset = noise_rows[noise_rows["noise_type"].str.contains(re.escape(label))]
        say(f"  [{label}] ({len(subset)} komentar)")
        for text in subset["text"].head(2):
            say(f"    - {' '.join(str(text).split())[:120]}")
        if subset.empty:
            say("    (tidak ada)")
    say()

    say("-" * 70)
    say("4. INDIKATOR TAMBAHAN (tidak otomatis dianggap noise)")
    say("-" * 70)
    for indicator in ["mengandung URL", "mengandung mention", "mengandung hashtag", "mengandung emoji", "mengandung angka", "mayoritas kapital"]:
        count = int(labeled["indikator"].str.contains(re.escape(indicator)).sum())
        say(f"  {indicator:<24}{count:>8} ({count / total * 100:.1f}%)")
    say("=" * 70)

    NOISE_CSV.parent.mkdir(parents=True, exist_ok=True)
    noise_rows[
        ["comment_id", "video_id", "text", "text_cleaned", "text_preprocessed", "noise_type", "jumlah_label"]
    ].to_csv(NOISE_CSV, index=False, encoding="utf-8")

    say(f"Dataset noise disimpan di: {NOISE_CSV.relative_to(PROJECT_ROOT)} ({len(noise_rows)} baris)")
    say("Dataset utama TIDAK diubah; label ini dipakai untuk analisis dan WordCloud noise.")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Ringkasan disimpan di: %s", SUMMARY_PATH.relative_to(PROJECT_ROOT))

    # Validasi.
    if noise_rows["comment_id"].duplicated().any():
        logger.error("comment_id ganda di dataset noise.")
        return 1
    if len(noise_rows) + len(clean_rows) != total:
        logger.error("Jumlah noise + bersih tidak sama dengan total.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
