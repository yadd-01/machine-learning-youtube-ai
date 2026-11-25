"""Cleaning teks komentar mentah (Phase 4).

Menjawab FR-15 s.d. FR-20. Urutan pemrosesan mengikuti ARCHITECTURE.md 4.3:
    case folding -> hapus URL -> hapus mention -> hapus simbol hashtag
    -> hapus emoji -> hapus angka -> hapus tanda baca/simbol
    -> rapikan spasi -> buang komentar kosong -> buang duplikat teks

Catatan penting: kata tidak baku (gak, bgt, nggak) TIDAK dihapus di sini.
Penanganannya ada di tahap normalisasi (Phase 5) memakai kamus manual.

Input : dataset/raw_comments.csv
Output: dataset/cleaned_comments.csv (kolom raw + text_cleaned)
        ringkasan di konsol + results/cleaning_summary.txt

Jalankan dari root proyek:
    python preprocessing/cleaning.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "dataset" / "raw_comments.csv"
CLEANED_CSV = PROJECT_ROOT / "dataset" / "cleaned_comments.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "cleaning_summary.txt"

logger = logging.getLogger("cleaning")

# ---------------------------------------------------------------------------
# Pola regex
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u27BF"
    "\u2B00-\u2BFF"
    "\uFE00-\uFE0F"   # variation selector
    "\u200D"          # zero width joiner
    "\U0001F3FB-\U0001F3FF"  # skin tone
    "]"
)
NUMBER_RE = re.compile(r"\d+")
NON_LETTER_RE = re.compile(r"[^a-z]+")  # setelah case folding: hanya huruf latin kecil + spasi
WHITESPACE_RE = re.compile(r"[\s\u00a0\u200b\u200c\u200d\ufeff]+")


# ---------------------------------------------------------------------------
# Langkah cleaning (satu fungsi = satu tanggung jawab)
# ---------------------------------------------------------------------------
def case_fold(text: str) -> str:
    """FR-15: samakan ke huruf kecil."""
    return text.lower()


def remove_urls(text: str) -> str:
    """FR-16: buang URL (http/https/www)."""
    return URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    """FR-16: buang mention @username."""
    return MENTION_RE.sub(" ", text)


def remove_hashtag_symbol(text: str) -> str:
    """FR-16: buang simbol '#', katanya dipertahankan."""
    return text.replace("#", " ")


def remove_emoji(text: str) -> str:
    """FR-17: buang emoji dan simbol pictograph."""
    return EMOJI_RE.sub(" ", text)


def remove_numbers(text: str) -> str:
    """FR-17: buang angka yang tidak relevan untuk analisis kata."""
    return NUMBER_RE.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """FR-17: ganti tanda baca/simbol/karakter non-latin dengan spasi.

    Diganti spasi (bukan dihapus) agar kata tidak menempel:
    'keren!!banget' -> 'keren banget'.
    """
    return NON_LETTER_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """FR-18: rapikan spasi berlebih, newline, dan karakter tak terlihat."""
    return WHITESPACE_RE.sub(" ", text).strip()


# Urutan langkah sesuai arsitektur.
STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("Case folding", case_fold),
    ("Hapus URL", remove_urls),
    ("Hapus mention", remove_mentions),
    ("Hapus simbol hashtag", remove_hashtag_symbol),
    ("Hapus emoji", remove_emoji),
    ("Hapus angka", remove_numbers),
    ("Hapus tanda baca & simbol", remove_punctuation),
    ("Rapikan spasi", normalize_whitespace),
]


def clean_text(text: str) -> str:
    """Terapkan seluruh langkah cleaning ke satu teks.

    Dipakai juga oleh analysis/noise_analysis.py agar aturan cleaning konsisten.
    """
    result = str(text)
    for _, func in STEPS:
        result = func(result)
    return result


def setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(console)


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan dulu: python scraping/scraping.py"
        )
    df = pd.read_csv(
        path,
        encoding="utf-8",
        dtype={"video_id": str, "comment_id": str},
        keep_default_na=False,
    )
    if "text" not in df.columns:
        raise ValueError(f"{path} tidak punya kolom 'text'.")
    df["text"] = df["text"].astype(str)
    return df


def apply_steps(texts: pd.Series) -> tuple[pd.Series, list[dict[str, int | str]]]:
    """Terapkan seluruh langkah cleaning sambil mencatat berapa teks yang berubah."""
    current = texts
    stats: list[dict[str, int | str]] = []
    for name, func in STEPS:
        result = current.map(func)
        stats.append(
            {
                "langkah": name,
                "berubah": int((result != current).sum()),
                "kosong": int(result.str.strip().eq("").sum()),
            }
        )
        current = result
    return current, stats


def show_before_after(df: pd.DataFrame, n: int = 8) -> list[str]:
    """FR-19: contoh teks sebelum vs sesudah cleaning."""
    lines = ["CONTOH SEBELUM vs SESUDAH CLEANING", "-" * 70]
    sample = df.sample(min(n, len(df)), random_state=42)
    for index, row in enumerate(sample.itertuples(index=False), start=1):
        before = " ".join(str(row.text).split())
        after = " ".join(str(row.text_cleaned).split())
        lines.append(f"{index}. SEBELUM : {before[:150]}")
        lines.append(f"   SESUDAH : {after[:150] or '(kosong)'}")
        lines.append("")
    return lines


def main() -> int:
    setup_logging()
    lines: list[str] = []

    def say(message: str = "") -> None:
        lines.append(message)
        logger.info(message)

    try:
        df = load_raw(RAW_CSV)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    rows_raw = len(df)
    chars_before = df["text"].str.len()
    words_before = df["text"].str.split().str.len()

    say("=" * 70)
    say("CLEANING DATA KOMENTAR (PHASE 4)")
    say("=" * 70)
    say(f"Input  : {RAW_CSV.relative_to(PROJECT_ROOT)} ({rows_raw:,} baris)".replace(",", "."))
    say()

    cleaned, step_stats = apply_steps(df["text"])
    df["text_cleaned"] = cleaned

    say("-" * 70)
    say("1. RINGKASAN PER LANGKAH (FR-16, FR-17, FR-18)")
    say("-" * 70)
    say(f"  {'Langkah':<28}{'Teks berubah':>14}{'Jadi kosong':>14}")
    for stat in step_stats:
        say(f"  {str(stat['langkah']):<28}{int(stat['berubah']):>14}{int(stat['kosong']):>14}")
    say()

    # FR-18: buang komentar kosong.
    empty_mask = df["text_cleaned"].str.strip().eq("")
    rows_empty = int(empty_mask.sum())
    df = df[~empty_mask].copy()

    # FR-18: buang duplikat teks (setelah cleaning), pertahankan kemunculan pertama.
    rows_before_dedup = len(df)
    df = df.drop_duplicates(subset="text_cleaned", keep="first").copy()
    rows_duplicate = rows_before_dedup - len(df)

    say("-" * 70)
    say("2. PEMBERSIHAN BARIS (FR-18)")
    say("-" * 70)
    say(f"  Baris awal                         : {rows_raw}")
    say(f"  Dibuang karena kosong              : {rows_empty}")
    say(f"  Dibuang karena teks duplikat       : {rows_duplicate}")
    say(f"  Baris akhir                        : {len(df)}")
    say(f"  Persentase data dipertahankan      : {len(df) / rows_raw * 100:.1f}%")
    say()

    chars_after = df["text_cleaned"].str.len()
    words_after = df["text_cleaned"].str.split().str.len()
    say("-" * 70)
    say("3. PERBANDINGAN SEBELUM vs SESUDAH (FR-19)")
    say("-" * 70)
    say(f"  {'Metrik':<24}{'Sebelum':>12}{'Sesudah':>12}")
    say(f"  {'Jumlah komentar':<24}{rows_raw:>12}{len(df):>12}")
    say(f"  {'Rata-rata karakter':<24}{chars_before.mean():>12.1f}{chars_after.mean():>12.1f}")
    say(f"  {'Rata-rata kata':<24}{words_before.mean():>12.1f}{words_after.mean():>12.1f}")
    say(f"  {'Komentar tanpa huruf':<24}{0:>12}{int(words_after.eq(0).sum()):>12}")
    say()

    # Pemeriksaan akhir: pastikan tidak ada sisa noise yang seharusnya hilang.
    sisa_url = int(df["text_cleaned"].str.contains(URL_RE).sum())
    sisa_emoji = int(df["text_cleaned"].str.contains(EMOJI_RE).sum())
    sisa_angka = int(df["text_cleaned"].str.contains(r"\d").sum())
    sisa_simbol = int(df["text_cleaned"].str.contains(r"[^a-z\s]").sum())
    say("-" * 70)
    say("4. PEMERIKSAAN AKHIR")
    say("-" * 70)
    say(f"  Sisa URL        : {sisa_url}")
    say(f"  Sisa emoji      : {sisa_emoji}")
    say(f"  Sisa angka      : {sisa_angka}")
    say(f"  Sisa simbol     : {sisa_simbol}")
    say()

    say("-" * 70)
    for line in show_before_after(df):
        say(line)
    say("-" * 70)

    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False, encoding="utf-8")
    say(f"Output disimpan di : {CLEANED_CSV.relative_to(PROJECT_ROOT)}")
    say(f"Kolom              : {', '.join(df.columns)}")
    say("=" * 70)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Ringkasan disimpan di: %s", SUMMARY_PATH.relative_to(PROJECT_ROOT))

    if sisa_url or sisa_emoji or sisa_angka or sisa_simbol:
        logger.error("Masih ada noise tersisa; periksa pola cleaning.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
