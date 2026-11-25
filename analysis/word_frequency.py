"""Perhitungan frekuensi kata per tahap pipeline (Phase 7).

Menjawab FR-31 s.d. FR-34 (bagian data): menyediakan tabel frekuensi untuk
WordCloud raw, cleaned, preprocessed, dan noise.

Input : dataset/raw_comments.csv
        dataset/cleaned_comments.csv
        dataset/preprocessed_comments.csv
        dataset/noise_comments.csv
Output: results/word_frequencies.csv   (stage, word, count)
        results/top_20_words.csv       (top 20 tahap preprocessed)
        ringkasan di konsol + results/word_frequency_summary.txt

Jalankan dari root proyek:
    python analysis/word_frequency.py
"""

from __future__ import annotations

import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "dataset" / "raw_comments.csv"
CLEANED_CSV = PROJECT_ROOT / "dataset" / "cleaned_comments.csv"
PREPROCESSED_CSV = PROJECT_ROOT / "dataset" / "preprocessed_comments.csv"
NOISE_CSV = PROJECT_ROOT / "dataset" / "noise_comments.csv"
FREQ_CSV = PROJECT_ROOT / "results" / "word_frequencies.csv"
TOP20_CSV = PROJECT_ROOT / "results" / "top_20_words.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "word_frequency_summary.txt"

LETTER_WORD_RE = re.compile(r"[a-z]{2,}", re.IGNORECASE)
MIN_WORD_LENGTH = 2

logger = logging.getLogger("word_frequency")


def setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(console)


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ditemukan. Jalankan tahap sebelumnya dulu.")
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} kehilangan kolom: {missing}")
    return df


def tokenize_raw(texts: pd.Series) -> list[str]:
    """Ambil kata dari teks mentah (URL/emoji/angka tidak ikut sebagai kata)."""
    tokens: list[str] = []
    for text in texts:
        tokens.extend(LETTER_WORD_RE.findall(str(text).lower()))
    return tokens


def tokenize_clean(texts: pd.Series) -> list[str]:
    return [
        token
        for text in texts
        for token in str(text).split()
        if len(token) >= MIN_WORD_LENGTH
    ]


def main() -> int:
    setup_logging()
    lines: list[str] = []

    def say(message: str = "") -> None:
        lines.append(message)
        logger.info(message)

    try:
        raw = read_csv(RAW_CSV, ["text"])
        cleaned = read_csv(CLEANED_CSV, ["text_cleaned"])
        preprocessed = read_csv(PREPROCESSED_CSV, ["text_preprocessed"])
        noise = read_csv(NOISE_CSV, ["text_cleaned", "text_preprocessed"])
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    tokens_by_stage: dict[str, list[str]] = {
        "raw": tokenize_raw(raw["text"]),
        "cleaned": tokenize_clean(cleaned["text_cleaned"]),
        "preprocessed": tokenize_clean(preprocessed["text_preprocessed"]),
        # Baris noise yang gugur saat cleaning tidak punya text_preprocessed;
        # pakai text_cleaned sebagai fallback.
        "noise": [
            token
            for text in noise["text_preprocessed"].where(
                noise["text_preprocessed"].str.strip().ne(""), noise["text_cleaned"]
            )
            for token in str(text).split()
            if len(token) >= MIN_WORD_LENGTH
        ],
    }

    counts = {stage: Counter(tokens) for stage, tokens in tokens_by_stage.items()}

    say("=" * 70)
    say("FREKUENSI KATA PER TAHAP (PHASE 7)")
    say("=" * 70)
    say(f"  {'Tahap':<16}{'Komentar':>10}{'Total kata':>12}{'Kata unik':>12}")
    rows = len(raw), len(cleaned), len(preprocessed), len(noise)
    for (stage, tokens), n_rows in zip(tokens_by_stage.items(), rows):
        say(f"  {stage:<16}{n_rows:>10}{len(tokens):>12}{len(counts[stage]):>12}")
    say()

    for stage, counter in counts.items():
        say("-" * 70)
        say(f"10 KATA TERBANYAK - {stage.upper()}")
        say("-" * 70)
        for index, (word, count) in enumerate(counter.most_common(10), start=1):
            say(f"  {index:>2}. {word:<18}{count}")
        say()

    table = pd.DataFrame(
        [
            {"stage": stage, "word": word, "count": count}
            for stage, counter in counts.items()
            for word, count in counter.most_common()
        ]
    )
    FREQ_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FREQ_CSV, index=False, encoding="utf-8")

    top20 = table[table["stage"] == "preprocessed"].head(20).copy()
    top20.insert(0, "rank", range(1, len(top20) + 1))
    top20["persen"] = (top20["count"] / len(tokens_by_stage["preprocessed"]) * 100).round(2)
    top20.to_csv(TOP20_CSV, index=False, encoding="utf-8")

    say(f"Tabel frekuensi disimpan di : {FREQ_CSV.relative_to(PROJECT_ROOT)} ({len(table)} baris)")
    say(f"Top 20 disimpan di          : {TOP20_CSV.relative_to(PROJECT_ROOT)}")
    say("=" * 70)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Ringkasan disimpan di: %s", SUMMARY_PATH.relative_to(PROJECT_ROOT))

    if table.empty:
        logger.error("Tabel frekuensi kosong.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
