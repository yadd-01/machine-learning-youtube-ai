"""EDA (Exploratory Data Analysis) dataset mentah komentar YouTube.

Menjawab FR-11 s.d. FR-14:
    - jumlah komentar, data kosong, dan duplikat
    - panjang karakter dan jumlah kata per komentar
    - contoh komentar dan histogram distribusi panjang
    - indikasi noise (URL, mention, hashtag, emoji, simbol berlebihan)

Input : dataset/raw_comments.csv
Output: ringkasan di konsol + results/eda_summary.txt
        grafik results/eda_length_distribution.png

Jalankan dari root proyek:
    python analysis/eda.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend tanpa jendela, aman untuk server/Colab
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "dataset" / "raw_comments.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
CHART_PATH = RESULTS_DIR / "eda_length_distribution.png"
SUMMARY_PATH = RESULTS_DIR / "eda_summary.txt"

logger = logging.getLogger("eda")

# ---------------------------------------------------------------------------
# Pola deteksi noise
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NUMBER_RE = re.compile(r"\d+")
REPEATED_SYMBOL_RE = re.compile(r"([!?.,\-_~*])\1{2,}")
EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # bendera
    "\U0001F300-\U0001F5FF"  # simbol & piktogram
    "\U0001F600-\U0001F64F"  # emotikon
    "\U0001F680-\U0001F6FF"  # transportasi
    "\U0001F900-\U0001F9FF"  # simbol tambahan
    "\U0001FA70-\U0001FAFF"  # simbol terbaru
    "\u2600-\u27BF"          # simbol misc & dingbats
    "\u2B00-\u2BFF"
    "\uFE0F"                 # variation selector
    "]"
)


def setup_logging() -> None:
    """Siapkan log ke konsol dan kumpulkan baris untuk file ringkasan."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    formatter = logging.Formatter("%(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(console)


def load_data(path: Path) -> pd.DataFrame:
    """Muat dataset mentah dan pastikan kolom wajib tersedia."""
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
    required = {"video_id", "comment_id", "author", "text", "published_at", "like_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} kehilangan kolom: {sorted(missing)}")

    df["text"] = df["text"].astype(str)
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len().fillna(0).astype(int)
    return df


def describe_lengths(series: pd.Series) -> dict[str, float]:
    """Statistik deskriptif panjang (karakter atau kata)."""
    return {
        "min": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "q3": float(series.quantile(0.75)),
        "max": float(series.max()),
        "std": float(series.std()),
    }


def format_stats(name: str, stats: dict[str, float], unit: str) -> str:
    return (
        f"  {name:<14}: min {stats['min']:.0f} | Q1 {stats['q1']:.0f} | median {stats['median']:.0f} | "
        f"mean {stats['mean']:.1f} | Q3 {stats['q3']:.0f} | max {stats['max']:.0f} | std {stats['std']:.1f} {unit}"
    )


def noise_table(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung indikasi noise per komentar (FR-14)."""
    text = df["text"]
    checks = {
        "Mengandung URL": text.str.contains(URL_RE),
        "Mengandung mention (@)": text.str.contains(MENTION_RE),
        "Mengandung hashtag (#)": text.str.contains(HASHTAG_RE),
        "Mengandung emoji": text.str.contains(EMOJI_RE),
        "Mengandung angka": text.str.contains(NUMBER_RE),
        "Simbol berulang (3+)": text.map(REPEATED_SYMBOL_RE.search).notna(),
        "Teks kosong": text.str.strip().eq(""),
        "Sangat pendek (< 3 kata)": df["word_count"] < 3,
        "Teks duplikat": text.duplicated() & text.str.strip().ne(""),
    }

    total = len(df)
    rows = [
        {
            "indikasi": label,
            "jumlah": int(mask.sum()),
            "persen": mask.mean() * 100 if total else 0.0,
        }
        for label, mask in checks.items()
    ]
    return pd.DataFrame(rows)


def plot_distributions(df: pd.DataFrame, path: Path) -> None:
    """Histogram panjang karakter dan jumlah kata (FR-13)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    specs = [
        (df["char_count"], "Panjang komentar (karakter)", "#2a6fdb", axes[0]),
        (df["word_count"], "Jumlah kata per komentar", "#e07b39", axes[1]),
    ]

    for series, title, color, ax in specs:
        upper = series.quantile(0.99)
        clipped = series.clip(upper=upper)
        bins = min(50, max(15, int(clipped.nunique())))
        ax.hist(clipped, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(series.median(), color="#333333", linestyle="--", linewidth=1.2,
                   label=f"median = {series.median():.0f}")
        ax.axvline(series.mean(), color="#b3261e", linestyle=":", linewidth=1.2,
                   label=f"mean = {series.mean():.1f}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{title} (dipotong pada persentil 99 = {upper:.0f})")
        ax.set_ylabel("Jumlah komentar")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Distribusi Panjang Komentar YouTube tentang AI (n = {len(df):,} komentar mentah)".replace(",", "."),
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    setup_logging()
    lines: list[str] = []

    def say(message: str = "") -> None:
        lines.append(message)
        logger.info(message)

    try:
        df = load_data(RAW_CSV)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    total = len(df)
    say("=" * 70)
    say("EDA DATASET MENTAH KOMENTAR YOUTUBE TENTANG AI")
    say("=" * 70)
    say(f"Sumber file            : {RAW_CSV.relative_to(PROJECT_ROOT)}")
    say(f"Jumlah baris           : {total:,}".replace(",", "."))
    say(f"comment_id unik        : {df['comment_id'].nunique():,}".replace(",", "."))
    say(f"Komentar duplikat      : {int(df['comment_id'].duplicated().sum())} (berdasarkan comment_id)")
    say(f"Video unik             : {df['video_id'].nunique()}")
    say(
        "Rentang tanggal        : "
        f"{df['published_at'].min()[:10]} s.d. {df['published_at'].max()[:10]}"
    )
    say()

    say("-" * 70)
    say("1. KELENGKAPAN DATA")
    say("-" * 70)
    for column in ["video_id", "comment_id", "author", "text", "published_at", "like_count"]:
        empty = int((df[column].astype(str).str.strip() == "").sum())
        say(f"  {column:<14}: {empty} kosong")
    say(f"  {'text duplikat':<14}: {int(df['text'].duplicated().sum())} (teks identik, beda comment_id)")
    say()

    say("-" * 70)
    say("2. DISTRIBUSI PANJANG KOMENTAR")
    say("-" * 70)
    char_stats = describe_lengths(df["char_count"])
    word_stats = describe_lengths(df["word_count"])
    say(format_stats("Karakter", char_stats, "karakter"))
    say(format_stats("Kata", word_stats, "kata"))
    say(f"  Komentar < 10 karakter : {int((df['char_count'] < 10).sum())}")
    say(f"  Komentar > 500 karakter: {int((df['char_count'] > 500).sum())}")
    say()

    say("-" * 70)
    say("3. DISTRIBUSI PER VIDEO")
    say("-" * 70)
    per_video = (
        df.groupby("video_id")
        .agg(
            komentar=("comment_id", "size"),
            rata_karakter=("char_count", "mean"),
            rata_kata=("word_count", "mean"),
            total_like=("like_count", "sum"),
        )
        .sort_values("komentar", ascending=False)
    )
    for video_id, row in per_video.iterrows():
        say(
            f"  {video_id} | {int(row['komentar']):>4} komentar | "
            f"rata {row['rata_karakter']:.0f} karakter / {row['rata_kata']:.0f} kata | "
            f"{int(row['total_like'])} like"
        )
    say()

    say("-" * 70)
    say("4. INDIKASI NOISE (FR-14)")
    say("-" * 70)
    noise = noise_table(df)
    for _, row in noise.iterrows():
        say(f"  {row['indikasi']:<24}: {int(row['jumlah']):>4} komentar ({row['persen']:.1f}%)")
    say("  Catatan: kata informal TIDAK dihitung noise; penanganannya di tahap normalisasi.")
    say()

    say("-" * 70)
    say("5. CONTOH KOMENTAR")
    say("-" * 70)
    say("  Terpanjang:")
    for text in df.nlargest(2, "char_count")["text"]:
        say(f"    - {text[:160].replace(chr(10), ' ')}...")
    say("  Terpendek (tidak kosong):")
    for text in df[df["char_count"] > 0].nsmallest(3, "char_count")["text"]:
        say(f"    - {text[:160].replace(chr(10), ' ')}")
    say("  Acak:")
    for text in df.sample(3, random_state=42)["text"]:
        say(f"    - {text[:160].replace(chr(10), ' ')}")
    say()

    plot_distributions(df, CHART_PATH)
    say(f"Grafik distribusi disimpan di: {CHART_PATH.relative_to(PROJECT_ROOT)}")
    say("=" * 70)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Ringkasan teks disimpan di: %s", SUMMARY_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
