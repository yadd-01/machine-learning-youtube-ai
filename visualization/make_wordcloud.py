"""Visualisasi WordCloud dan bar chart Top 20 kata (Phase 7).

Menjawab FR-31 s.d. FR-34:
    - WordCloud raw, cleaned, preprocessed (wajib)
    - WordCloud noise (opsional)
    - Bar chart Top 20 kata (wajib)
    - Semua judul/ukuran jelas dan dibuat dari frekuensi dataset nyata

Input : results/word_frequencies.csv
        results/top_20_words.csv
Output: results/wordcloud_raw.png
        results/wordcloud_cleaned.png
        results/wordcloud_preprocessed.png
        results/wordcloud_noise.png
        results/top_20_words.png
        ringkasan di konsol + results/wordcloud_summary.txt

Nama file sengaja `make_wordcloud.py` (bukan `wordcloud.py`) agar tidak
menimpa/membayangi library `wordcloud` saat impor.

Jalankan dari root proyek:
    python visualization/make_wordcloud.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend tanpa jendela
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREQ_CSV = PROJECT_ROOT / "results" / "word_frequencies.csv"
TOP20_CSV = PROJECT_ROOT / "results" / "top_20_words.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "wordcloud_summary.txt"

# Tampilan tiap WordCloud: (judul, keterangan, colormap, warna latar).
CLOUD_STYLE = {
    "raw": (
        "WordCloud Komentar Mentah",
        "Sebelum cleaning: masih ada stopword, singkatan, dan kata tidak baku",
        "viridis",
        "#0b1220",
    ),
    "cleaned": (
        "WordCloud Setelah Cleaning",
        "Setelah case folding, hapus URL/emoji/angka/simbol",
        "Dark2",
        "#ffffff",
    ),
    "preprocessed": (
        "WordCloud Setelah Preprocessing",
        "Setelah normalisasi, stopword removal, dan stemming",
        "tab10",
        "#ffffff",
    ),
    "noise": (
        "WordCloud Komentar Noise",
        "Komentar berlabel noise (spam, kosong, duplikat, sangat pendek, simbol berlebihan)",
        "autumn",
        "#1a1a1a",
    ),
}

# Batas kata yang digambar; frekuensi 1 tetap disimpan di CSV tapi tidak
# memenuhi WordCloud (terlalu banyak kata sekali muncul).
MIN_CLOUD_COUNT = 2
MAX_WORDS = 200
CLOUD_WIDTH = 2400
CLOUD_HEIGHT = 1500
CLOUD_DPI = 150

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]

logger = logging.getLogger("make_wordcloud")


def setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(console)


def find_font() -> str | None:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def load_frequencies(path: Path = FREQ_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan dulu: python analysis/word_frequency.py"
        )
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    if not {"stage", "word", "count"} <= set(df.columns):
        raise ValueError(f"{path} harus punya kolom stage, word, count")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["word"] = df["word"].astype(str)
    return df[df["word"].str.strip().ne("") & df["count"].gt(0)]


def frequencies_for(df: pd.DataFrame, stage: str, min_count: int = MIN_CLOUD_COUNT) -> dict[str, int]:
    subset = df[(df["stage"] == stage) & (df["count"] >= min_count)]
    return dict(zip(subset["word"], subset["count"]))


def draw_wordcloud(
    frequencies: dict[str, int],
    path: Path,
    title: str,
    subtitle: str,
    colormap: str,
    background: str,
    font_path: str | None,
) -> None:
    """Gambar satu WordCloud beserta judul dan keterangan ukuran data."""
    if not frequencies:
        raise ValueError(f"Tidak ada kata dengan frekuensi >= {MIN_CLOUD_COUNT} untuk {title}")

    cloud = WordCloud(
        width=CLOUD_WIDTH,
        height=CLOUD_HEIGHT,
        background_color=background,
        colormap=colormap,
        max_words=MAX_WORDS,
        prefer_horizontal=0.9,
        relative_scaling=0.5,
        min_font_size=12,
        collocations=False,
        random_state=42,
        font_path=font_path,
    ).generate_from_frequencies(frequencies)

    total_words = sum(frequencies.values())
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=background)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")

    ax.set_title(title, fontsize=26, fontweight="bold", color="white" if background.startswith("#0") or background == "#1a1a1a" else "#1b1b1b", pad=24)
    fig.text(
        0.5,
        0.925,
        f"{len(frequencies):,} kata unik  |  {total_words:,} total kata (frekuensi >= {MIN_CLOUD_COUNT})".replace(",", "."),
        ha="center",
        fontsize=12,
        color="#d9d9d9" if background.startswith("#0") or background == "#1a1a1a" else "#4a4a4a",
    )
    fig.text(
        0.5,
        0.03,
        subtitle,
        ha="center",
        fontsize=11,
        style="italic",
        color="#c9c9c9" if background.startswith("#0") or background == "#1a1a1a" else "#6a6a6a",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=CLOUD_DPI, facecolor=background)
    plt.close(fig)


def draw_top20(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Bar chart horizontal Top 20 kata hasil preprocessed (FR-33)."""
    top = df[df["stage"] == "preprocessed"].nlargest(20, "count").copy()
    if top.empty:
        raise ValueError("Tidak ada data frekuensi untuk tahap preprocessed")

    top = top.iloc[::-1]  # kata terbesar di atas
    total = int(df.loc[df["stage"] == "preprocessed", "count"].sum())
    labels = [f"{word}  ({count / total * 100:.1f}%)" for word, count in zip(top["word"], top["count"])]

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(labels, top["count"], color="#2a6fdb", edgecolor="white")
    for bar, count in zip(bars, top["count"]):
        ax.text(bar.get_width() + total * 0.002, bar.get_y() + bar.get_height() / 2,
                f"{count:,}".replace(",", "."), va="center", fontsize=10, color="#333333")

    ax.set_title(
        f"Top 20 Kata Paling Sering Muncul Setelah Preprocessing (n = {total:,} token)".replace(",", "."),
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Frekuensi kemunculan")
    ax.set_ylabel("Kata")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return top.iloc[::-1]


def main() -> int:
    setup_logging()
    lines: list[str] = []

    def say(message: str = "") -> None:
        lines.append(message)
        logger.info(message)

    try:
        frequencies = load_frequencies()
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1

    stages = list(CLOUD_STYLE)
    missing = [stage for stage in stages if stage not in set(frequencies["stage"])]
    if missing:
        logger.error("Tabel frekuensi kehilangan tahap: %s", missing)
        return 1

    font_path = find_font()

    say("=" * 70)
    say("VISUALISASI WORDCLOUD & TOP 20 KATA (PHASE 7)")
    say("=" * 70)
    say(f"Input frekuensi : {FREQ_CSV.relative_to(PROJECT_ROOT)}")
    say(f"Font            : {Path(font_path).name if font_path else 'default WordCloud'}")
    say(f"Parameter       : maks {MAX_WORDS} kata, minimal {MIN_CLOUD_COUNT}x muncul, {CLOUD_WIDTH}x{CLOUD_HEIGHT} px")
    say()

    outputs: list[tuple[str, Path, dict[str, int]]] = []
    for stage in stages:
        freq = frequencies_for(frequencies, stage)
        if not freq:
            logger.error("Tahap '%s' tidak punya kata dengan frekuensi >= %d.", stage, MIN_CLOUD_COUNT)
            return 1
        path = RESULTS_DIR / f"wordcloud_{stage}.png"
        title, subtitle, colormap, background = CLOUD_STYLE[stage]
        draw_wordcloud(freq, path, title, subtitle, colormap, background, font_path)
        outputs.append((stage, path, freq))

        say("-" * 70)
        say(f"{title.upper()}")
        say("-" * 70)
        say(f"  Kata unik  : {len(freq):,} (frekuensi >= {MIN_CLOUD_COUNT})".replace(",", "."))
        say(f"  Total kata : {sum(freq.values()):,}".replace(",", "."))
        say(f"  File       : {path.relative_to(PROJECT_ROOT)}")
        top_ten = sorted(freq.items(), key=lambda item: item[1], reverse=True)[:10]
        say("  10 teratas :")
        for index, (word, count) in enumerate(top_ten, start=1):
            say(f"    {index:>2}. {word:<18}{count}")
        say()

    try:
        top20 = draw_top20(frequencies, RESULTS_DIR / "top_20_words.png")
    except ValueError as error:
        logger.error("%s", error)
        return 1

    say("-" * 70)
    say("BAR CHART TOP 20 KATA")
    say("-" * 70)
    say(f"  File : {(RESULTS_DIR / 'top_20_words.png').relative_to(PROJECT_ROOT)}")
    for index, row in enumerate(top20.itertuples(index=False), start=1):
        say(f"  {index:>2}. {row.word:<18}{row.count}")
    say("=" * 70)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Ringkasan disimpan di: %s", SUMMARY_PATH.relative_to(PROJECT_ROOT))

    for _, path, _ in outputs:
        if not path.exists() or path.stat().st_size == 0:
            logger.error("File visualisasi gagal dibuat: %s", path)
            return 1
    if not (RESULTS_DIR / "top_20_words.png").exists():
        logger.error("Bar chart Top 20 gagal dibuat.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
