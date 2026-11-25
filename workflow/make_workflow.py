"""Diagram workflow pipeline analisis komentar YouTube (Phase 8).

Menjawab FR-35: menghasilkan `workflow/workflow.png` yang menggambarkan alur
scraping sampai dokumentasi, lengkap dengan input/output dan angka nyata.

Angka pada diagram dibaca langsung dari dataset dan file konfigurasi, sehingga
tidak menjadi usang ketika data berubah.

Jalankan dari root proyek:
    python workflow/make_workflow.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "dataset" / "raw_comments.csv"
CLEANED_CSV = PROJECT_ROOT / "dataset" / "cleaned_comments.csv"
PREPROCESSED_CSV = PROJECT_ROOT / "dataset" / "preprocessed_comments.csv"
NOISE_CSV = PROJECT_ROOT / "dataset" / "noise_comments.csv"
FREQ_CSV = PROJECT_ROOT / "results" / "word_frequencies.csv"
TOP20_CSV = PROJECT_ROOT / "results" / "top_20_words.csv"
STOPWORDS_PATH = PROJECT_ROOT / "preprocessing" / "stopwords_id.txt"
NORMALIZATION_PATH = PROJECT_ROOT / "preprocessing" / "normalization_dict.csv"
EXCEPTIONS_PATH = PROJECT_ROOT / "preprocessing" / "stem_exceptions.csv"
OUTPUT_PATH = PROJECT_ROOT / "workflow" / "workflow.png"

# Warna per lapisan pipeline.
COLORS = {
    "akuisisi": "#1f6feb",
    "proses": "#2da44e",
    "analisis": "#bf8700",
    "visual": "#cf222e",
    "dokumen": "#57606a",
}

logger = logging.getLogger("workflow")


def setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(console)


def fmt(number: int | float) -> str:
    """Format angka gaya Indonesia: 3.000, 10,5%."""
    if isinstance(number, float):
        return f"{number:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{int(number):,}".replace(",", ".")


def count_lines(path: Path) -> int:
    """Hitung entri (baris bukan komentar/kosong) pada file teks/CSV."""
    if not path.exists():
        return 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if path.suffix == ".csv" and stripped.split(",")[0].strip() in {"kata", "tidak_baku"}:
            continue
        total += 1
    return total


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ditemukan. Jalankan tahap sebelumnya dulu.")
    return pd.read_csv(path, encoding="utf-8", keep_default_na=False)


def collect_metrics() -> dict[str, str]:
    raw = read_csv(RAW_CSV)
    cleaned = read_csv(CLEANED_CSV)
    preprocessed = read_csv(PREPROCESSED_CSV)
    noise = read_csv(NOISE_CSV)

    raw_text = raw["text"].astype(str)
    raw_words = raw_text.str.split().str.len()
    tokens = [token for text in preprocessed["text_preprocessed"] for token in str(text).split()]

    if FREQ_CSV.exists():
        freq_rows = len(pd.read_csv(FREQ_CSV, encoding="utf-8", keep_default_na=False))
    else:
        freq_rows = 0

    if TOP20_CSV.exists():
        top = pd.read_csv(TOP20_CSV, encoding="utf-8", keep_default_na=False).iloc[0]
        top_word, top_count = str(top["word"]), int(top["count"])
    else:
        top_word, top_count = "-", 0

    noise_count = len(noise)
    total = len(raw)
    return {
        "raw": fmt(total),
        "videos": fmt(raw["video_id"].nunique()),
        "median_chars": fmt(int(raw_text.str.len().median())),
        "median_words": fmt(int(raw_words.median())),
        "cleaned": fmt(len(cleaned)),
        "cleaned_pct": fmt(len(cleaned) / total * 100),
        "dropped": fmt(total - len(cleaned)),
        "tokens": fmt(len(tokens)),
        "vocab": fmt(len(set(tokens))),
        "dict": fmt(count_lines(NORMALIZATION_PATH)),
        "stopwords": fmt(count_lines(STOPWORDS_PATH)),
        "exceptions": fmt(count_lines(EXCEPTIONS_PATH)),
        "noise": fmt(noise_count),
        "noise_pct": fmt(noise_count / total * 100),
        "clean": fmt(total - noise_count),
        "freq_rows": fmt(freq_rows),
        "top_word": top_word,
        "top_count": fmt(top_count),
    }


def add_card(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    phase: str,
    title: str,
    script: str,
    output: str,
    metric: str,
    note: str,
    color: str,
) -> None:
    """Gambar satu kartu tahap pipeline pada koordinat axes (0-1)."""
    card = FancyBboxPatch(
        (x - width / 2, y),
        width,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.8,
        edgecolor=color,
        facecolor="#ffffff",
        zorder=3,
    )
    ax.add_patch(card)

    band = FancyBboxPatch(
        (x - width / 2 + 0.006, y + height - 0.048),
        width - 0.012,
        0.042,
        boxstyle="round,pad=0,rounding_size=0.008",
        linewidth=0,
        facecolor=color,
        alpha=0.96,
        zorder=4,
    )
    ax.add_patch(band)

    left = x - width / 2 + 0.014
    ax.text(x, y + height - 0.027, phase, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=5)
    ax.text(left, y + height - 0.083, title, ha="left", va="center",
            fontsize=12.5, fontweight="bold", color="#1b1b1b", zorder=5)

    ax.text(left, y + height - 0.128, script, ha="left", va="center",
            fontsize=8.2, family="monospace", color="#3b3b3b", zorder=5)
    ax.text(left, y + height - 0.163, output, ha="left", va="center",
            fontsize=8.2, color="#5a5a5a", zorder=5)
    ax.text(left, y + 0.062, metric, ha="left", va="center",
            fontsize=9.6, fontweight="bold", color=color, zorder=5)
    if "\n" in note:
        ax.text(left, y + 0.010, note, ha="left", va="bottom",
                fontsize=7.4, color="#606060", zorder=5, linespacing=1.25)
    else:
        ax.text(left, y + 0.020, note, ha="left", va="center",
                fontsize=7.6, color="#606060", zorder=5)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#8b949e") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.0,
        color=color,
        shrinkA=6,
        shrinkB=6,
        zorder=2,
    )
    ax.add_patch(arrow)


def main() -> int:
    setup_logging()

    try:
        metrics = collect_metrics()
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1

    width, height = 0.205, 0.265
    top_y, bottom_y = 0.575, 0.115
    centers_top = [0.14, 0.38, 0.62, 0.86]
    centers_bottom = [0.86, 0.62, 0.38, 0.14]

    fig, ax = plt.subplots(figsize=(16, 10), facecolor="#f6f7f9")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5, 0.958,
        "Workflow Analisis Komentar YouTube tentang AI",
        ha="center", fontsize=24, fontweight="bold", color="#14181f",
    )
    fig.text(
        0.5, 0.918,
        f"Pipeline batch Python — {metrics['raw']} komentar mentah dari {metrics['videos']} video, "
        "seluruh angka berasal dari data hasil scraping nyata",
        ha="center", fontsize=12, color="#4b5563",
    )

    add_card(
        ax, centers_top[0], top_y, width, height,
        phase="PHASE 2", title="Scraping komentar",
        script="scraping/scraping.py",
        output="dataset/raw_comments.csv",
        metric=f"{metrics['raw']} komentar • {metrics['videos']} video",
        note="YouTube Data API v3 • resume otomatis",
        color=COLORS["akuisisi"],
    )
    add_card(
        ax, centers_top[1], top_y, width, height,
        phase="PHASE 3", title="EDA dataset mentah",
        script="analysis/eda.py",
        output="results/eda_summary.txt + PNG",
        metric=f"median {metrics['median_chars']} karakter • {metrics['median_words']} kata",
        note="distribusi panjang • indikasi noise",
        color=COLORS["analisis"],
    )
    add_card(
        ax, centers_top[2], top_y, width, height,
        phase="PHASE 4", title="Cleaning teks",
        script="preprocessing/cleaning.py",
        output="dataset/cleaned_comments.csv",
        metric=f"{metrics['cleaned']} valid ({metrics['cleaned_pct']}%)",
        note=f"{metrics['dropped']} baris kosong/duplikat dibuang",
        color=COLORS["proses"],
    )
    add_card(
        ax, centers_top[3], top_y, width, height,
        phase="PHASE 5", title="Preprocessing teks",
        script="preprocessing/preprocessing.py",
        output="dataset/preprocessed_comments.csv",
        metric=f"{metrics['tokens']} token • {metrics['vocab']} kosakata",
        note=f"{metrics['dict']} kamus • {metrics['stopwords']} stopword\n{metrics['exceptions']} pengecualian stemming",
        color=COLORS["proses"],
    )

    add_card(
        ax, centers_bottom[0], bottom_y, width, height,
        phase="PHASE 6", title="Noise analysis",
        script="analysis/noise_analysis.py",
        output="dataset/noise_comments.csv",
        metric=f"{metrics['noise']} noise ({metrics['noise_pct']}%)",
        note=f"{metrics['clean']} komentar bersih dipertahankan",
        color=COLORS["analisis"],
    )
    add_card(
        ax, centers_bottom[1], bottom_y, width, height,
        phase="PHASE 7", title="Frekuensi kata",
        script="analysis/word_frequency.py",
        output="results/word_frequencies.csv",
        metric=f"{metrics['freq_rows']} baris • 4 tahap",
        note=f"kata terbanyak: {metrics['top_word']} ({metrics['top_count']})",
        color=COLORS["analisis"],
    )
    add_card(
        ax, centers_bottom[2], bottom_y, width, height,
        phase="PHASE 7", title="WordCloud + Top 20",
        script="visualization/make_wordcloud.py",
        output="4 WordCloud + bar chart (PNG)",
        metric="2.400 × 1.500 px",
        note="frekuensi ≥ 2 • maks 200 kata",
        color=COLORS["visual"],
    )
    add_card(
        ax, centers_bottom[3], bottom_y, width, height,
        phase="PHASE 9", title="Dokumentasi akhir",
        script="README.md + laporan",
        output="PRD.md • ARCHITECTURE.md",
        metric="13 bagian laporan",
        note="pendahuluan sampai saran",
        color=COLORS["dokumen"],
    )

    # Alur baris atas (kiri -> kanan).
    for left, right in zip(centers_top, centers_top[1:]):
        add_arrow(ax, (left + width / 2, top_y + height / 2), (right - width / 2, top_y + height / 2))

    # Turun dari preprocessing ke noise analysis.
    add_arrow(
        ax,
        (centers_top[3], top_y),
        (centers_bottom[0], bottom_y + height),
        color=COLORS["proses"],
    )

    # Alur baris bawah (kanan -> kiri).
    for right, left in zip(centers_bottom, centers_bottom[1:]):
        add_arrow(ax, (right - width / 2, bottom_y + height / 2), (left + width / 2, bottom_y + height / 2))

    # Keterangan lapisan.
    legend_y = 0.055
    items = [
        ("Akuisisi", COLORS["akuisisi"]),
        ("Pemrosesan", COLORS["proses"]),
        ("Analisis", COLORS["analisis"]),
        ("Visualisasi", COLORS["visual"]),
        ("Dokumentasi", COLORS["dokumen"]),
    ]
    legend_x = 0.10
    for label, color in items:
        ax.plot([legend_x], [legend_y], marker="s", markersize=8, color=color, linestyle="none")
        ax.text(legend_x + 0.012, legend_y, label, va="center", fontsize=8.5, color="#4b5563")
        legend_x += 0.105

    fig.text(
        0.5, 0.022,
        "Setiap tahap diperiksa sebelum lanjut • diagram dibuat otomatis oleh workflow/make_workflow.py",
        ha="center", fontsize=8.5, style="italic", color="#6b7280",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("=" * 66)
    logger.info("WORKFLOW (PHASE 8)")
    logger.info("=" * 66)
    logger.info("  Komentar mentah : %s dari %s video", metrics["raw"], metrics["videos"])
    logger.info("  Setelah cleaning: %s", metrics["cleaned"])
    logger.info("  Token/vocab     : %s / %s", metrics["tokens"], metrics["vocab"])
    logger.info("  Noise           : %s (%s%%)", metrics["noise"], metrics["noise_pct"])
    logger.info("  Frekuensi       : %s baris", metrics["freq_rows"])
    logger.info("  Output          : %s", OUTPUT_PATH.relative_to(PROJECT_ROOT))
    logger.info("=" * 66)

    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        logger.error("Gagal membuat diagram workflow.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
