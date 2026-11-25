"""Scraping komentar YouTube untuk analisis text mining topik AI.

Alur:
    1. Baca daftar video dari config/videos.txt (URL atau Video ID).
    2. Ambil komentar top-level lewat YouTube Data API v3 (commentThreads.list).
    3. Deduplikasi berdasarkan comment_id, termasuk terhadap hasil run sebelumnya.
    4. Simpan progres ke dataset/raw_comments.csv setiap selesai satu video,
       sehingga scraping bisa dilanjutkan besok jika kuota habis.

Jalankan dari root proyek:
    python scraping/scraping.py --dry-run    # cek daftar video tanpa memanggil API
    python scraping/scraping.py              # scraping penuh
    python scraping/scraping.py --fresh      # abaikan CSV lama, mulai dari nol
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------------------------------------------------------------------------
# Konstanta path (semua relatif terhadap root proyek)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_CSV = DATASET_DIR / "raw_comments.csv"
DEFAULT_VIDEOS_FILE = PROJECT_ROOT / "config" / "videos.txt"
LOG_FILE = PROJECT_ROOT / "scraping.log"

CSV_COLUMNS = [
    "video_id",
    "comment_id",
    "author",
    "text",
    "published_at",
    "like_count",
]

PAGE_SIZE = 100  # maksimum yang diizinkan API, sekaligus paling hemat kuota
MAX_RETRY = 3

# Alasan error API yang perlu penanganan khusus.
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}
DISABLED_REASONS = {"commentsDisabled", "videoCommentsDisabled"}
NOT_FOUND_REASONS = {"videoNotFound", "notFound"}

logger = logging.getLogger("scraping")


class QuotaExhausted(RuntimeError):
    """Kuota API harian habis; scraping harus dihentikan dan dilanjutkan lain waktu."""


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    """Log ke konsol dan ke scraping.log."""
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(file_handler)


def extract_video_id(value: str) -> str | None:
    """Ambil Video ID (11 karakter) dari berbagai format URL YouTube.

    Mendukung: watch?v=, youtu.be/, /shorts/, /embed/, /live/, /v/, dan ID polos.
    Mengembalikan None jika tidak ada ID yang valid.
    """
    value = value.strip()
    if not value:
        return None

    patterns = [
        r"(?:youtube\.com|youtube-nocookie\.com)/watch\?(?:[^#]*&)?v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)

    # ID polos (tanpa URL)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    return None


def load_video_list(path: Path) -> list[str]:
    """Baca daftar video dari file teks, buang duplikat, pertahankan urutan."""
    if not path.exists():
        raise FileNotFoundError(
            f"File daftar video tidak ditemukan: {path}\n"
            "Buat file tersebut (lihat contoh di config/videos.txt)."
        )

    video_ids: list[str] = []
    invalid: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        video_id = extract_video_id(line)
        if video_id is None:
            invalid.append(raw_line.strip())
        elif video_id not in video_ids:
            video_ids.append(video_id)

    if invalid:
        logger.warning("Baris tidak dikenali dan dilewati (%d): %s", len(invalid), invalid)

    return video_ids


def load_existing(path: Path) -> tuple[pd.DataFrame, set[str]]:
    """Muat CSV hasil run sebelumnya (untuk melanjutkan scraping)."""
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS), set()

    df = pd.read_csv(
        path,
        encoding="utf-8",
        dtype={"video_id": str, "comment_id": str},
        keep_default_na=False,
    )
    missing = [col for col in CSV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{path} tidak punya kolom wajib: {missing}. "
            "Hapus file tersebut atau jalankan dengan --fresh."
        )

    df = df[CSV_COLUMNS]
    seen = set(df["comment_id"].astype(str))
    logger.info("Melanjutkan dari %s: %d komentar sudah tersimpan.", path.name, len(df))
    return df, seen


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Simpan CSV secara atomik (tulis ke file sementara lalu ganti)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Pemanggilan API
# ---------------------------------------------------------------------------
def _http_error_reason(error: HttpError) -> str:
    """Ambil 'reason' dari body error Google API."""
    try:
        payload = json.loads(error.content.decode("utf-8"))
        errors = payload.get("error", {}).get("errors", [])
        if errors:
            return str(errors[0].get("reason", ""))
    except (ValueError, AttributeError):
        pass
    return ""


def _execute_with_retry(request: Any) -> dict[str, Any]:
    """Jalankan request API, ulangi hanya untuk error server/jaringan."""
    delay = 2.0
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return request.execute()
        except HttpError as error:
            status = int(getattr(error.resp, "status", 0) or 0)
            if status >= 500 and attempt < MAX_RETRY:
                logger.warning("API error %s, ulangi dalam %.0f dtk (%d/%d).", status, delay, attempt, MAX_RETRY)
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (ConnectionError, TimeoutError, OSError) as error:
            if attempt < MAX_RETRY:
                logger.warning("Gangguan jaringan (%s), ulangi dalam %.0f dtk (%d/%d).", error, delay, attempt, MAX_RETRY)
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("Permintaan API gagal setelah beberapa percobaan.")


def fetch_comments(youtube: Any, video_id: str, max_needed: int) -> list[dict[str, Any]]:
    """Ambil komentar top-level sebuah video sampai habis atau max_needed terpenuhi."""
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    page = 0

    while len(rows) < max_needed:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=PAGE_SIZE,
            order="relevance",
            textFormat="plainText",
            pageToken=page_token,
        )
        try:
            response = _execute_with_retry(request)
        except HttpError as error:
            reason = _http_error_reason(error)
            status = int(getattr(error.resp, "status", 0) or 0)

            if reason in QUOTA_REASONS:
                raise QuotaExhausted(f"Kuota API habis (reason={reason}).") from error
            if reason in DISABLED_REASONS or (status == 403 and not reason):
                logger.warning("Video %s: komentar tidak dapat diakses (reason=%s). Dilewati.", video_id, reason or "-")
                break
            if reason in NOT_FOUND_REASONS or status == 404:
                logger.warning("Video %s tidak ditemukan. Dilewati.", video_id)
                break
            raise

        items = response.get("items", [])
        page += 1
        logger.info("Video %s: halaman %d, %d komentar.", video_id, page, len(items))

        for item in items:
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            comment_id = item.get("id") or snippet.get("id")
            if not comment_id:
                continue
            rows.append(
                {
                    "video_id": video_id,
                    "comment_id": comment_id,
                    "author": snippet.get("authorDisplayName", ""),
                    "text": snippet.get("textOriginal") or snippet.get("textDisplay", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "like_count": int(snippet.get("likeCount", 0) or 0),
                }
            )
            if len(rows) >= max_needed:
                break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return rows


# ---------------------------------------------------------------------------
# Orkestrasi
# ---------------------------------------------------------------------------
def collect_all(
    youtube: Any,
    video_ids: Iterable[str],
    target: int,
    df: pd.DataFrame,
    seen: set[str],
) -> tuple[pd.DataFrame, bool]:
    """Kumpulkan komentar dari daftar video sampai target tercapai.

    Mengembalikan (dataframe, quota_exhausted).
    """
    for video_id in video_ids:
        remaining = target - len(df)
        if remaining <= 0:
            break

        logger.info("--- Video %s (butuh %d komentar lagi) ---", video_id, remaining)
        try:
            rows = fetch_comments(youtube, video_id, remaining)
        except QuotaExhausted:
            logger.error("Kuota API habis. Progres disimpan; lanjutkan besok dengan perintah yang sama.")
            return df, True

        new_rows = [row for row in rows if row["comment_id"] not in seen]
        for row in new_rows:
            seen.add(row["comment_id"])

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows, columns=CSV_COLUMNS)], ignore_index=True)
            df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
            save_csv(df, RAW_CSV)

        logger.info(
            "Video %s selesai: +%d komentar baru (duplikat dilewati: %d). Total unik: %d.",
            video_id,
            len(new_rows),
            len(rows) - len(new_rows),
            len(df),
        )
        time.sleep(0.5)  # jeda sopan antar video

    return df, False


def report(df: pd.DataFrame, target: int) -> bool:
    """Cetak ringkasan akhir dan kembalikan True jika target benar-benar tercapai."""
    unique_ids = df["comment_id"].nunique() if not df.empty else 0
    logger.info("=" * 60)
    logger.info("Total baris       : %d", len(df))
    logger.info("comment_id unik   : %d", unique_ids)
    logger.info("Video terwakili   : %d", df["video_id"].nunique() if not df.empty else 0)
    logger.info("Target            : %d", target)

    if unique_ids >= target:
        logger.info("STATUS: target tercapai (%d >= %d).", unique_ids, target)
        return True

    logger.warning("STATUS: target BELUM tercapai (%d < %d).", unique_ids, target)
    logger.warning("Tambahkan video cadangan di config/videos.txt lalu jalankan ulang (progres otomatis dilanjutkan).")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scraping komentar YouTube untuk analisis text mining AI.")
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS_FILE, help="File daftar video (default: config/videos.txt).")
    parser.add_argument("--target", type=int, default=None, help="Target komentar unik (default: TARGET_COMMENTS di .env atau 2000).")
    parser.add_argument("--fresh", action="store_true", help="Abaikan dataset/raw_comments.csv dan mulai dari nol.")
    parser.add_argument("--dry-run", action="store_true", help="Hanya tampilkan daftar video yang terbaca, tanpa memanggil API.")
    args = parser.parse_args(argv)

    setup_logging()
    load_dotenv(PROJECT_ROOT / ".env")

    target = args.target or int(os.getenv("TARGET_COMMENTS", "2000"))
    if target <= 0:
        logger.error("Target komentar harus lebih besar dari 0.")
        return 1

    try:
        video_ids = load_video_list(args.videos)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    if not video_ids:
        logger.error("Tidak ada Video ID valid di %s.", args.videos)
        return 1

    logger.info("Daftar video: %d video unik dari %s.", len(video_ids), args.videos.name)

    if args.dry_run:
        for index, video_id in enumerate(video_ids, start=1):
            logger.info("  %2d. https://www.youtube.com/watch?v=%s", index, video_id)
        logger.info("Dry run selesai. Tidak ada panggilan API dan tidak ada file yang ditulis.")
        return 0

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key or api_key == "isi_api_key_anda":
        logger.error(
            "YOUTUBE_API_KEY belum diisi. Salin .env.example menjadi .env lalu isi API Key. "
            "Panduan ada di README.md bagian 'Membuat YouTube Data API Key'."
        )
        return 1

    if args.fresh and RAW_CSV.exists():
        logger.info("--fresh: menghapus %s.", RAW_CSV.name)
        RAW_CSV.unlink()

    try:
        df, seen = load_existing(RAW_CSV)
    except (ValueError, pd.errors.ParserError) as error:
        logger.error("Gagal membaca %s: %s", RAW_CSV, error)
        return 1

    if len(df) >= target:
        logger.info("Dataset sudah berisi %d komentar (target %d). Tidak ada yang perlu diambil.", len(df), target)
        report(df, target)
        return 0

    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    try:
        df, quota_exhausted = collect_all(youtube, video_ids, target, df, seen)
    except KeyboardInterrupt:
        logger.warning("Dihentikan pengguna. Progres terakhir sudah tersimpan di %s.", RAW_CSV)
        return 130
    except HttpError as error:
        logger.error("Error API tidak terduga: %s", error)
        return 1

    reached = report(df, target)
    if quota_exhausted:
        return 2
    return 0 if reached else 3


if __name__ == "__main__":
    sys.exit(main())
