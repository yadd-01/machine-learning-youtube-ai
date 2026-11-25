"""Kamus normalisasi kata tidak baku (Phase 5).

Kamus disimpan di file CSV terpisah (`normalization_dict.csv`) supaya bisa
diedit manual tanpa mengubah kode (FR-22).

Format CSV:
    tidak_baku,baku,catatan
    gak,tidak,gaul

Aturan:
    - Satu token dipetakan ke satu token (tanpa spasi).
    - Kata yang tidak ada di kamus dibiarkan apa adanya.
    - Baris diawali '#' diabaikan.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = PROJECT_ROOT / "preprocessing" / "normalization_dict.csv"

logger = logging.getLogger("normalization")


def load_dictionary(path: Path = DICT_PATH) -> dict[str, str]:
    """Muat kamus normalisasi dan validasi isinya.

    Raises:
        FileNotFoundError: file kamus tidak ada.
        ValueError: kolom wajib tidak ada, atau ada entri tidak valid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Kamus normalisasi tidak ditemukan: {path}")

    df = pd.read_csv(
        path,
        comment="#",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    required = {"tidak_baku", "baku"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} kehilangan kolom: {sorted(missing)}")

    dictionary: dict[str, str] = {}
    conflicts: list[str] = []
    invalid: list[str] = []

    for row in df.itertuples(index=False):
        key = str(row.tidak_baku).strip().lower()
        value = str(row.baku).strip().lower()
        if not key and not value:
            continue
        if not key or not value or " " in key or " " in value:
            invalid.append(f"{key!r} -> {value!r}")
            continue
        if key in dictionary and dictionary[key] != value:
            conflicts.append(f"{key}: {dictionary[key]!r} vs {value!r}")
            continue
        dictionary[key] = value

    if invalid:
        raise ValueError(
            f"Entri kamus tidak valid (harus satu token tanpa spasi): {invalid[:5]}"
        )
    if conflicts:
        raise ValueError(f"Kamus punya pemetaan ganda: {conflicts[:5]}")

    logger.info("Kamus normalisasi dimuat: %d entri dari %s.", len(dictionary), path.name)
    return dictionary


def normalize_tokens(tokens: list[str], dictionary: dict[str, str]) -> list[str]:
    """Ganti setiap token tidak baku dengan bentuk bakunya (jika ada di kamus)."""
    return [dictionary.get(token, token) for token in tokens]
