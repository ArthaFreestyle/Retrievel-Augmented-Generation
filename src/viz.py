"""Gaya gambar seragam untuk seluruh figures/*.png (§10 notebook 10)."""
from __future__ import annotations

from typing import Any


def set_style() -> None:
    """Terapkan gaya matplotlib seragam (font, dpi, palet warna) untuk
    seluruh figure Bab IV. Panggil sekali di awal notebook 10."""
    raise NotImplementedError


def save_figure(fig: Any, out_path: str, dpi: int = 300) -> None:
    """Simpan figure ke figures/ pada 300 dpi sesuai checklist kesiapan (§10)."""
    raise NotImplementedError
