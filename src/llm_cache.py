"""Cache SQLite berbasis hash prompt untuk seluruh panggilan Gemini API (P3).

Kunci = sha256(model + prompt + parameter generasi). Efeknya: menjalankan
ulang notebook 05/07 dengan cache penuh harus nyaris gratis dan hasilnya
identik antar-eksekusi.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def cache_key(model: str, prompt: str, params: dict[str, Any]) -> str:
    """sha256(model + prompt + parameter generasi, mis. temperature) -> kunci cache."""
    raise NotImplementedError


def get_cached(db_path: str | Path, key: str) -> dict[str, Any] | None:
    """Ambil respons tersimpan untuk `key`, atau None jika cache miss."""
    raise NotImplementedError


def set_cached(db_path: str | Path, key: str, response: dict[str, Any]) -> None:
    """Simpan respons + metadata (model, timestamp) untuk `key`."""
    raise NotImplementedError


def call_with_cache(
    db_path: str | Path,
    model: str,
    prompt: str,
    params: dict[str, Any],
    call_fn: Callable[[str, str, dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Cek cache dulu; jika miss, panggil call_fn(model, prompt, params) lalu
    simpan hasilnya.

    Return: (response, cache_hit). cache_hit dipakai untuk MENGECUALIKAN
    baris ini dari statistik latensi (§6 notebook 05) — mengikutsertakannya
    akan membuat angka latensi tidak bermakna.
    """
    raise NotImplementedError
