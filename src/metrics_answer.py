"""Wrapper RAGAS + metrik abstain terpisah (§7 notebook 07)."""
from __future__ import annotations

from typing import Any


def run_ragas(predictions: list[dict[str, Any]], benchmark: list[dict[str, Any]]) -> Any:
    """Hitung faithfulness, answer_relevance, answer_correctness (judge =
    Gemini) untuk pertanyaan dalam-cakupan.

    Untuk kondisi closed-book (konteks kosong), faithfulness TIDAK
    terdefinisi: laporkan sebagai n/a, JANGAN diisi 0 — mengisi 0
    menyesatkan pembacaan tabel hasil (§7 catatan perancangan).
    """
    raise NotImplementedError


def abstain_recall(predictions: list[dict[str, Any]], benchmark: list[dict[str, Any]]) -> float:
    """Proporsi item luar-cakupan yang benar ditolak sistem."""
    raise NotImplementedError


def false_answer_rate(predictions: list[dict[str, Any]], benchmark: list[dict[str, Any]]) -> float:
    """Proporsi item luar-cakupan yang tetap dijawab — metrik keselamatan
    paling penting pada domain ini."""
    raise NotImplementedError


def over_refusal_rate(predictions: list[dict[str, Any]], benchmark: list[dict[str, Any]]) -> float:
    """Proporsi item dalam-cakupan yang justru ditolak sistem."""
    raise NotImplementedError
