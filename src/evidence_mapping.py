"""Pemetaan evidence (doc_id, halaman, span_text) -> chunk_id (P2).

Ini yang membuat benchmark tetap hidup walau strategi chunking berubah:
ground-truth TIDAK PERNAH diikat langsung ke chunk_id, hanya ke span teks.
"""
from __future__ import annotations

from typing import Any


def find_span_in_document(span_text: str, doc_id: str, page: int, min_ratio: float = 0.9) -> bool:
    """Cek span_text benar-benar ditemukan dalam teks dokumen pada halaman
    yang diklaim (fuzzy match, rasio >= min_ratio). Menangkap salah ketik
    evidence sejak notebook 04 (§6 notebook 04)."""
    raise NotImplementedError


def map_evidence_to_chunks(
    evidence: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[str]:
    """Untuk setiap span evidence, temukan chunk yang memuatnya
    -> daftar relevant_chunk_ids."""
    raise NotImplementedError


def build_qrels(benchmark: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hasilkan data/benchmark/qrels.jsonl: {"qid", "relevant_chunk_ids"}
    untuk setiap item dalam-cakupan (item luar-cakupan tidak punya qrels)."""
    raise NotImplementedError
