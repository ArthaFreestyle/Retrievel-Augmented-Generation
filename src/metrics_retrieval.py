"""Metrik deterministik retrieval: recall@k, MRR, nDCG (§6 notebook 06).

Dihitung terhadap data/benchmark/qrels.jsonl, hanya untuk kondisi B & C.
"""
from __future__ import annotations

from typing import Any


def recall_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> float:
    """Proporsi relevant_chunk_ids yang muncul di top-k retrieved_chunk_ids."""
    raise NotImplementedError


def mrr(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str]) -> float:
    """Reciprocal rank dari kemunculan relevan pertama di retrieved_chunk_ids."""
    raise NotImplementedError


def ndcg_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @k (relevansi biner)."""
    raise NotImplementedError


def evaluate_retrieval(
    predictions: list[dict[str, Any]],
    qrels: dict[str, list[str]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Hitung recall@k / MRR / nDCG@10 agregat dan dipecah per kategori
    pertanyaan (§6 notebook 06)."""
    raise NotImplementedError
