"""Jalur retrieval: Dense, BM25, fusi RRF, dan reranking cross-encoder.

Dipakai oleh src/pipeline.py::run_pipeline — satu jalur kode untuk ketiga
kondisi ablation (P1).
"""
from __future__ import annotations

from typing import Any


def dense_search(query: str, k: int) -> list[dict[str, Any]]:
    """Cari k kandidat teratas dari indeks FAISS.
    Return: [{"chunk_id": str, "score_dense": float}, ...] terurut menurun."""
    raise NotImplementedError


def bm25_search(query: str, k: int) -> list[dict[str, Any]]:
    """Cari k kandidat teratas dari indeks BM25.
    Return: [{"chunk_id": str, "score_bm25": float}, ...] terurut menurun."""
    raise NotImplementedError


def rrf_fuse(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion dua daftar kandidat menjadi satu daftar
    tergabung dengan field score_rrf, terurut menurun."""
    raise NotImplementedError


def cross_encoder_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """Re-skor kandidat dengan cross-encoder (BAAI/bge-reranker-v2-m3) dan
    kembalikan top_n dengan field score_rerank tambahan."""
    raise NotImplementedError
