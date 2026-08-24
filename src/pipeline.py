"""Runner tunggal untuk ketiga kondisi ablation (P1).

WAJIB: seluruh eksperimen (cond_a/b/c) memanggil run_pipeline() ini,
dibedakan HANYA oleh objek Config. Dilarang menyalin-tempel logika pipeline
ke notebook terpisah — itu meruntuhkan validitas klaim ablation (lihat
perancangan.md §9, risiko "Kode pipeline disalin ke 3 notebook").
"""
from __future__ import annotations

import time
from typing import Any

from . import generator, retrievers
from .config import Config


def run_pipeline(question: str, cfg: Config) -> dict[str, Any]:
    """Jalankan satu pertanyaan lewat retrieval (opsional) + generasi.

    cfg.retrieval.mode menentukan kondisi:
      "none"   -> cond_a (closed-book)
      "dense"  -> cond_b (naive RAG)
      "hybrid" -> cond_c (hybrid + rerank, jika cfg.rerank.enabled)

    Return: dict dengan field retrieved/answer/abstained/citations/
    latency_ms, sesuai skema Prediction §5.4 (qid/condition/run_id diisi
    oleh notebook 05 pemanggil, bukan di sini).
    """
    t0 = time.perf_counter()
    ctx: list[dict[str, Any]] = []
    latency_ms: dict[str, float] = {}

    if cfg.retrieval.mode != "none":
        t_retr = time.perf_counter()
        cands = retrievers.dense_search(question, k=cfg.retrieval.top_k_dense)
        if cfg.retrieval.mode == "hybrid":
            bm25_cands = retrievers.bm25_search(question, k=cfg.retrieval.top_k_bm25)
            cands = retrievers.rrf_fuse(cands, bm25_cands)
        latency_ms["retrieval"] = (time.perf_counter() - t_retr) * 1000

        if cfg.rerank.enabled:
            t_rerank = time.perf_counter()
            cands = retrievers.cross_encoder_rerank(question, cands, top_n=cfg.rerank.top_n)
            latency_ms["rerank"] = (time.perf_counter() - t_rerank) * 1000

        ctx = cands[: cfg.retrieval.top_k_context]

    t_gen = time.perf_counter()
    result = generator.generate(question, ctx, cfg.generator)
    latency_ms["generation"] = (time.perf_counter() - t_gen) * 1000
    latency_ms["total"] = (time.perf_counter() - t0) * 1000

    result["retrieved"] = ctx
    result["latency_ms"] = latency_ms
    return result
