"""Wrapper Gemini API dengan structured output untuk generasi jawaban.

Skema output: {"abstain": bool, "answer": str, "citations": [chunk_id]}.
Deteksi abstain berbasis boolean, bukan pencocokan frasa regex (§6 notebook 05).
"""
from __future__ import annotations

from typing import Any

PROMPT_TEMPLATE_ID = "rag_answer_v1"


def build_prompt(question: str, context_chunks: list[dict[str, Any]]) -> str:
    """Rakit satu prompt template yang identik untuk kondisi B & C.

    Untuk kondisi A (closed-book), context_chunks kosong tetapi instruksi
    abstain tetap disertakan — jika tidak, kondisi A dihukum secara tidak
    adil pada metrik abstain dan ablation-nya cacat (§6 notebook 05).
    """
    raise NotImplementedError


def generate(
    question: str,
    context_chunks: list[dict[str, Any]],
    generator_cfg: Any,
) -> dict[str, Any]:
    """Panggil Gemini (lewat src.llm_cache.call_with_cache) dengan structured
    output.

    Return sesuai skema Prediction §5.4 (subset yang menjadi tanggung jawab
    generator): answer, abstained, citations, tokens, cache_hit.
    """
    raise NotImplementedError
