"""Bangun indeks dense (FAISS) dan sparse (BM25) dari chunks.jsonl.

Dipakai oleh notebooks/03_build_indexes.ipynb dan src/retrievers.py.

Kontrak posisi<->chunk_id: `encode_chunks` mengembalikan embedding dalam
urutan `chunks` yang diberikan; `build_faiss_index` dan `build_bm25_index`
menerima `chunk_ids` sejajar urutan itu dan menyimpan pemetaannya sendiri
di `out_dir`, sehingga `load_faiss_index`/`load_bm25_index` bisa
mengembalikan `chunk_id` yang benar untuk tiap hit tanpa notebook pemanggil
perlu melacak urutan secara manual.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = None
_stopword_remover = None


def encode_chunks(
    chunks: list[dict[str, Any]],
    model_name: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """Encode teks seluruh chunk dengan sentence-transformers (fp16, batch)
    -> matriks embedding (n_chunks x dim)."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray, chunk_ids: list[str], out_dir: str | Path) -> None:
    """Bangun IndexFlatIP + normalisasi L2 dan simpan ke out_dir, beserta
    pemetaan posisi -> chunk_id. Jangan pakai IVF/HNSW pada korpus sekecil
    ini — hanya menambah variabel pengganggu (§3 notebook 03)."""
    import faiss

    if len(chunk_ids) != embeddings.shape[0]:
        raise ValueError(
            f"jumlah chunk_ids ({len(chunk_ids)}) != jumlah baris embeddings ({embeddings.shape[0]})"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normed = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(normed)

    index = faiss.IndexFlatIP(normed.shape[1])
    index.add(normed)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)


def load_faiss_index(index_dir: str | Path) -> tuple[Any, list[str]]:
    """Muat indeks FAISS beserta pemetaan posisi -> chunk_id.

    Return: (index, chunk_ids) — chunk_ids[i] adalah chunk_id untuk
    vektor pada posisi i di index."""
    import faiss

    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "chunk_ids.json", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    return index, chunk_ids


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer


def _get_stopword_remover():
    global _stopword_remover
    if _stopword_remover is None:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        _stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return _stopword_remover


def tokenize_id(text: str) -> list[str]:
    """Tokenisasi + stopword removal + stemming Bahasa Indonesia (Sastrawi),
    dipakai untuk BM25. Stopword removal dijalankan sebelum stemming karena
    kamus stopword Sastrawi sudah berisi banyak bentuk berimbuhan umum,
    sehingga mengurangi jumlah kata yang perlu distem."""
    cleaned = _get_stopword_remover().remove(text.lower())
    stemmed = _get_stemmer().stem(cleaned)
    return _TOKEN_RE.findall(stemmed)


def build_bm25_index(chunks: list[dict[str, Any]], out_dir: str | Path) -> None:
    """Bangun indeks rank_bm25 dari token setiap chunk dan simpan (termasuk
    konfigurasi tokenizer) ke out_dir."""
    from rank_bm25 import BM25Okapi

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = [c["chunk_id"] for c in chunks]
    tokenized_corpus = [tokenize_id(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": chunk_ids,
                "tokenizer": "indexing.tokenize_id",
            },
            f,
        )


def load_bm25_index(index_dir: str | Path) -> dict[str, Any]:
    """Muat indeks BM25 beserta tokenizer yang SAMA dengan yang dipakai
    saat build — wajib konsisten agar skor BM25 valid saat query.

    Return: {"bm25": BM25Okapi, "chunk_ids": list[str], "tokenizer": str}."""
    index_dir = Path(index_dir)
    with open(index_dir / "bm25.pkl", "rb") as f:
        return pickle.load(f)
