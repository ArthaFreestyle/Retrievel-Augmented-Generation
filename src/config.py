"""Loader konfigurasi & util reproducibility bersama untuk seluruh notebook.

Objek Config yang dihasilkan load_config() adalah SATU-SATUNYA hal yang
membedakan ketiga kondisi ablation A/B/C (P1) — lihat src/pipeline.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class PathsConfig:
    data_raw_pdf: str = "data/raw/pdf"
    data_raw_web: str = "data/raw/web"
    data_interim: str = "data/interim"
    data_processed: str = "data/processed"
    data_benchmark: str = "data/benchmark"
    indexes_faiss: str = "indexes/faiss"
    indexes_bm25: str = "indexes/bm25"
    runs: str = "runs"
    results: str = "results"
    figures: str = "figures"
    cache_db: str = "cache/llm_cache.sqlite"


@dataclass
class ModelsConfig:
    embedding: str = "BAAI/bge-m3"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    generator: str = "gemini-3.6-flash"
    judge: str = "gemini-3.1-pro-preview"


@dataclass
class ChunkingConfig:
    chunk_size: int = 2000
    soft_max: int = 2600
    chunk_overlap: int = 200
    unit: str = "char"
    min_chunk_chars: int = 250
    max_dot_ratio: float = 0.20


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"  # none | dense | hybrid
    top_k_dense: int = 5
    top_k_bm25: int = 20
    top_k_context: int = 5


@dataclass
class RerankConfig:
    enabled: bool = False
    top_n: int = 5


@dataclass
class GeneratorConfig:
    temperature: float = 0.0
    max_output_tokens: int = 1024
    system_prompt_path: str = "configs/prompts/system_prompt.txt"


@dataclass
class ExperimentConfig:
    limit: int | None = 20
    run_id: str | None = None


@dataclass
class ApiConfig:
    gemini_api_key_env: str = "GEMINI_API_KEY"


@dataclass
class Config:
    condition: str = "base"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


_SECTION_TYPES: dict[str, type] = {
    "paths": PathsConfig,
    "models": ModelsConfig,
    "chunking": ChunkingConfig,
    "retrieval": RetrievalConfig,
    "rerank": RerankConfig,
    "generator": GeneratorConfig,
    "experiment": ExperimentConfig,
    "api": ApiConfig,
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _raw_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    extends = data.pop("extends", None)
    if extends:
        parent = _raw_yaml(path.parent / extends)
        data = _deep_merge(parent, data)
    return data


def load_config(path: str | Path) -> Config:
    """Muat file YAML (mendukung `extends: base.yaml`) menjadi Config.

    Kunci top-level yang tidak dikenal (mis. `project:`) diabaikan.
    """
    raw = _raw_yaml(Path(path))

    kwargs: dict[str, Any] = {}
    for key in ("condition", "seed"):
        if key in raw:
            kwargs[key] = raw[key]

    for section, cls in _SECTION_TYPES.items():
        if section in raw and isinstance(raw[section], dict):
            kwargs[section] = cls(**raw[section])

    return Config(**kwargs)


def set_seed(seed: int) -> None:
    """Set seed Python/NumPy/PyTorch (jika terpasang) agar hasil dapat direproduksi."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_directories(cfg: Config, project_root: str | Path = ".") -> list[Path]:
    """Buat seluruh direktori pada cfg.paths, plus runs/cond_{a,b,c}, jika belum ada."""
    root = Path(project_root)
    created: list[Path] = []
    for f in fields(cfg.paths):
        rel = getattr(cfg.paths, f.name)
        p = root / rel
        target = p.parent if p.suffix else p  # cache_db menunjuk ke sebuah file
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    for cond in ("cond_a", "cond_b", "cond_c"):
        p = root / cfg.paths.runs / cond
        p.mkdir(parents=True, exist_ok=True)
        created.append(p)
    return created
