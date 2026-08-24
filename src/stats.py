"""Statistik: kappa, korelasi, uji signifikansi berpasangan, bootstrap CI.

Dipakai oleh notebooks/09_metric_validation.ipynb dan
notebooks/10_analysis_and_figures.ipynb.
"""
from __future__ import annotations

from typing import Any, Callable


def cohens_kappa(ratings_a: list[int], ratings_b: list[int], weights: str | None = None) -> float:
    """Cohen's kappa antar dua anotator. weights='quadratic' untuk skala
    ordinal 1-4 (§9 notebook 09) — selisih 1<->4 diberi bobot lebih berat
    daripada selisih 3<->4."""
    raise NotImplementedError


def bootstrap_ci(
    values: list[float],
    statistic: Callable[[list[float]], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Interval kepercayaan bootstrap (percentile) untuk sebuah statistik."""
    raise NotImplementedError


def correlation(x: list[float], y: list[float], method: str = "pearson") -> float:
    """method: 'pearson' (linear) atau 'spearman' (monotonik)."""
    raise NotImplementedError


def paired_significance_test(
    scores_a: list[float],
    scores_b: list[float],
    method: str = "wilcoxon",
) -> dict[str, float]:
    """Uji berpasangan (Wilcoxon signed-rank / bootstrap berpasangan) untuk
    dua kondisi yang dievaluasi pada pertanyaan yang sama — BUKAN uji t
    independen (§10 notebook 10)."""
    raise NotImplementedError


def holm_correction(p_values: list[float]) -> list[float]:
    """Koreksi Holm untuk perbandingan berganda (3 pasangan kondisi A-B,
    A-C, B-C)."""
    raise NotImplementedError
