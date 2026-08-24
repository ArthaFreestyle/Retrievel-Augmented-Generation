"""Unduh, ekstraksi teks, dan normalisasi dokumen sumber LPMB.

Kontrak keluaran: data/processed/documents.jsonl (skema §5.1, plus field
`extractable` — lihat §6 notebook 01 di perancangan.md). Dipakai oleh
notebooks/01_ingest_documents.ipynb.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 65536


def sha256_of_file(path: str | Path) -> str:
    """Hitung sha256 sebuah berkas, dipakai untuk deduplikasi & audit korpus."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


def scan_local_pdfs(dir_path: str | Path) -> list[Path]:
    """Daftar semua *.pdf di dir_path (tidak rekursif), terurut nama."""
    return sorted(Path(dir_path).glob("*.pdf"))


def download_pdf(url: str, dest_dir: str | Path, timeout: int = 30) -> Path:
    """Unduh satu PDF ke dest_dir. Idempoten terhadap re-run: jika sebuah
    berkas dengan nama yang sama dan sha256 identik sudah ada, tidak
    diunduh ulang."""
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").split("/")[-1] or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "skripsi-rag-lpmb/0.1"})
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    return dest_path


def snapshot_web_page(url: str, dest_dir: str | Path, timeout: int = 30) -> Path:
    """Simpan salinan HTML mentah sebuah laman ke dest_dir (arsip, karena
    laman LPMB berubah antar-periode). Nama berkas = slug(url) + tanggal
    UTC, sehingga snapshot pada tanggal berbeda tidak saling menimpa."""
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "skripsi-rag-lpmb/0.1"})
    resp.raise_for_status()

    slug = re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")[-80:]
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest_path = dest_dir / f"{date_tag}__{slug}.html"
    dest_path.write_text(resp.text, encoding="utf-8")
    return dest_path


def extract_text_pymupdf(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Ekstraksi teks per halaman dengan PyMuPDF.
    Return: [{"page": int, "text": str}, ...]."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        return [{"page": i, "text": page.get_text("text")} for i, page in enumerate(doc, start=1)]
    finally:
        doc.close()


def extract_text_pdfplumber(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Ekstraksi teks per halaman dengan pdfplumber's extract_text() polos
    (tanpa penanganan tabel khusus). Dipertahankan sebagai utilitas mandiri;
    jalur aktif ingesti memakai extract_text_auto()."""
    import pdfplumber

    pages: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append({"page": i, "text": page.extract_text() or ""})
    return pages


def _clean_table_cell(cell: str | None) -> str:
    """Satu sel tabel -> satu baris teks rapi: newline internal (isi sel
    bertingkat, mis. "Lokal: maks. 30\\njuta") jadi "; ", spasi berlebih
    dirapikan, dan "|" diloloskan supaya tidak merusak sintaks tabel
    Markdown."""
    if not cell:
        return ""
    flat = cell.replace("\n", " ; ").replace("|", "/")
    return re.sub(r"\s+", " ", flat).strip()


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Ubah rows dari pdfplumber Table.extract() jadi satu tabel Markdown.

    Baris pertama dipakai sebagai header. Kolom yang kosong di SEMUA baris
    (artefak sel gabungan colspan/rowspan — pdfplumber mengisi None pada
    sel yang "tertutup" oleh sel gabungan) dibuang, supaya tabel tidak
    penuh kolom hantu kosong. Header berbaris-banyak (mis. sub-kolom
    "Fakultas"/"LPMB" di bawah "Biaya") TIDAK direkonstruksi/digabung
    otomatis — baris header kedua tetap tampil apa adanya sebagai baris
    data pertama. Yang dijaga di sini hanya KOHESI PER-BARIS: nilai suatu
    baris (skema, jumlah anggota, biaya) tidak pernah tercampur ke baris
    lain, yang justru masalah utama extract_text() pada tabel berkolom
    banyak (lihat README, temuan ekstraksi Tabel 2.3).

    Return "" (bukan tabel) jika < 2 kolom nyata tersisa setelah dibuang
    kolom hantu, atau tidak ada baris data selain header — find_tables()
    kerap mendeteksi sel header berbaris-banyak sebagai "tabel" 1-kolom
    tersendiri yang bersarang di dalam tabel sungguhan (lihat README,
    kasus "Mitra Sasaran" hal. 32); tabel semacam ini bukan tabel asli,
    jadi bbox-nya juga TIDAK dikecualikan dari ekstraksi teks naratif oleh
    pemanggil (extract_text_auto) — kontennya tetap muncul lewat teks
    biasa, bukan hilang."""
    if not rows:
        return ""
    cleaned = [[_clean_table_cell(c) for c in row] for row in rows]
    n_cols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (n_cols - len(r)) for r in cleaned]
    keep_cols = [c for c in range(n_cols) if any(r[c] for r in cleaned)]
    if len(keep_cols) < 2:
        return ""
    cleaned = [[r[c] for c in keep_cols] for r in cleaned]

    header, *data_rows = cleaned
    if not data_rows:
        return ""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in data_rows)
    return "\n".join(lines)


def extract_text_auto(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Gabungkan PyMuPDF (default, teks umum) dengan pdfplumber (halaman
    yang terdeteksi memuat tabel via find_tables()) — lihat §6 notebook 01.

    Untuk halaman bertabel: teks naratif diekstraksi dengan bbox tabel yang
    LOLOS _rows_to_markdown() (>= 2 kolom nyata + >= 1 baris data) DIFILTER
    KELUAR (page.filter()) sebelum extract_text(), supaya isi tabel tidak
    muncul dua kali (sekali sebagai teks naratif yang terinterleaving acak,
    sekali sebagai tabel Markdown). Tabel yang DITOLAK _rows_to_markdown()
    (mis. "tabel" 1-kolom hasil find_tables() salah mendeteksi sel header
    berbaris-banyak sebagai tabel bersarang tersendiri) sengaja TIDAK
    ikut difilter dari teks naratif — bbox-nya dibiarkan, supaya kontennya
    tetap muncul lewat extract_text() biasa, bukan hilang begitu saja.

    Tabel yang lolos diekstraksi terpisah lewat find_tables()/.extract()
    dan dirender ke Markdown (_rows_to_markdown) — lihat notebooks/02 &
    src/chunking.py untuk bagaimana blok tabel ini nanti dijadikan SATU
    chunk utuh yang tidak pernah dipecah.

    Return: [{"page": int, "text": str, "tables": list[str],
              "method": "pymupdf"|"pdfplumber_table"}, ...]
    """
    import pdfplumber

    pymupdf_by_page = {p["page"]: p["text"] for p in extract_text_pymupdf(pdf_path)}

    merged: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            kept_tables_md: list[str] = []
            kept_bboxes: list[tuple[float, float, float, float]] = []
            for t in page.find_tables():
                md = _rows_to_markdown(t.extract())
                if md:
                    kept_tables_md.append(md)
                    kept_bboxes.append(t.bbox)

            if kept_bboxes:

                def _outside_tables(obj, bboxes=kept_bboxes):
                    x0, top, x1, bottom = obj.get("x0", 0), obj.get("top", 0), obj.get("x1", 0), obj.get(
                        "bottom", 0
                    )
                    return not any(
                        x0 >= bx0 - 1 and x1 <= bx1 + 1 and top >= btop - 1 and bottom <= bbot + 1
                        for (bx0, btop, bx1, bbot) in bboxes
                    )

                narrative = page.filter(_outside_tables).extract_text() or ""
                merged.append({"page": i, "text": narrative, "tables": kept_tables_md, "method": "pdfplumber_table"})
            else:
                merged.append({"page": i, "text": pymupdf_by_page.get(i, ""), "tables": [], "method": "pymupdf"})
    return merged


def is_extractable(pages: list[dict[str, Any]], min_chars_per_page: float = 20.0) -> bool:
    """Heuristik: dokumen dianggap TIDAK extractable (kemungkinan besar
    gambar/poster) jika rata-rata karakter teks per halaman di bawah
    ambang. Dicatat, bukan dibuang (Batasan Masalah #2)."""
    if not pages:
        return False
    total_chars = sum(len(p["text"].strip()) for p in pages)
    return (total_chars / len(pages)) >= min_chars_per_page


def make_doc_id(title: str, year: int, existing_ids: set[str] | None = None) -> str:
    """Buat doc_id unik berformat lpmb-{year}-{slug(title)} (§5.1). Jika
    bertabrakan dengan existing_ids, tambahkan sufiks numerik."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    base = f"lpmb-{year}-{slug}"[:60].rstrip("-")
    doc_id = base
    existing_ids = existing_ids or set()
    i = 2
    while doc_id in existing_ids:
        doc_id = f"{base}-{i}"
        i += 1
    return doc_id


def write_interim_text(doc_id: str, pages: list[dict[str, Any]], out_dir: str | Path) -> Path:
    """Tulis teks hasil ekstraksi satu dokumen ke data/interim/{doc_id}.txt,
    dengan penanda halaman agar notebook 02 dapat memulihkan page_start/end.

    Tabel (jika ada di p["tables"], lihat extract_text_auto) disisipkan
    setelah teks naratif halaman, masing-masing dibungkus penanda
    `=== TABEL HALAMAN {n} #{j} ===` / `=== AKHIR TABEL ===`. Format ini
    HARUS sinkron dengan src/chunking.py::_TABLE_BLOCK_RE, yang mem-parsing
    blok ini sebagai unit atomik (satu tabel = satu chunk, tidak pernah
    ikut dipecah recursive_chunk maupun digabung dengan paragraf lain)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.txt"
    blocks = []
    for p in pages:
        block = f"=== HALAMAN {p['page']} ===\n{p['text']}"
        for j, table_md in enumerate(p.get("tables") or [], start=1):
            block += f"\n\n=== TABEL HALAMAN {p['page']} #{j} ===\n{table_md}\n=== AKHIR TABEL ==="
        blocks.append(block)
    out_path.write_text("\n\n".join(blocks), encoding="utf-8")
    return out_path


def build_document_record(
    doc_id: str,
    title: str,
    source_type: str,
    source_url: str | None,
    local_path: str,
    program: str,
    year: int,
    n_pages: int,
    sha256: str,
    extractable: bool = True,
) -> dict[str, Any]:
    """Rakit satu baris sesuai skema data/processed/documents.jsonl (§5.1).

    Dokumen yang hanya berupa gambar/poster HARUS tetap dicatat dengan
    extractable=False, bukan dibuang (lihat Batasan Masalah #2) — jumlahnya
    dilaporkan sebagai temuan di naskah.
    """
    return {
        "doc_id": doc_id,
        "title": title,
        "source_type": source_type,
        "source_url": source_url,
        "local_path": local_path,
        "sha256": sha256,
        "program": program,
        "year": year,
        "n_pages": n_pages,
        "retrieved_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "extractable": extractable,
    }


def write_documents_jsonl(records: list[dict[str, Any]], out_path: str | Path) -> None:
    """Tulis daftar record dokumen ke JSONL, satu objek per baris."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
