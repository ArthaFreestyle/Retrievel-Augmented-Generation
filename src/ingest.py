"""Unduh, ekstraksi teks, dan normalisasi dokumen sumber LPMB.

Kontrak keluaran: data/processed/documents.jsonl (skema §5.1, plus field
`extractable` — lihat §6 notebook 01 di perancangan.md). Dipakai oleh
notebooks/01_ingest_documents.ipynb.
"""
from __future__ import annotations

import hashlib
import warnings
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
    UTC, sehingga snapshot pada tanggal berbeda tidak saling menimpa.

    Menulis juga sidecar `{dest_path.stem}.meta.json` berisi `{"url",
    "retrieved_at"}` di sebelah berkas HTML — slug nama berkas dipotong/
    diubah dari url aslinya, jadi tanpa sidecar ini url sumber tidak bisa
    dipulihkan lagi saat ekstraksi (lihat read_web_snapshot_meta)."""
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "skripsi-rag-lpmb/0.1"})
    resp.raise_for_status()

    slug = re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")[-80:]
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest_path = dest_dir / f"{date_tag}__{slug}.html"
    dest_path.write_text(resp.text, encoding="utf-8")

    retrieved_at = datetime.now(timezone.utc).astimezone().isoformat()
    meta_path = dest_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({"url": url, "retrieved_at": retrieved_at}, ensure_ascii=False),
        encoding="utf-8",
    )
    return dest_path


def read_web_snapshot_meta(html_path: str | Path) -> dict[str, Any] | None:
    """Baca sidecar `{html_path.stem}.meta.json` yang ditulis
    snapshot_web_page() (url asli + retrieved_at). None jika tidak ada
    (mis. snapshot ditaruh manual tanpa lewat snapshot_web_page)."""
    meta_path = Path(html_path).with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def scan_local_html(dir_path: str | Path) -> list[Path]:
    """Daftar semua *.html di dir_path (tidak rekursif), terurut nama."""
    return sorted(Path(dir_path).glob("*.html"))


def extract_text_html(html_path: str | Path) -> list[dict[str, Any]]:
    """Ekstraksi teks utama dari satu snapshot HTML.

    Beda dari PDF, laman web tidak berpaginasi — dibungkus sebagai satu
    "halaman" (page=1) supaya konsumen hilir (write_interim_text,
    is_extractable, build_document_record) yang sudah dirancang untuk
    list[{"page", "text", ...}] tetap dipakai tanpa cabang kode terpisah.

    Tag `script`/`style`/`nav`/`header`/`footer`/`noscript` dibuang
    sebelum ekstraksi teks (murni derau navigasi/layout, bukan isi
    laman), lalu baris kosong berlebih dirapikan."""
    from bs4 import BeautifulSoup

    html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return [{"page": 1, "text": text, "tables": [], "method": "bs4"}]


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


# Penanda baris judul (huruf tebal) di data/interim/*.txt. HARUS sinkron
# dengan src/chunking.py::BOLD_MARK. Deteksi judul di chunking memakai
# penanda ini (bukan pola nomor saja), karena butir daftar bernomor
# ("2. Publikasi Jurnal ...") dan judul sub-bagian ("2. Tahap Pengusulan")
# berpola sama -- yang membedakan hanya ketebalan hurufnya.
BOLD_MARK = "[[B]] "
ROW_SPLIT_CHARS = 2600  # tabel yang lebih panjang dari ini dipecah per baris (sekitar soft_max chunking)
_NUMBERING_SPAN_RE = re.compile(r"^\s*(?:\d{1,3}|[A-Za-z])?\s*[.)]?\s*$")


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _bold_line_keys(pdf_path: str | Path) -> dict[int, set[str]]:
    """Per halaman: himpunan baris (spasi dibuang) yang seluruh teksnya
    tebal menurut PyMuPDF. Span yang hanya berisi nomor butir ("9.") tidak
    dihitung -- di PDF ini nomor kadang tidak ikut tebal padahal judulnya
    tebal."""
    import pymupdf

    keys: dict[int, set[str]] = {}
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            found: set[str] = set()
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    body = [s for s in spans if not _NUMBERING_SPAN_RE.match(s["text"])]
                    if body and all(s["flags"] & 16 for s in body):
                        found.add(_squash("".join(s["text"] for s in spans)))
            keys[i] = found
    finally:
        doc.close()
    return keys


def _mark_bold_lines(text: str, bold_keys: set[str]) -> str:
    out = []
    for ln in text.splitlines():
        if ln.strip() and _squash(ln) in bold_keys:
            ln = BOLD_MARK + ln.strip()
        out.append(ln)
    return "\n".join(out)


def _merged_cols(lead: dict[str, Any]) -> tuple[int, ...]:
    """Jumlah kolom tabel pembuka setelah kolom-hantu digabung / yang berisi data (di-cache di tabelnya)."""
    if "merged_cols" not in lead:
        try:
            try:
                from src.table_clean import merged_cols
            except ImportError:  # notebook: sys.path "../src"
                from table_clean import merged_cols
            lead["merged_cols"] = merged_cols(lead["frags"][0])
        except Exception:
            lead["merged_cols"] = ()
    return lead["merged_cols"]


def _pages_with_broken_block_order(pdf_path: str | Path) -> set[int]:
    """Halaman yang urutan blok teks PyMuPDF-nya melompat ke atas (blok berikutnya mulai
    > 20 pt di atas blok sebelumnya, di luar header/footer), mis. daftar "a. Lokal / b. Regional"
    yang keluar sesudah teks di bawahnya. Teks halaman seperti itu dibaca per baris oleh pdfplumber."""
    import pymupdf

    broken: set[int] = set()
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            h = page.rect.height
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip() and b[3] > h * 0.07 and b[1] < h * 0.93]
            if any(b[1] < a[1] - 20 and b[0] < a[2] for a, b in zip(blocks, blocks[1:])):
                broken.add(i)
    finally:
        doc.close()
    return broken


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

    Baris yang seluruhnya tebal diberi awalan BOLD_MARK (lihat
    _bold_line_keys) -- dipakai chunking untuk membedakan judul asli dari
    butir daftar.

    Return: [{"page": int, "text": str, "tables": list[str],
              "method": "pymupdf"|"pdfplumber_table"}, ...]
    """
    import pdfplumber

    try:
        from src import table_luaran
    except ImportError:  # notebook: sys.path "../src"
        import table_luaran
    try:
        from src.table_clean import table_to_text
    except ImportError:  # notebook: sys.path "../src"
        from table_clean import table_to_text

    pymupdf_by_page = {p["page"]: p["text"] for p in extract_text_pymupdf(pdf_path)}
    broken_order = _pages_with_broken_block_order(pdf_path)
    bold_keys = _bold_line_keys(pdf_path)

    merged: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        # Lintasan 1: kumpulkan tabel lolos-seleksi per halaman dan kelompokkan
        # potongan antarhalaman (tabel di dasar halaman n + tabel di puncak
        # halaman n+1 dengan jumlah kolom mentah sama = satu tabel).
        found: dict[int, list[dict[str, Any]]] = {}
        luaran_map, luaran_notes = table_luaran.extract_all(pdf)  # Tabel B-I Lampiran 2: satu baris = satu blok
        for note in luaran_notes:
            warnings.warn(f"tabel luaran: {note}")
        last_open: dict[str, Any] | None = None  # tabel terakhir halaman sebelumnya, bila menjorok ke dasar
        for i, page in enumerate(pdf.pages, start=1):
            found[i] = []
            for t in page.find_tables():
                rows = t.extract()
                md = _rows_to_markdown(rows)
                if not md and table_luaran.table_key(i, t.bbox) not in luaran_map:
                    continue
                n_cols = max(len(r) for r in rows)
                cont = (
                    last_open is not None
                    and n_cols in (last_open["n_cols"], *_merged_cols(last_open["lead"]))
                    and t.bbox[1] < page.height * 0.2
                    and not found[i]
                )
                item = {"rows": rows, "bbox": t.bbox, "md": md, "n_cols": n_cols,
                        "lead": last_open["lead"] if cont else None}
                if table_luaran.table_key(i, t.bbox) in luaran_map:
                    item["luaran"] = table_luaran.table_key(i, t.bbox)
                if cont:
                    item["lead"]["frags"].append(rows)
                    item["lead"]["frag_pages"].append(i)
                else:
                    item["frags"] = [rows]
                    item["frag_pages"] = [i]
                    item["lead"] = item
                found[i].append(item)
            tail = found[i][-1] if found[i] else None
            last_open = tail if tail and tail["bbox"][3] > page.height * 0.8 else None

        for i, page in enumerate(pdf.pages, start=1):
            kept_tables_md: list[str] = []
            kept_bboxes: list[tuple[float, float, float, float]] = []
            kept_notes: list[str] = []
            for item in found[i]:
                kept_bboxes.append(item["bbox"])
                if item.get("luaran"):
                    # tiap baris jadi blok tabel sendiri (= satu chunk), di halaman tempat baris itu dimulai
                    sep = f"\n=== AKHIR TABEL ===\n=== TABEL HALAMAN {i} #{len(kept_tables_md) + 1} ===\n"
                    kept_tables_md.append(sep.join(luaran_map[item["luaran"]]))
                    continue
                lead = item["lead"]
                if "txt" not in lead:  # hitung sekali per tabel (potongan antarhalaman sudah terkumpul)
                    try:
                        lead["txt"], lead["notes"], lines = table_to_text(lead["frags"], with_rows=True)
                    except Exception:  # tabel janggal -> kembali ke Markdown, jangan gagalkan ingest
                        lead["txt"], lead["notes"], lines = lead["md"], ["table_to_text gagal, dipakai Markdown"], []
                    # tabel besar: satu baris = satu blok (= satu chunk), di halaman tempat baris itu berada
                    lead["split"] = None
                    if len(lines) > 1 and len(lead["txt"]) > ROW_SPLIT_CHARS:
                        lead["split"] = {}
                        for frag_no, line in lines:
                            lead["split"].setdefault(lead["frag_pages"][frag_no], []).append(line)
                if lead["split"] is not None:
                    sep = f"\n=== AKHIR TABEL ===\n=== TABEL HALAMAN {i} #{len(kept_tables_md) + 1} ===\n"
                    kept_tables_md.append(sep.join(lead["split"].get(i, [])))
                elif lead is not item:
                    kept_tables_md.append("")  # sambungan: isinya sudah dirender di tabel pembukanya
                    continue
                else:
                    kept_tables_md.append(lead["txt"])
                if lead is item:
                    kept_notes += [f"hal. {i}: {n}" for n in lead["notes"]]

            if kept_bboxes:

                def _outside_tables(obj, bboxes=kept_bboxes):
                    x0, top, x1, bottom = obj.get("x0", 0), obj.get("top", 0), obj.get("x1", 0), obj.get(
                        "bottom", 0
                    )
                    return not any(
                        x0 >= bx0 - 1 and x1 <= bx1 + 1 and top >= btop - 1 and bottom <= bbot + 1
                        for (bx0, btop, bx1, bbot) in bboxes
                    )

                # Tabel disisipkan di POSISINYA (urut dari atas halaman) lewat
                # pemotongan halaman per pita vertikal, bukan ditumpuk di akhir
                # halaman -- supaya tabel mewarisi judul bagian yang benar saat
                # satu halaman memuat beberapa judul.
                filtered = page.filter(_outside_tables)
                order = sorted(range(len(kept_bboxes)), key=lambda k: kept_bboxes[k][1])
                parts: list[str] = []
                y = 0.0
                for n, k in enumerate(order, start=1):
                    top, bottom = kept_bboxes[k][1], kept_bboxes[k][3]
                    if top - y > 1:
                        parts.append(filtered.crop((0, y, page.width, min(top, page.height))).extract_text() or "")
                    if kept_tables_md[k]:
                        parts.append(f"=== TABEL HALAMAN {i} #{n} ===\n{kept_tables_md[k]}\n=== AKHIR TABEL ===")
                    y = max(y, bottom)
                if page.height - y > 1:
                    parts.append(filtered.crop((0, y, page.width, page.height)).extract_text() or "")
                text = ("\n\n".join(_mark_bold_lines(t, bold_keys.get(i, set())) if not t.startswith("=== TABEL") else t for t in parts if t.strip()))
                merged.append(
                    {"page": i, "text": text, "tables": [t for t in kept_tables_md if t], "tables_inline": True,
                     "table_notes": kept_notes, "method": "pdfplumber_table"}
                )
            else:
                if i in broken_order:
                    text = _mark_bold_lines(page.extract_text() or "", bold_keys.get(i, set()))
                    merged.append({"page": i, "text": text, "tables": [], "method": "pdfplumber_order"})
                    continue
                text = _mark_bold_lines(pymupdf_by_page.get(i, ""), bold_keys.get(i, set()))
                merged.append({"page": i, "text": text, "tables": [], "method": "pymupdf"})
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
    setelah teks naratif halaman -- kecuali p["tables_inline"], yang berarti
    tabel sudah tertanam di p["text"] pada posisinya -- masing-masing dibungkus penanda
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
        for j, table_md in enumerate([] if p.get("tables_inline") else (p.get("tables") or []), start=1):
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


def assemble_documents(
    rows: Any,
    extracted: dict[Any, list[dict[str, Any]]],
    source_type: str,
    existing_ids: set[str],
    interim_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Loop satu sumber (baris metadata + teks hasil ekstraksi per berkas)
    -> record documents.jsonl + teks interim tertulis. Dipakai notebook 01
    untuk PDF maupun web agar loop perakitan record tidak diduplikasi dua
    kali antar-sumber (P4).

    `rows` iterable via itertuples() (mis. DataFrame pandas) dengan kolom
    title/year/local_path/filename/sha256/program, dan opsional
    `source_url` (dipakai bila kolomnya ada, None bila tidak — hanya
    snapshot web yang punya ini, lewat sidecar meta.json).

    `existing_ids` dimutasi in-place supaya doc_id tetap unik lintas
    PANGGILAN berturutan (mis. PDF dulu baru web, pada set yang sama).

    Return: (records, filenames_not_extractable)."""
    records: list[dict[str, Any]] = []
    not_extractable: list[str] = []

    for row in rows.itertuples():
        pages = extracted[row.local_path]
        doc_id = make_doc_id(row.title, row.year, existing_ids)
        existing_ids.add(doc_id)

        extractable = is_extractable(pages)
        if not extractable:
            not_extractable.append(row.filename)

        write_interim_text(doc_id, pages, interim_dir)

        records.append(
            build_document_record(
                doc_id=doc_id,
                title=row.title,
                source_type=source_type,
                source_url=getattr(row, "source_url", None),
                local_path=Path(*Path(row.local_path).parts[1:]).as_posix(),
                program=row.program,
                year=row.year,
                n_pages=len(pages),
                sha256=row.sha256,
                extractable=extractable,
            )
        )
    return records, not_extractable


def write_documents_jsonl(records: list[dict[str, Any]], out_path: str | Path) -> None:
    """Tulis daftar record dokumen ke JSONL, satu objek per baris."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
