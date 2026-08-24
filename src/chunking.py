"""Pembersihan teks & chunking rekursif berbasis struktur dokumen (heading & paragraf).

Strategi "structure-aware": dokumen dipecah dulu berdasarkan heading yang
terdeteksi (§6 notebook 02). Heading level "BAB" adalah batas KERAS (selalu
memulai chunk baru); heading level lain ("A.", "2.1.", dst) adalah batas
LUNAK — hanya dipakai untuk mengisi `section_heading`, tidak memaksa
pemotongan, supaya sub-bagian pendek tidak pecah jadi chunk-chunk kecil.
Paragraf yang lebih panjang dari chunk_size dipecah lebih lanjut lewat
recursive_chunk() (paragraf -> kalimat -> kata -> karakter).

Kontrak keluaran: data/processed/chunks.jsonl (skema §5.2).
Dipakai oleh notebooks/02_preprocess_chunking.ipynb.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_PAGE_MARKER_RE = re.compile(r"^=== HALAMAN (\d+) ===$", re.MULTILINE)
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$|^\s*\d{1,4}\s*$", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Format HARUS sinkron dengan src/ingest.py::write_interim_text.
_TABLE_BLOCK_RE = re.compile(r"=== TABEL HALAMAN \d+ #\d+ ===\n(.*?)\n=== AKHIR TABEL ===", re.DOTALL)

_HEADING_PATTERNS = [
    re.compile(r"^BAB\s+[IVXLCDM\d]+\b.*$", re.IGNORECASE),  # "BAB 2", "BAB II ..."
    re.compile(r"^[A-Z]\.\s+\S.*$"),                          # "A. Persyaratan Peserta"
    re.compile(r"^\d+(\.\d+)*\.?\s+\S.*$"),                   # "2.1. Pendahuluan", "1. ..."
]


def parse_interim_pages(interim_path: str | Path) -> list[dict[str, Any]]:
    """Baca kembali data/interim/{doc_id}.txt (ditulis oleh
    ingest.write_interim_text) menjadi [{"page": int, "text": str}, ...]."""
    raw = Path(interim_path).read_text(encoding="utf-8")
    parts = _PAGE_MARKER_RE.split(raw)
    pages: list[dict[str, Any]] = []
    for i in range(1, len(parts), 2):
        pages.append({"page": int(parts[i]), "text": parts[i + 1].strip("\n")})
    return pages


def _extract_table_blocks(raw_page_text: str) -> tuple[str, list[str]]:
    """Pisahkan blok tabel Markdown (ditulis ingest.write_interim_text) dari
    teks naratif satu halaman, SEBELUM clean_text() dipanggil — supaya
    baris Markdown ("| a | b |") tidak ikut kena heuristik pembersihan yang
    dirancang untuk prosa (deteksi header/footer berulang, baris nomor
    halaman berdiri sendiri).

    Return: (teks_naratif_tanpa_blok_tabel, [markdown_tabel, ...])."""
    tables = [t.strip() for t in _TABLE_BLOCK_RE.findall(raw_page_text)]
    narrative = _TABLE_BLOCK_RE.sub("", raw_page_text)
    return narrative, tables


def _detect_repeated_lines(pages: list[dict[str, Any]], min_ratio: float = 0.3) -> set[str]:
    """Baris yang muncul persis sama (setelah strip) pada >= min_ratio
    halaman dianggap header/footer berulang (mis. nama lembaga). Blok
    tabel dikeluarkan dulu supaya isi sel tabel tidak ikut mempengaruhi
    deteksi header/footer prosa."""
    counts: Counter[str] = Counter()
    for p in pages:
        narrative, _ = _extract_table_blocks(p["text"])
        lines = {ln.strip() for ln in narrative.splitlines() if ln.strip()}
        counts.update(lines)
    n_pages = max(len(pages), 1)
    return {line for line, c in counts.items() if c / n_pages >= min_ratio}


def clean_text(raw_text: str, repeated_lines: set[str] | None = None) -> str:
    """Buang header/footer berulang & baris nomor halaman berdiri sendiri,
    normalisasi whitespace."""
    repeated_lines = repeated_lines or set()
    lines = []
    for ln in raw_text.splitlines():
        stripped = ln.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped in repeated_lines or _PAGE_NUMBER_LINE_RE.match(stripped):
            continue
        lines.append(re.sub(r"[ \t]+", " ", stripped))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_DOT_LEADER_RE = re.compile(r"\.{3,}")


def detect_section_heading(line: str) -> str | None:
    """Deteksi heading berbasis pola umum dokumen LPMB ("BAB", "A.",
    "2.1."). Baris panjang (> 15 kata) tidak dianggap heading — heading
    biasanya berupa judul singkat, bukan kalimat penuh. Baris berisi dot
    leader (mis. "2.2. Program Pengabdian ..................... 5", entri
    daftar isi) juga ditolak — polanya "angka. Judul" cocok dengan
    _HEADING_PATTERNS yang sama dipakai untuk heading sungguhan seperti
    "2.1. Pendahuluan", padahal ini entri TOC, bukan heading (lihat
    README, temuan section_heading berisi baris daftar isi)."""
    stripped = line.strip()
    if not stripped or len(stripped.split()) > 15:
        return None
    if _DOT_LEADER_RE.search(stripped):
        return None
    for pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return stripped
    return None


def _heading_level(line: str) -> int:
    """Level 0 = BAB (batas keras); level 1 = sub-heading (batas lunak)."""
    return 0 if re.match(r"^BAB\s+", line, re.IGNORECASE) else 1


def _looks_like_heading_continuation(line: str) -> bool:
    """True jika baris pendek berikutnya kemungkinan lanjutan judul (mis.
    baris kedua judul BAB yang ditulis ALL CAPS / Title Case)."""
    stripped = line.strip()
    words = stripped.split()
    if not stripped or len(words) > 12:
        return False
    if stripped.isupper():
        return True
    capitalized = sum(1 for w in words if w[:1].isupper())
    return capitalized >= max(1, len(words) - 1)


def _split_paragraph_blocks(cleaned_text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", cleaned_text) if b.strip()]


def _word_boundary_tail(text: str, max_chars: int) -> str:
    """Ambil akhiran `text` sepanjang <= max_chars TANPA memenggal kata.
    text[-max_chars:] mentah bisa jatuh di tengah kata (mis. "Pelaksana"
    -> "elaksana") karena potongannya murni berbasis jumlah karakter;
    di sini kata/paragraf parsial di awal potongan dibuang, walau artinya
    overlap jadi lebih pendek dari max_chars. Kalau tidak ada spasi/newline
    sama sekali dalam jendela (satu kata lebih panjang dari max_chars),
    tidak ada potongan aman -> kembalikan string kosong."""
    if max_chars <= 0 or not text:
        return ""
    if max_chars >= len(text):
        return text.strip()
    tail = text[-max_chars:]
    if not text[-max_chars - 1].isspace():
        m = re.search(r"\s", tail)
        if m is None:
            return ""
        tail = tail[m.end() :]
    return tail.strip()


def recursive_chunk(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Pecah SATU teks (biasanya satu paragraf yang melebihi chunk_size)
    menjadi potongan <= chunk_size, menghormati batas paragraf/kalimat
    sebelum jatuh ke potongan kata, lalu karakter sebagai upaya terakhir.
    Potongan berurutan overlap sebesar chunk_overlap karakter, dipangkas ke
    batas kata terdekat (lihat _word_boundary_tail) supaya potongan tidak
    pernah mulai di tengah kata."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    units = _cascade_split(text)
    pieces: list[str] = []
    buf = ""
    for u in units:
        candidate = f"{buf} {u}".strip() if buf else u
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
            tail = _word_boundary_tail(buf, chunk_overlap)
            buf = f"{tail} {u}".strip() if tail else u
            if len(buf) <= chunk_size:
                continue
        # unit tunggal (mungkin gabungan tail+u) masih > chunk_size -> paksa potong per karakter
        step = max(chunk_size - chunk_overlap, 1)
        for i in range(0, len(buf), step):
            pieces.append(buf[i : i + chunk_size])
        buf = ""
    if buf:
        pieces.append(buf)
    return pieces


def _cascade_split(text: str) -> list[str]:
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) > 1:
        return sentences
    return text.split(" ")


def _extract_units(pages: list[dict[str, Any]], repeated_lines: set[str]) -> list[dict[str, Any]]:
    """Ubah halaman jadi urutan unit heading/paragraf/tabel, tiap unit
    menyimpan halaman asal & heading terakhir yang aktif. Unit "table"
    (dari blok Markdown yang disisipkan ingest.write_interim_text) selalu
    diletakkan SETELAH seluruh unit prosa halaman tsb — urutan relatif
    tabel terhadap teks di sekitarnya pada halaman yang sama tidak
    direkonstruksi persis, cukup untuk korpus ini karena tabel biasanya
    mendominasi sisa halaman setelah judulnya."""
    units: list[dict[str, Any]] = []
    current_heading: str | None = None

    for p in pages:
        narrative_raw, table_blocks = _extract_table_blocks(p["text"])
        cleaned = clean_text(narrative_raw, repeated_lines)
        for block in _split_paragraph_blocks(cleaned):
            block_lines = block.splitlines()
            heading_lines: list[str] = []
            idx = 0

            first = block_lines[0].strip()
            if detect_section_heading(first):
                heading_lines.append(first)
                idx = 1
                while idx < len(block_lines) and _looks_like_heading_continuation(block_lines[idx]):
                    heading_lines.append(block_lines[idx].strip())
                    idx += 1

            if heading_lines:
                heading_text = " ".join(heading_lines)
                current_heading = heading_text
                units.append(
                    {
                        "type": "heading",
                        "text": heading_text,
                        "page": p["page"],
                        "level": _heading_level(heading_lines[0]),
                    }
                )
                remaining = " ".join(l.strip() for l in block_lines[idx:] if l.strip())
                if remaining:
                    units.append({"type": "para", "text": remaining, "page": p["page"], "heading": current_heading})
            else:
                para_text = " ".join(l.strip() for l in block_lines if l.strip())
                if para_text:
                    units.append({"type": "para", "text": para_text, "page": p["page"], "heading": current_heading})

        for table_md in table_blocks:
            units.append({"type": "table", "text": table_md, "page": p["page"], "heading": current_heading})

    return units


def _units_to_pieces(
    units: list[dict[str, Any]], chunk_size: int, chunk_overlap: int, soft_max: int
) -> list[dict[str, Any]]:
    """Pecah paragraf yang > soft_max lewat recursive_chunk(); tandai piece
    pertama setelah heading BAB dengan hard_break_before. Unit "table"
    SELALU jadi piece standalone-nya sendiri, berapa pun panjangnya —
    tidak pernah lewat recursive_chunk (yang akan merusak strukturnya) dan
    tidak pernah digabung dengan paragraf lain oleh _pack_pieces (lihat
    penanganan "standalone" di sana): satu tabel = satu chunk utuh.

    soft_max (> chunk_size, mis. 1.3x) adalah TOLERANSI: paragraf sepanjang
    (chunk_size, soft_max] TIDAK dipaksa lewat recursive_chunk (pemotongan
    kalimat/kata yang mengorbankan koherensi semantik) — dibiarkan mengalir
    lewat _pack_pieces normal (bukan standalone), yang secara alami
    mengeluarkannya sebagai satu chunk sendiri begitu ukurannya sudah
    melebihi chunk_size (lihat overflow-check di _pack_pieces), lengkap
    dengan overlap ke piece tetangga. Hanya paragraf > soft_max yang
    sungguh dipaksa dipecah."""
    pieces: list[dict[str, Any]] = []
    pending_hard_break = False

    for u in units:
        if u["type"] == "heading":
            if u["level"] == 0:
                pending_hard_break = True
            continue

        if u["type"] == "table":
            pieces.append(
                {
                    "text": u["text"],
                    "page": u["page"],
                    "heading": u["heading"],
                    "hard_break_before": pending_hard_break,
                    "standalone": True,
                    "is_table": True,
                }
            )
            pending_hard_break = False
            continue

        oversized = len(u["text"]) > soft_max
        sub_texts = recursive_chunk(u["text"], chunk_size, chunk_overlap) if oversized else [u["text"]]
        for i, sub in enumerate(sub_texts):
            pieces.append(
                {
                    "text": sub,
                    "page": u["page"],
                    "heading": u["heading"],
                    "hard_break_before": pending_hard_break and i == 0,
                    # sub-piece dari recursive_chunk() sudah membawa overlap-nya
                    # sendiri antar-piece — jangan dipaketkan/di-overlap lagi
                    # oleh _pack_pieces, atau teks pada batasnya terduplikasi.
                    "standalone": oversized,
                }
            )
            pending_hard_break = False

    return pieces


_PARA_SEP = "\n\n"


def _pack_pieces(pieces: list[dict[str, Any]], chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    """Susun piece menjadi chunk <= chunk_size (batas lunak — lihat catatan
    toleransi di _units_to_pieces, piece tunggal antara chunk_size dan
    soft_max boleh keluar sebagai satu chunk sendirian lebih besar dari
    chunk_size). Tiap piece adalah satu paragraf utuh (§_extract_units) —
    saat >1 piece dipaketkan ke satu chunk, digabung dengan "\\n\\n" (bukan
    spasi) supaya batas paragraf tetap terlihat di teks chunk, bukan
    diratakan jadi satu kalimat panjang.

    Overlap antar-chunk mengambil UNIT UTUH: paragraf terakhir dari chunk
    sebelumnya (buf[-1]), bukan potongan N-karakter dari seluruh teks
    chunk (yang bisa memotong di tengah kalimat walau sudah snap ke batas
    kata). Hanya jika paragraf terakhir itu sendiri lebih panjang dari
    anggaran overlap, ia dipangkas ke batas kata terdekat
    (_word_boundary_tail) — tidak pernah memenggal kata. Anggaran overlap
    disesuaikan (adaptif) agar tetap muat bersama piece berikutnya di
    dalam chunk_size.

    Tiap chunk keluaran diberi flag internal "is_first_after_hard_break":
    True jika ia piece pertama setelah heading BAB — dipakai
    build_chunk_records untuk memutuskan chunk yatim mana yang AMAN dilebur
    ke chunk sebelumnya (tidak melebur lintas BAB)."""
    chunks: list[dict[str, Any]] = []
    buf: list[dict[str, Any]] = []
    pending_new_section = True  # True: chunk finalize berikutnya adalah yang pertama sejak start/hard break

    def buf_text() -> str:
        return _PARA_SEP.join(p["text"] for p in buf)

    def finalize() -> None:
        nonlocal pending_new_section
        text = buf_text().strip()
        if not text:
            return
        pages = [p["page"] for p in buf]
        heading = next((p["heading"] for p in buf if p["heading"]), None)
        chunks.append(
            {
                "text": text,
                "page_start": min(pages),
                "page_end": max(pages),
                "section_heading": heading,
                "is_table": False,
                "is_first_after_hard_break": pending_new_section,
            }
        )
        pending_new_section = False

    for piece in pieces:
        if piece.get("hard_break_before"):
            if buf:
                finalize()
                buf = []
            pending_new_section = True

        if piece.get("standalone"):
            # Piece tabel (utuh, tidak boleh dipecah) atau sub-piece
            # oversized yang sudah dipotong (+ di-overlap) oleh
            # recursive_chunk() sendiri; keluarkan apa adanya, jangan
            # digabung/di-overlap ulang di sini.
            finalize()
            buf = []
            chunks.append(
                {
                    "text": piece["text"].strip(),
                    "page_start": piece["page"],
                    "page_end": piece["page"],
                    "section_heading": piece["heading"],
                    "is_table": piece.get("is_table", False),
                    "is_first_after_hard_break": pending_new_section,
                }
            )
            pending_new_section = False
            continue

        sep_len = len(_PARA_SEP) if buf else 0
        piece_len = len(piece["text"])

        if buf and len(buf_text()) + sep_len + piece_len > chunk_size:
            last_piece_text = buf[-1]["text"]
            last_page = buf[-1]["page"]
            last_heading = buf[-1]["heading"]
            finalize()

            overlap_budget = min(chunk_overlap, max(chunk_size - piece_len - len(_PARA_SEP), 0))
            if len(last_piece_text) <= overlap_budget:
                overlap_text = last_piece_text
            else:
                overlap_text = _word_boundary_tail(last_piece_text, overlap_budget)
            buf = [{"text": overlap_text, "page": last_page, "heading": last_heading}] if overlap_text else []

        buf.append(piece)

    finalize()
    return chunks


def _approx_token_count(text: str) -> int:
    """Estimasi kasar (~4 karakter/token); BUKAN tokenizer sesungguhnya,
    hanya untuk diagnostik ukuran chunk (lihat n_tokens di §5.2)."""
    return max(1, round(len(text) / 4))


def _dot_ratio(text: str) -> float:
    """Fraksi karakter yang berupa titik. Tinggi (mis. > 0.25) menandakan
    dot leader daftar isi ("Kegiatan .......................... 5"), bukan
    prosa sungguhan."""
    return text.count(".") / len(text) if text else 0.0


def _merge_two(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Gabungkan dua chunk berurutan (a lalu b) jadi satu record chunk."""
    return {
        "text": f"{a['text']}{_PARA_SEP}{b['text']}",
        "page_start": min(a["page_start"], b["page_start"]),
        "page_end": max(a["page_end"], b["page_end"]),
        "section_heading": a["section_heading"] or b["section_heading"],
        "is_table": False,
        "is_first_after_hard_break": a.get("is_first_after_hard_break", False),
    }


def _merge_orphan_chunks(raw_chunks: list[dict[str, Any]], min_chunk_chars: int, soft_max: int) -> tuple[list[dict[str, Any]], int]:
    """Lebur chunk "yatim" (< min_chunk_chars, non-tabel) ke chunk
    TETANGGA alih-alih dibuang — teks pendek biasanya ekor/kepala sah
    suatu bagian, bukan sampah. Dua lintasan:

    1. Mundur (ke chunk sebelumnya) — pilihan utama, mempertahankan urutan
       teks natural. Tidak dilebur bila: chunk sebelumnya tabel, chunk itu
       piece pertama setelah heading BAB (melebur lintas BAB akan
       mencampur dua bab berbeda dalam satu chunk_id), tidak ada chunk
       sebelumnya (awal dokumen), atau hasil gabungannya akan melebihi
       soft_max.
    2. Maju (ke chunk berikutnya) — fallback untuk orphan yang lintasan (1)
       tidak bisa menanganinya, PALING SERING karena tetangga sebelumnya
       kebetulan tabel (banyak terjadi di korpus ini: caption/paragraf
       pendek sering langsung mengikuti tabel). Sama, tidak melebur lintas
       BAB maupun ke/dari tabel.

    Sisa orphan yang masih tidak bisa dilebur oleh keduanya (mis. orphan
    tunggal yang diapit tabel di kedua sisi) dibiarkan untuk fallback
    filter min_chunk_chars di build_chunk_records."""
    n_merged = 0

    # Lintasan 1: mundur.
    backward: list[dict[str, Any]] = []
    for ch in raw_chunks:
        can_merge = (
            not ch.get("is_table")
            and len(ch["text"]) < min_chunk_chars
            and not ch.get("is_first_after_hard_break")
            and backward
            and not backward[-1].get("is_table")
        )
        if can_merge and len(backward[-1]["text"]) + len(_PARA_SEP) + len(ch["text"]) <= soft_max:
            backward[-1] = _merge_two(backward[-1], ch)
            n_merged += 1
            continue
        backward.append(ch)

    # Lintasan 2: maju, untuk orphan yang tersisa.
    forward: list[dict[str, Any]] = []
    i = 0
    while i < len(backward):
        ch = backward[i]
        nxt = backward[i + 1] if i + 1 < len(backward) else None
        can_merge = (
            not ch.get("is_table")
            and len(ch["text"]) < min_chunk_chars
            and nxt is not None
            and not nxt.get("is_table")
            and not nxt.get("is_first_after_hard_break")
            and len(ch["text"]) + len(_PARA_SEP) + len(nxt["text"]) <= soft_max
        )
        if can_merge:
            forward.append(_merge_two(ch, nxt))
            n_merged += 1
            i += 2
            continue
        forward.append(ch)
        i += 1

    # Lintasan 3: orphan yang masih tersisa (terjepit tabel di kedua sisi
    # — pola paling umum di korpus ini: caption/pengantar tabel pendek
    # tepat sebelum tabel berikutnya) dilebur ke tabel BERIKUTNYA sebagai
    # preamble. _extract_units menjamin seluruh narasi satu halaman selalu
    # mendahului tabel-tabel halaman itu, jadi paragraf pendek tepat
    # sebelum sebuah tabel hampir selalu caption/pengantar tabel itu,
    # bukan derau lepas — membuangnya kehilangan sinyal retrieval yang
    # berguna (mis. "Tabel 2.2. Tahapan Pengusulan ..."). Tidak ada guard
    # soft_max di sini: tabel sudah dikecualikan dari batas ukuran apa pun.
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(forward):
        ch = forward[i]
        nxt = forward[i + 1] if i + 1 < len(forward) else None
        can_merge = (
            not ch.get("is_table")
            and len(ch["text"]) < min_chunk_chars
            and nxt is not None
            and nxt.get("is_table")
            and not nxt.get("is_first_after_hard_break")
        )
        if can_merge:
            result.append(
                {
                    "text": f"{ch['text']}{_PARA_SEP}{nxt['text']}",
                    "page_start": min(ch["page_start"], nxt["page_start"]),
                    "page_end": max(ch["page_end"], nxt["page_end"]),
                    "section_heading": ch["section_heading"] or nxt["section_heading"],
                    "is_table": True,
                    "is_first_after_hard_break": ch.get("is_first_after_hard_break", False),
                }
            )
            n_merged += 1
            i += 2
            continue
        result.append(ch)
        i += 1

    return result, n_merged


def build_chunk_records(
    doc_id: str,
    pages: list[dict[str, Any]],
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int = 250,
    max_dot_ratio: float = 0.20,
    soft_max: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Gabungkan clean_text + deteksi heading + recursive_chunk per dokumen
    menjadi baris sesuai skema §5.2.

    soft_max (default 1.3x chunk_size bila None) adalah toleransi luapan:
    paragraf sepanjang (chunk_size, soft_max] jadi satu chunk sendiri tanpa
    dipaksa lewat recursive_chunk — lihat catatan di _units_to_pieces.

    chunk_id berformat f"{doc_id}::p{page_start:02d}::c{seq:03d}";
    chunk_strategy disertakan di setiap record agar indeks dari strategi
    chunking berbeda tidak pernah tertukar (P2). Chunk "yatim" (< n_chars
    min_chunk_chars) dilebur ke chunk sebelumnya (_merge_orphan_chunks)
    bila aman dilakukan; sisanya yang tidak bisa dilebur (mis. chunk
    pertama dokumen/BAB) dibuang sebagai fallback. Chunk dengan rasio
    titik > max_dot_ratio juga dibuang (dot leader daftar isi). Kedua
    filter ini TIDAK berlaku untuk chunk tabel (is_table=True): tabel
    kecil (mis. 2-3 baris) tetap konten sah, dan sel tabel numerik/kode
    dokumen bisa wajar mengandung banyak titik.

    Return: (records, stats) — stats = {"n_raw_chunks", "n_kept",
    "n_merged_orphan", "n_dropped_short", "n_dropped_dot_noise"}.
    """
    soft_max = soft_max if soft_max is not None else round(chunk_size * 1.3)
    repeated_lines = _detect_repeated_lines(pages)
    units = _extract_units(pages, repeated_lines)
    pieces = _units_to_pieces(units, chunk_size, chunk_overlap, soft_max)
    raw_chunks = _pack_pieces(pieces, chunk_size, chunk_overlap)
    raw_chunks, n_merged_orphan = _merge_orphan_chunks(raw_chunks, min_chunk_chars, soft_max)

    records: list[dict[str, Any]] = []
    seq_by_page: dict[int, int] = {}
    n_dropped_short = 0
    n_dropped_dot_noise = 0

    for ch in raw_chunks:
        text = ch["text"]
        is_table = ch.get("is_table", False)
        if not is_table and len(text) < min_chunk_chars:
            n_dropped_short += 1
            continue
        if not is_table and _dot_ratio(text) > max_dot_ratio:
            n_dropped_dot_noise += 1
            continue
        page_start = ch["page_start"]
        seq = seq_by_page.get(page_start, 0) + 1
        seq_by_page[page_start] = seq
        records.append(
            {
                "chunk_id": f"{doc_id}::p{page_start:02d}::c{seq:03d}",
                "doc_id": doc_id,
                "page_start": page_start,
                "page_end": ch["page_end"],
                "section_heading": ch["section_heading"],
                "text": text,
                "n_chars": len(text),
                "n_tokens": _approx_token_count(text),
                "chunk_strategy": chunk_strategy,
                "is_table": is_table,
            }
        )

    stats = {
        "n_raw_chunks": len(raw_chunks),
        "n_kept": len(records),
        "n_merged_orphan": n_merged_orphan,
        "n_dropped_short": n_dropped_short,
        "n_dropped_dot_noise": n_dropped_dot_noise,
    }
    return records, stats


def write_chunks_jsonl(records: list[dict[str, Any]], out_path: str | Path) -> None:
    """Tulis daftar record chunk ke JSONL, satu objek per baris."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
