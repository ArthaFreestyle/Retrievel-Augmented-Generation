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
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:  # dipakai sebagai paket (src.chunking) maupun modul biasa (notebook: sys.path "../src")
    from src.form_detect import is_form, ringkas_form
    from src.list_chunker import _kind as _list_kind, chunk_section, merge_siblings
except ImportError:
    from form_detect import is_form, ringkas_form
    from list_chunker import _kind as _list_kind, chunk_section, merge_siblings

_PAGE_MARKER_RE = re.compile(r"^=== HALAMAN (\d+) ===$", re.MULTILINE)
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$|^\s*\d{1,4}\s*$", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Format HARUS sinkron dengan src/ingest.py::write_interim_text.
_TABLE_BLOCK_RE = re.compile(r"=== TABEL HALAMAN \d+ #\d+ ===\n(.*?)\n=== AKHIR TABEL ===", re.DOTALL)
_TABLE_SPLIT_RE = re.compile(r"(=== TABEL HALAMAN \d+ #\d+ ===\n.*?\n=== AKHIR TABEL ===)", re.DOTALL)
# HARUS sinkron dengan src/ingest.py::BOLD_MARK.
BOLD_MARK = "[[B]] "

_HEADING_PATTERNS = [
    re.compile(r"^BAB\s+[IVXLCDM\d]+\b.*$", re.IGNORECASE),  # "BAB 2", "BAB II ..."
    re.compile(r"^(DAFTAR\s+LAMPIRAN|LAMPIRAN\s+\d+)\b.*$", re.IGNORECASE),  # "Lampiran 2." (bukan "Lampiran II" di badan teks)
    re.compile(r"^Tabel\s+[A-Z]\.\s+\S.*$"),                 # "Tabel I. Luaran berupa ..."
    re.compile(r"^[A-Z]\.\s+\S.*$"),                          # "A. Persyaratan Peserta"
    re.compile(r"^\d+(\.\d+)*\.?\s+\S.*$"),                   # "2.1. Pendahuluan", "1. ..."
]
# Penanda butir daftar: "a. ", "1. ", "2) ", "- ", "• ". Baris berawalan ini
# TIDAK digabung ke baris sebelumnya (lihat _extract_units).
_LIST_ITEM_RE = re.compile(r"^(?:[A-Za-z]|\d{1,2})[.)]\s+\S|^[-\u2022\u2013]\s+\S")
_QUOTE_MAP = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " "})


def _normalize_chars(line: str) -> str:
    """NFKC + kutip lengkung -> lurus. Karakter pengganti U+FFFD (PDF ini
    menulis ’ dan “ ” sebagai kotak hitam) diperbaiki: di antara dua huruf
    jadi apostrof ("SDG\ufffds"), selain itu jadi tanda kutip ganda."""
    line = unicodedata.normalize("NFKC", line).translate(_QUOTE_MAP)
    line = re.sub(r"(?<=\w)\ufffd(?=\w)", "'", line)
    return line.replace("\ufffd", '"')


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
        lines = {_normalize_chars(_split_marker(ln)[1]) for ln in narrative.splitlines() if ln.strip()}
        counts.update(lines)
    n_pages = max(len(pages), 1)
    return {line for line, c in counts.items() if c / n_pages >= min_ratio}


def clean_text(raw_text: str, repeated_lines: set[str] | None = None) -> str:
    """Buang header/footer berulang & baris nomor halaman berdiri sendiri,
    normalisasi karakter (NFKC, kutip lengkung) & whitespace. Awalan
    BOLD_MARK dipertahankan."""
    repeated_lines = repeated_lines or set()
    lines = []
    for ln in raw_text.splitlines():
        bold, stripped = _split_marker(ln)
        stripped = _normalize_chars(stripped).strip()
        if not stripped:
            lines.append("")
            continue
        if stripped in repeated_lines or _PAGE_NUMBER_LINE_RE.match(stripped):
            continue
        lines.append((BOLD_MARK if bold else "") + re.sub(r"[ \t]+", " ", stripped))
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
    stripped = _split_marker(line)[1]
    # judul tabel ("Tabel E. Luaran berupa ...") bisa panjang; judul lain > 15 kata bukan heading
    if not stripped or len(stripped.split()) > (30 if stripped.startswith("Tabel ") else 15):
        return None
    if _DOT_LEADER_RE.search(stripped):
        return None
    for pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return stripped
    return None


# Judul lampiran/tabel ("Tabel B. ...") di PDF ini kadang tidak tercetak tebal ("Lampiran 3. Format ...")
# sehingga tidak lolos syarat BOLD_MARK; dikenali dari pola + posisi (lihat _extract_units).
_LAMPIRAN_TITLE_RE = re.compile(r"^(?:(?i:LAMPIRAN)\s+\d+|Tabel\s+[A-Z])\.\s+\S")


def _heading_level(line: str) -> int:
    """0 = BAB / Lampiran (mereset jalur); 1 = "2.4."; 2 = "2.4.1." dst.
    -1 = butir bernomor/berhuruf ("4. Tahap ...", "A. ..."): levelnya
    ditentukan relatif terhadap jalur aktif di _HeadingPath.push."""
    if re.match(r"^(BAB\s+[IVXLCDM\d]+\b|DAFTAR\s+LAMPIRAN|LAMPIRAN\s+\d+)", line, re.IGNORECASE):
        return 0
    m = re.match(r"^(\d+(?:\.\d+)+)", line)
    if m:
        return m.group(1).count(".")
    return -1


def _looks_like_heading_continuation(line: str) -> bool:
    """Fallback untuk dokumen tanpa informasi huruf tebal (mis. HTML):
    baris lanjutan judul BAB ditulis ALL CAPS / Title Case."""
    stripped = line.strip()
    words = stripped.split()
    if not stripped or len(words) > 12:
        return False
    if stripped.isupper():
        return True
    capitalized = sum(1 for w in words if w[:1].isupper())
    return capitalized >= max(1, len(words) - 1)


class _HeadingPath:
    """Tumpukan judul aktif -> jalur seperti
    "BAB 2 ... > 2.4. Tahapan Kegiatan > 4. Tahap Penyeleksian"."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str, bool]] = []  # (level, teks, butir_bernomor)

    def push(self, text: str) -> str | None:
        level = _heading_level(text)
        is_item = level == -1
        if is_item:
            # Butir bernomor/berhuruf: satu tingkat di bawah judul terdekat
            # yang BUKAN butir, sehingga "1.", "2.", "3." bersaudara.
            parent = next((lv for lv, _, item in reversed(self.stack) if not item), 0 if not self.stack else -1)
            level = parent + 1 if self.stack else 1
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text, is_item))
        return self.path

    @property
    def path(self) -> str | None:
        return " > ".join(t for _, t, _ in self.stack) or None


def _split_marker(line: str) -> tuple[bool, str]:
    if line.startswith(BOLD_MARK):
        return True, line[len(BOLD_MARK):].strip()
    return False, line.strip()


def _word_boundary_tail(text: str, max_chars: int) -> str:
    """Ambil akhiran `text` sepanjang <= max_chars TANPA memenggal kata.
    Hanya dipakai sebagai jalan terakhir (pemotongan per kata di
    recursive_chunk); overlap normal memakai _overlap_tail."""
    if max_chars <= 0 or not text:
        return ""
    if max_chars >= len(text):
        return text.strip()
    tail = text[-max_chars:]
    if not text[-max_chars - 1].isspace():
        m = re.search(r"\s", tail)
        if m is None:
            return ""
        tail = tail[m.end():]
    return tail.strip()


def _overlap_tail(text: str, budget: int) -> str:
    """Overlap yang selalu mulai di awal unit: baris utuh dari belakang
    (butir daftar = satu baris, lihat _extract_units); bila baris terakhir
    saja melebihi anggaran, kalimat utuh dari belakang. Jika tak ada yang
    muat -> "" (lebih baik tanpa overlap daripada overlap yang mulai di
    tengah kalimat)."""
    text = text.strip()
    if budget <= 0 or not text:
        return ""
    lines = text.split("\n")
    picked: list[str] = []
    total = 0
    for ln in reversed(lines):
        extra = len(ln) + (1 if picked else 0)
        if total + extra > budget:
            break
        picked.insert(0, ln)
        total += extra
    if picked:
        return "\n".join(picked)
    sents = [x for x in _SENTENCE_SPLIT_RE.split(lines[-1]) if x.strip()]
    for sent in reversed(sents):
        extra = len(sent) + (1 if picked else 0)
        if total + extra > budget:
            break
        picked.insert(0, sent)
        total += extra
    return " ".join(picked)


_SPLIT_LEVELS = ["\n\n", "\n", None, " "]  # None = batas kalimat


def _split_at(text: str, level: str | None) -> list[str]:
    if level is None:
        return [x for x in _SENTENCE_SPLIT_RE.split(text) if x.strip()]
    return [x for x in text.split(level) if x.strip()]


def recursive_chunk(text: str, chunk_size: int = 800, chunk_overlap: int = 120, _depth: int = 0) -> list[str]:
    """Pecah SATU teks menjadi potongan <= chunk_size dengan memotong di
    batas terbesar yang tersedia: paragraf -> baris (mis. butir daftar)
    -> kalimat -> kata; per karakter hanya bila satu kata melebihi
    chunk_size. Potongan bertetangga di-overlap oleh _overlap_tail
    (mulai di awal baris/kalimat; tingkat kata memakai _word_boundary_tail)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if _depth >= len(_SPLIT_LEVELS):
        step = max(chunk_size - chunk_overlap, 1)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]

    level = _SPLIT_LEVELS[_depth]
    parts = _split_at(text, level)
    if len(parts) <= 1:
        return recursive_chunk(text, chunk_size, chunk_overlap, _depth + 1)
    joiner = level if level is not None else " "

    pieces: list[str] = []
    buf = ""
    for part in parts:
        if len(part) > chunk_size:  # satu bagian saja kebesaran -> turun satu tingkat
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(recursive_chunk(part, chunk_size, chunk_overlap, _depth + 1))
            continue
        candidate = f"{buf}{joiner}{part}" if buf else part
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        pieces.append(buf)
        tail = _word_boundary_tail(buf, chunk_overlap) if level == " " else _overlap_tail(buf, chunk_overlap)
        buf = f"{tail}{joiner}{part}" if tail and len(tail) + len(joiner) + len(part) <= chunk_size else part
    if buf:
        pieces.append(buf)
    return pieces


def _merge_across_pages(units: list[dict[str, Any]]) -> None:
    """Sambung paragraf yang terpotong di batas halaman (in-place):
    paragraf pertama sebuah halaman yang diawali huruf kecil, ketika
    paragraf terakhir halaman sebelumnya belum berakhir dengan tanda baca
    penutup, adalah lanjutan kalimat yang sama."""
    out: list[dict[str, Any]] = []
    for u in units:
        prev = out[-1] if out else None
        if (
            prev
            and prev["type"] == "para"
            and u["type"] == "para"
            and u.get("page_first_unit")
            and u["page"] == prev.get("page_end", prev["page"]) + 1
            and u["heading"] == prev["heading"]
            and not prev["text"].rstrip().endswith((".", "!", "?", ":", ";", ")", '"'))
            and u["text"][:1].islower()
        ):
            first, _, rest = u["text"].partition("\n")
            prev["text"] = f"{prev['text']} {first}" + (f"\n{rest}" if rest else "")
            prev["page_end"] = u["page"]
            continue
        out.append(u)
    units[:] = out


_CAPTION_IN_PARA_RE = re.compile(r"(?:^|(?<=\s))Tabel\s+(?:[A-Z]|\d+(?:\.\d+)*)\.\s+(?=\S)")


def _split_trailing_caption(text: str) -> tuple[str, str]:
    """Judul tabel ("Tabel 2.1. ...", "Tabel D. ...") yang tertempel di ujung paragraf
    tepat sebelum tabelnya: kembalikan (paragraf_tanpa_judul, judul). Judul kosong bila tak ada."""
    matches = list(_CAPTION_IN_PARA_RE.finditer(text))
    if not matches:
        return text, ""
    start = matches[-1].start()
    caption = text[start:].strip()
    if len(caption.split()) > 30 or caption.endswith("."):  # kalimat biasa yang kebetulan memuat "Tabel A."
        return text, ""
    return text[:start].rstrip(), re.sub(r"\s+", " ", caption)


def _extract_units(pages: list[dict[str, Any]], repeated_lines: set[str]) -> list[dict[str, Any]]:
    """Ubah halaman jadi urutan unit heading/paragraf/tabel. Judul dikenali
    dari BOLD_MARK + pola nomor (butir daftar tidak tebal -> bukan judul);
    dokumen tanpa informasi tebal (HTML) memakai pola saja, hanya di awal
    blok. Judul TETAP ada di teks (dilekatkan ke paragraf pertama di
    bawahnya oleh _units_to_pieces) dan memperbarui jalur judul.

    Di dalam paragraf, baris berawalan penanda butir ("a.", "1)") tidak
    digabung ke baris sebelumnya melainkan dipisah "\\n", supaya splitter
    memotong di batas butir.

    Teks halaman = prosa dan blok tabel berselang-seling sesuai urutan aslinya
    (ingest.extract_text_auto menyisipkan tabel di posisinya), sehingga tabel
    mewarisi judul yang benar; interim lama yang menaruh tabel di akhir
    halaman tetap terbaca."""
    units: list[dict[str, Any]] = []
    path = _HeadingPath()
    has_bold = any(BOLD_MARK in p["text"] for p in pages)

    for p in pages:
        first_on_page = True
        para: list[str] = []

        def flush() -> None:
            nonlocal first_on_page
            if not para:
                return
            units.append(
                {
                    "type": "para",
                    "text": "\n".join(para),
                    "page": p["page"],
                    "heading": path.path,
                    "page_first_unit": first_on_page,
                }
            )
            first_on_page = False
            para.clear()

        for seg_idx, seg in enumerate(_TABLE_SPLIT_RE.split(p["text"])):
            if seg_idx % 2 == 1:
                m = _TABLE_BLOCK_RE.search(seg)
                prev = units[-1] if units else None
                if m and prev and prev["type"] == "para" and prev["page"] == p["page"]:
                    rest, caption = _split_trailing_caption(prev["text"])
                    if caption:
                        if rest:
                            prev["text"] = rest
                        else:
                            units.pop()
                        units.append({"type": "heading", "text": caption, "page": p["page"], "path": path.push(caption)})
                if m:
                    units.append({"type": "table", "text": _normalize_chars(m.group(1).strip()), "page": p["page"], "heading": path.path})
                    first_on_page = False
                continue

            lines = clean_text(seg, repeated_lines).split("\n")
            i = 0
            prev_blank = True
            after_heading = False
            while i < len(lines):
                bold, body = _split_marker(lines[i])
                if not body:
                    flush()
                    prev_blank = True
                    i += 1
                    continue
                body = re.sub(r"^(\d+\.|[A-Z]\.)(?=[^\s\d.])", r"\1 ", body)
                is_heading = detect_section_heading(body) is not None and (bold if has_bold else prev_blank)
                if not is_heading and has_bold and not bold and (prev_blank or after_heading):
                    is_heading = bool(_LAMPIRAN_TITLE_RE.match(body)) and detect_section_heading(body) is not None
                after_heading = False
                if is_heading:
                    flush()
                    heading_lines = [body]
                    i += 1
                    is_bab = _heading_level(body) == 0
                    max_lines = 5 if is_bab else 3
                    while i < len(lines) and len(heading_lines) < max_lines:
                        nb, nbody = _split_marker(lines[i])
                        if not nbody or detect_section_heading(nbody):
                            break
                        # judul tabel yang tidak tebal dan terpotong baris: baris penuh lalu sisanya
                        wrapped_caption = (
                            body.startswith("Tabel ") and not bold and len(heading_lines[-1]) >= 70 and len(nbody.split()) <= 8
                        )
                        if has_bold:
                            # Lanjutan judul bertingkat: tebal, dan (selain BAB) pendek.
                            if not wrapped_caption and (not nb or (not is_bab and len(nbody.split()) > 5)):
                                break
                        elif not (is_bab and _looks_like_heading_continuation(nbody)):
                            break
                        heading_lines.append(nbody)
                        i += 1
                    heading_text = " ".join(heading_lines)
                    units.append(
                        {
                            "type": "heading",
                            "text": heading_text,
                            "page": p["page"],
                            "path": path.push(heading_text),
                        }
                    )
                    first_on_page = False
                    prev_blank = False
                    after_heading = True
                    continue
                # Baris prosa: butir daftar mulai baris baru, selebihnya disambung.
                if not para or _LIST_ITEM_RE.match(body):
                    para.append(body)
                else:
                    para[-1] = f"{para[-1]} {body}"
                prev_blank = False
                i += 1
            flush()

    # Judul tabel di ujung paragraf yang tabelnya tidak ada (dibuang saat ekstraksi, mis. bagan
    # arsiran) tidak berguna: buang juga judulnya.
    for k, u in enumerate(units):
        nxt = units[k + 1] if k + 1 < len(units) else None
        if u["type"] == "para" and (nxt is None or nxt["type"] != "table"):
            rest, caption = _split_trailing_caption(u["text"])
            if caption:
                u["text"] = rest
    units[:] = [u for u in units if u["type"] != "para" or u["text"].strip()]

    _merge_across_pages(units)
    return units


_FILL_LEADER_RE = re.compile(r"[ \t]*(?:[.…·_][ \t]?){4,}[ \t]*")


def tandai_isian(text: str) -> str:
    """Ganti deretan >= 4 titik/garis bawah/elipsis (tempat mengisi pada
    formulir) dengan "[diisi]", bukan dihapus: "Isian: ....... Orang" harus
    tetap terbaca sebagai tempat mengisi jumlah orang. Hanya aman setelah
    halaman daftar isi dibuang (build_chunk_records memanggilnya sesudah
    _drop_front_matter); "dst...", nomor bagian, dan angka rupiah tidak
    tersentuh."""
    return re.sub(r" {2,}", " ", _FILL_LEADER_RE.sub(" [diisi] ", text)).strip()


def _drop_front_matter(units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Buang semua unit sebelum judul BAB 1 pertama yang sungguhan (entri
    daftar isi sudah ditolak detect_section_heading lewat dot leader).
    Tanpa BAB 1, dokumen dibiarkan utuh."""
    for idx, u in enumerate(units):
        if u["type"] == "heading" and re.match(r"^BAB\s+(1|I)\b", u["text"], re.IGNORECASE):
            return units[idx:], idx
    return units, 0


def _group_list_runs(units: list[dict[str, Any]], chunk_size: int) -> list[dict[str, Any]]:
    """Paragraf berurutan di bawah judul yang sama yang memuat daftar butir dan
    total panjangnya > chunk_size digabung jadi SATU unit "list_run" (halaman
    sudah disambung), supaya dipotong per butir oleh list_chunker -- bukan
    mengikuti batas halaman/paragraf hasil ekstraksi."""
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(units):
        u = units[i]
        if u["type"] != "para":
            out.append(u)
            i += 1
            continue
        j = i
        while j + 1 < len(units) and units[j + 1]["type"] == "para" and units[j + 1]["heading"] == u["heading"]:
            j += 1
        run = units[i : j + 1]
        lines = [
            (ln.strip(), x["page"], x.get("page_end", x["page"]))
            for x in run
            for ln in x["text"].split("\n")
            if ln.strip()
        ]
        total = sum(len(ln) + 1 for ln, _, _ in lines)
        n_items = sum(1 for ln, _, _ in lines if _list_kind(ln))
        if total > chunk_size and n_items >= 3:
            out.append(
                {
                    "type": "para",
                    "text": "\n".join(ln for ln, _, _ in lines),
                    "page": run[0]["page"],
                    "page_end": run[-1].get("page_end", run[-1]["page"]),
                    "heading": u["heading"],
                    "list_run": True,
                    "line_pages": lines,
                }
            )
        else:
            out.extend(run)
        i = j + 1
    return out


def _list_run_pieces(u: dict[str, Any], prefix: str, starts_section: bool, chunk_size: int) -> list[dict[str, Any]]:
    """Potong list_run per butir; halaman tiap potongan dicari dari baris-barisnya."""
    parts = chunk_section(u["text"], max_chars=max(chunk_size - len(prefix) - 1, 200))
    lines, k, pieces = u["line_pages"], 0, []
    title = prefix.split("\n")[-1] if prefix else (u["heading"] or "").rsplit(" > ", 1)[-1]
    for n, part in enumerate(parts):
        pages = []
        for ln in part.split("\n"):
            if ln.startswith("(lanjutan) "):
                continue
            while k < len(lines) and lines[k][0] != ln:
                k += 1
            if k < len(lines):
                pages += [lines[k][1], lines[k][2]]
                k += 1
        if n > 0 and not part.startswith("(lanjutan) ") and title:
            part = f"(lanjutan) {title}\n{part}"  # daftar tanpa kalimat pengantar: pakai judul bagian
        pieces.append(
            {
                "text": f"{prefix}\n{part}" if (n == 0 and prefix) else part,
                "page": min(pages) if pages else u["page"],
                "page_end": max(pages) if pages else u.get("page_end", u["page"]),
                "heading": u["heading"],
                "hard_break_before": starts_section and n == 0,
                "standalone": True,  # daftar tidak di-overlap (menghindari butir muncul dua kali)
            }
        )
    return pieces


def _units_to_pieces(
    units: list[dict[str, Any]], chunk_size: int, chunk_overlap: int, soft_max: int
) -> list[dict[str, Any]]:
    """Pecah paragraf yang > soft_max lewat recursive_chunk(). SETIAP judul
    adalah batas keras: piece pertama sesudahnya ditandai
    hard_break_before. Teks judul (semua judul yang menumpuk sejak piece
    terakhir) dilekatkan di awal piece pertama itu, jadi judul tetap ada di
    teks chunk. Unit "table" SELALU jadi piece standalone, tidak pernah
    dipecah recursive_chunk ataupun digabung dengan paragraf lain.

    soft_max (> chunk_size, mis. 1.3x) adalah TOLERANSI: paragraf sepanjang
    (chunk_size, soft_max] tidak dipaksa dipecah -- keluar sebagai satu
    chunk sendiri. Hanya paragraf > soft_max yang dipecah."""
    pieces: list[dict[str, Any]] = []
    pending_headings: list[str] = []
    prev_table_heading: str | None = None

    for u in _group_list_runs(units, chunk_size):
        if u["type"] == "heading":
            if _heading_level(u["text"]) == 0:
                pending_headings = []  # bagian tanpa isi (mis. Lampiran 13) tidak ikut menempel ke bagian berikutnya
            pending_headings.append(u["text"])
            continue

        starts_section = bool(pending_headings)
        if u["type"] == "table":
            # tabel yang sudah memuat judulnya sendiri (mis. Tabel I per baris) tidak perlu didahului judul lagi
            pending_headings = [h for h in pending_headings if h not in u["text"]]
        prefix = "\n".join(pending_headings)
        pending_headings = []
        page_end = u.get("page_end", u["page"])
        text = f"{prefix}\n{u['text']}" if prefix else u["text"]
        if u["type"] == "table" and not prefix and u["heading"] and prev_table_heading == u["heading"]:
            # baris-baris tabel besar yang dipecah per baris: tiap baris membawa judul bagiannya
            if u["heading"].rsplit(" > ", 1)[-1] not in text:
                text = f"{u['heading']}\n{text}"
        prev_table_heading = u["heading"] if u["type"] == "table" else None

        if u.get("list_run"):
            pieces.extend(_list_run_pieces(u, prefix, starts_section, chunk_size))
            continue

        if u["type"] == "table":
            pieces.append(
                {
                    "text": text,
                    "page": u["page"],
                    "page_end": page_end,
                    "heading": u["heading"],
                    "hard_break_before": starts_section,
                    "standalone": True,
                    "is_table": True,
                }
            )
            continue

        oversized = len(text) > soft_max
        sub_texts = recursive_chunk(text, chunk_size, chunk_overlap) if oversized else [text]
        for i, sub in enumerate(sub_texts):
            pieces.append(
                {
                    "text": sub,
                    # Bahan overlap tanpa teks judul (judul bukan isi yang perlu diulang).
                    "overlap_text": u["text"] if (i == 0 and not oversized) else sub,
                    "page": u["page"],
                    "page_end": page_end,
                    "heading": u["heading"],
                    "hard_break_before": starts_section and i == 0,
                    # sub-piece dari recursive_chunk() sudah membawa overlap-nya
                    # sendiri -- jangan dipaketkan/di-overlap lagi oleh _pack_pieces.
                    "standalone": oversized,
                }
            )

    return pieces


_PARA_SEP = "\n\n"


def _pack_pieces(pieces: list[dict[str, Any]], chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    """Susun piece menjadi chunk <= chunk_size (batas lunak, lihat
    toleransi soft_max di _units_to_pieces). Piece digabung dengan "\\n\\n"
    supaya batas paragraf tetap terlihat. Setiap hard_break_before menutup
    chunk berjalan, jadi chunk tidak pernah melintasi judul.

    Overlap = baris/kalimat utuh terakhir dari chunk sebelumnya
    (_overlap_tail), tanpa teks judul; kosong bila tak ada yang muat.
    Flag internal "is_first_after_hard_break" menandai chunk pembuka bagian."""
    chunks: list[dict[str, Any]] = []
    buf: list[dict[str, Any]] = []
    pending_new_section = True

    def buf_text() -> str:
        return _PARA_SEP.join(p["text"] for p in buf)

    def finalize() -> None:
        nonlocal pending_new_section
        text = buf_text().strip()
        if not text:
            return
        chunks.append(
            {
                "text": text,
                "page_start": min(p["page"] for p in buf),
                "page_end": max(p.get("page_end", p["page"]) for p in buf),
                "section_heading": next((p["heading"] for p in buf if p["heading"]), None),
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
            finalize()
            buf = []
            chunks.append(
                {
                    "text": piece["text"].strip(),
                    "page_start": piece["page"],
                    "page_end": piece.get("page_end", piece["page"]),
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
            last = buf[-1]
            finalize()
            overlap_budget = min(chunk_overlap, max(chunk_size - piece_len - len(_PARA_SEP), 0))
            overlap_text = _overlap_tail(last.get("overlap_text", last["text"]), overlap_budget)
            buf = (
                [{"text": overlap_text, "page": last.get("page_end", last["page"]), "heading": last["heading"]}]
                if overlap_text
                else []
            )

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

    Semua penggabungan HANYA antar-chunk yang section_heading-nya sama --
    chunk tidak pernah melintasi judul, supaya jalur judul tetap benar.

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
            and backward
            and not backward[-1].get("is_table")
            and backward[-1]["section_heading"] == ch["section_heading"]
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
            and nxt["section_heading"] == ch["section_heading"]
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
            and nxt["section_heading"] == ch["section_heading"]
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


def _merge_sibling_chunks(chunks: list[dict[str, Any]], chunk_size: int, min_chunk_chars: int) -> list[dict[str, Any]]:
    """Gabungkan chunk pendek yang bersaudara (induk judul sama) sampai chunk_size
    (list_chunker.merge_siblings). Hanya chunk non-tabel dari bagian yang
    seluruhnya satu chunk yang ikut; tabel dan bagian yang terpecah jadi
    beberapa chunk bertindak sebagai pembatas. Judul hasil gabungan = judul induk."""
    heading_count = Counter(ch["section_heading"] for ch in chunks)
    out: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []

    def flush() -> None:
        if len(run) > 1:
            merged = merge_siblings(
                [(c["section_heading"] or "", c["text"]) for c in run], max_chars=chunk_size, min_chars=min_chunk_chars
            )
            k = 0
            for heading, text in merged:
                members, joined = [], None
                while k < len(run) and joined != text:
                    members.append(run[k])
                    k += 1
                    joined = "\n\n".join(c["text"] for c in members)
                if len(members) == 1:
                    out.append(members[0])
                    continue
                out.append(
                    {
                        **members[0],
                        "text": text,
                        "page_start": min(c["page_start"] for c in members),
                        "page_end": max(c["page_end"] for c in members),
                        "section_heading": heading or members[0]["section_heading"],
                    }
                )
        else:
            out.extend(run)
        run.clear()

    for ch in chunks:
        if ch.get("is_table") or not ch["section_heading"] or heading_count[ch["section_heading"]] > 1:
            flush()
            out.append(ch)
        else:
            run.append(ch)
    flush()
    return out


def _merge_intro_chunks(chunks: list[dict[str, Any]], max_intro: int = 500) -> list[dict[str, Any]]:
    """Chunk pembuka bagian yang cuma pengantar ("... sebagai berikut :") dilebur
    ke awal chunk sesudahnya (bagian berikutnya), bukan dibiarkan kosong."""
    out: list[dict[str, Any]] = []
    carry: dict[str, Any] | None = None
    for ch in chunks:
        if carry is not None:
            if ch.get("is_table"):
                out.append(carry)
            else:
                ch = {**ch, "text": f"{carry['text']}\n{ch['text']}", "page_start": carry["page_start"]}
            carry = None
        t = ch["text"].strip()
        if not ch.get("is_table") and len(t) <= max_intro and t.endswith(":"):
            carry = ch
            continue
        out.append(ch)
    if carry is not None:
        out.append(carry)
    return out


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
    units, n_front_matter_units = _drop_front_matter(units)
    for u in units:
        if u["type"] in ("para", "table"):
            u["text"] = "\n".join(tandai_isian(ln) for ln in u["text"].split("\n"))
    pieces = _units_to_pieces(units, chunk_size, chunk_overlap, soft_max)
    raw_chunks = _pack_pieces(pieces, chunk_size, chunk_overlap)
    raw_chunks = _merge_intro_chunks(raw_chunks)
    raw_chunks = _merge_sibling_chunks(raw_chunks, chunk_size, min_chunk_chars)
    raw_chunks, n_merged_orphan = _merge_orphan_chunks(raw_chunks, min_chunk_chars, soft_max)

    records: list[dict[str, Any]] = []
    seq_by_page: dict[int, int] = {}
    n_dropped_short = 0
    n_dropped_dot_noise = 0

    for ch in raw_chunks:
        text = ch["text"]
        is_table = ch.get("is_table", False)
        # Chunk pembuka bagian yang pendek (judul + satu-dua kalimat) sah dan
        # mandiri; yang dibuang hanya sisa pendek non-pembuka yang tak bisa dilebur.
        if not is_table and len(text) < min_chunk_chars and not ch.get("is_first_after_hard_break"):
            n_dropped_short += 1
            continue
        if not is_table and _dot_ratio(text) > max_dot_ratio:
            n_dropped_dot_noise += 1
            continue
        is_form_chunk = not is_table and is_form(text)
        if is_form_chunk:
            text = ringkas_form(text, ch["section_heading"] or "")
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
                "is_form": is_form_chunk,
            }
        )

    stats = {
        "n_front_matter_units_dropped": n_front_matter_units,
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
