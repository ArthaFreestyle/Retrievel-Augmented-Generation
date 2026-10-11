"""Tabel B-I (Lampiran 2): satu baris tabel = satu blok teks pendek.

Semua tabel ini berkolom sama (Kategori, Jenis, Status, Bukti Luaran, Target,
Kriteria, Isian, Bukti Pendukung, Tidak Valid) dan dikenali dari judul kolom
"Transaksi luaran pada proposal". Judul kolomnya bertingkat dan sel
"Kategori"/"Jenis" menyatu vertikal, sehingga find_tables()+table_to_text()
menghasilkan nama kolom berantai yang diulang di setiap baris. Di sini teks
dibaca per pita horizontal (antar garis tabel) dan per kolom (batas x diambil
dari garis tabel), lalu tiap baris ditulis dengan nama kolom pendek.

Baris yang terpotong batas halaman: sisanya muncul sebagai tabel tanpa judul
kolom di puncak halaman berikutnya. Potongan itu dibaca dengan batas kolomnya
sendiri (atau batas tabel sebelumnya) lalu ditempelkan sel demi sel ke baris
terakhir.

Dipakai oleh src/ingest.py::extract_text_auto.
"""
from __future__ import annotations

import re
from typing import Any

DOC_TITLE = "Panduan Usulan Pengmas Internal UNAIR 2026 > Lampiran 2"
_KEYS = ["kategori", "jenis", "status", "bukti", "target", "kriteria", "isian", "pendukung", "tidak_valid"]
_BULLET_RE = re.compile(r"[•●]")
_CAPTION_RE = re.compile(r"^Tabel\s+[A-Z]\.\s")
_CLUSTER_GAP = 12  # garis tabel yang berjarak <= 12 pt dianggap satu batas kolom


def table_key(page_no: int, bbox) -> tuple:
    return (page_no, *(round(x) for x in bbox))


def _bounds(table) -> list[float]:
    """Batas kolom logis dari garis vertikal tabel: garis berdekatan dilebur jadi satu."""
    xs = sorted({round(c[0]) for c in table.cells} | {round(c[2]) for c in table.cells})
    groups: list[list[int]] = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= _CLUSTER_GAP:
            groups[-1].append(x)
        else:
            groups.append([x])
    b = [sum(g) / len(g) for g in groups]
    b[0] = float(groups[0][0])
    return b


def _centang_text(page, table) -> str:
    """Tabel dengan tanda centang berupa gambar (Tabel A): sel berisi gambar = "ada",
    sel berisi "-" = "tidak ada". Kosong bila bukan tabel semacam itu."""
    rows = table.extract()
    if not rows or (rows[0][0] or "").strip().rstrip(".") != "No":
        return ""
    schemes = {k: (c or "").strip() for k, c in enumerate(rows[0]) if (c or "").strip().startswith("Skema")}
    if not schemes:
        return ""

    def has_image(cell) -> bool:
        x0, top, x1, bottom = cell
        return any(
            x0 <= (im["x0"] + im["x1"]) / 2 <= x1 and top <= (im["top"] + im["bottom"]) / 2 <= bottom
            for im in page.images
        )

    lines = []
    for r, row in zip(table.rows[1:], rows[1:]):
        label = re.sub(r"\s+", " ", (row[1] or "")).strip()
        parts = []
        for k, name in schemes.items():
            cell, text = r.cells[k], (row[k] or "").strip()
            if text == "-":
                parts.append(f"{name}: tidak ada (-)")
            elif cell and has_image(cell):
                parts.append(f"{name}: ada (centang)")
            else:
                parts.append(f"{name}: {text or 'tidak terbaca'}")
        lines.append(f"{(row[0] or '').strip()}. {label} -- " + "; ".join(parts))
    if not lines:
        return ""
    head = f"{DOC_TITLE} > {_caption(page, table)}".rstrip(" >")
    return "\n".join([head, "Tanda centang berarti luaran ada pada skema itu; tanda - berarti tidak ada.", *lines])


def _is_header_table(page, table) -> bool:
    txt = re.sub(r"\s+", " ", page.crop(table.bbox).extract_text() or "")
    return "Transaksi" in txt and ("Kelayakan" in txt or "Capaian" in txt)


def _caption(page, table) -> str:
    """Judul "Tabel X. ..." tepat di atas tabel (baris berawalan Tabel + lanjutan wrap-nya)."""
    top = table.bbox[1]
    if top < 5:
        return ""
    txt = page.crop((0, 0, page.width, top)).extract_text() or ""
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    for k in range(len(lines) - 1, -1, -1):
        if _CAPTION_RE.match(lines[k]):
            return " ".join(lines[k : k + 3])
    return ""


def _header_bottom(page, table) -> float:
    ws = [
        w for w in page.crop(table.bbox).extract_words()
        if w["text"] in ("Capaian", "Kelayakan")
    ]
    return max(w["bottom"] for w in ws) + 1 if ws else table.bbox[1]


def _read_bands(page, table, bounds, skip_until: float) -> list[dict[str, list[str]]]:
    ys = sorted({round(c[1]) for c in table.cells} | {round(c[3]) for c in table.cells})
    bands = []
    for a, b in zip(ys, ys[1:]):
        if b - a < 5 or a < skip_until:
            continue
        # baris teks yang melintasi garis pita ikut terbaca di dua pita bila hanya di-crop;
        # tiap huruf dimiliki pita yang memuat titik tengahnya.
        owned = page.filter(lambda o, a=a, b=b: o.get("object_type") != "char" or a <= (o["top"] + o["bottom"]) / 2 < b)
        cells = {}
        for k, key in enumerate(_KEYS):
            txt = owned.crop((bounds[k], a, bounds[k + 1], b), strict=False).extract_text() or ""
            cells[key] = [ln.strip() for ln in txt.split("\n") if ln.strip()]
        bands.append(cells)
    return bands


def extract_all(pdf) -> tuple[dict[tuple, list[str]], list[str]]:
    """Baca semua tabel luaran di dokumen.

    Return: ({kunci tabel (halaman, bbox) -> [teks baris yang DIMULAI di tabel itu]},
    catatan). Kunci tabel potongan lanjutan juga ada (nilai [])."""
    rows: list[dict[str, Any]] = []
    owner: dict[tuple, list[dict[str, Any]]] = {}
    notes: list[str] = []
    state: dict[str, Any] | None = None

    for i, page in enumerate(pdf.pages, start=1):
        for t in page.find_tables():
            key = table_key(i, t.bbox)
            bounds = _bounds(t)
            centang = None if _is_header_table(page, t) else _centang_text(page, t)
            if centang:
                owner[key] = [{"text": centang}]
                continue
            if _is_header_table(page, t):
                caption = _caption(page, t)
                head_txt = re.sub(r"\s+", " ", page.crop(t.bbox).extract_text() or "")
                if caption or state is None or i != state["page"] + 1:
                    if not caption:
                        notes.append(f"hal. {i}: judul tabel luaran tidak ditemukan")
                    state = {"caption": caption, "page": i}
                # tanpa judul baru di halaman berikutnya = tabel yang sama dengan judul kolom diulang
                state["isian"] = "pelaksana" if "Pelaksana" in head_txt else "peneliti"
                state["bounds"] = bounds
                skip = _header_bottom(page, t)
            elif state and i == state["page"] + 1 and t.bbox[1] < page.height * 0.2:
                skip = 0.0
                state["page"] = i
            else:
                continue
            if len(bounds) != 10:
                if len(state["bounds"]) != 10:
                    notes.append(f"hal. {i}: batas kolom tabel luaran tidak terbaca")
                    continue
                bounds = state["bounds"]
                notes.append(f"hal. {i}: batas kolom dari tabel sebelumnya")
            else:
                state["bounds"] = bounds
            state["page"] = i
            owner[key] = []
            for cells in _read_bands(page, t, bounds, skip):
                prev = rows[-1] if rows and rows[-1]["state"] is state else None
                if cells["status"] or prev is None:
                    row = {"state": state, "key": key, **{k: list(v) for k, v in cells.items()}}
                    if prev is not None:  # sel Kategori/Jenis yang menyatu vertikal
                        for k in ("kategori", "jenis"):
                            if not row[k]:
                                row[k] = list(prev[k])
                    rows.append(row)
                    owner[key].append(row)
                else:
                    for k, v in cells.items():
                        prev[k] += v
    return {k: [r["text"] if "text" in r else row_to_text(r) for r in v] for k, v in owner.items()}, notes


def _items(lines: list[str]) -> list[str]:
    flat = " ".join(lines)
    return [re.sub(r"\s+", " ", x).strip() for x in _BULLET_RE.split(flat) if x.strip()]


def row_to_text(row: dict[str, Any]) -> str:
    def one(key: str) -> str:
        return re.sub(r"\s+", " ", " ".join(row[key])).strip()

    st = row["state"]
    lines = [f"{DOC_TITLE} > {st['caption']}", ""]
    if one("kategori"):
        lines.append(f"Kategori: {one('kategori')}")
    lines += [
        f"Luaran: {one('jenis')}",
        f"Status target: {one('status')}",
        f"Bukti luaran: {one('bukti')}",
    ]
    kriteria = "; ".join(_items(row["kriteria"]))
    if kriteria and kriteria.lower() != one("jenis").lower():
        lines.append(f"Kriteria validasi: {kriteria}")
    if row["isian"]:
        lines.append(f"Isian {st['isian']}: " + ", ".join(_items(row["isian"])))
    if row["pendukung"]:
        lines.append("Bukti pendukung: " + "; ".join(_items(row["pendukung"])))
    if row["tidak_valid"]:
        lines.append("Tidak valid jika: " + ", atau ".join(_items(row["tidak_valid"])))
    return "\n".join(lines)
