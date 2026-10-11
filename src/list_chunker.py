"""Memotong satu bagian dokumen tanpa memutus daftar di tengah butir.

chunk_section(text, max_chars=2000) -> list[str]
  - satuan potong = satu butir tingkat teratas beserta sub-butirnya (mis. "w." + "1." s.d. "5.")
  - potongan lanjutan diawali kalimat pengantar daftarnya, bukan overlap karakter
merge_intro(sections) -> sections
  - bagian pendek yang hanya berisi pengantar ("... sebagai berikut :") digabung ke bagian berikutnya
"""
import re

MARKERS = [
    ("alpha.", re.compile(r"^[a-z]\.\s")),
    ("num.", re.compile(r"^\d{1,2}\.\s")),
    ("num)", re.compile(r"^\d{1,2}\)\s")),
    ("alpha)", re.compile(r"^[a-z]\)\s")),
    ("bullet", re.compile(r"^[•●▪\-–]\s")),
]


def _kind(line):
    for name, rx in MARKERS:
        if rx.match(line):
            return name
    return None


def _units(text):
    """Pisahkan teks menjadi pengantar + daftar butir tingkat teratas (tiap butir = list baris)."""
    preamble, units, levels = [], [], []
    for line in (l.strip() for l in text.split("\n")):
        if not line:
            continue
        k = _kind(line)
        if k is None:
            (units[-1] if units else preamble).append(line)
            continue
        if k in levels:
            levels = levels[: levels.index(k) + 1]
        else:
            levels.append(k)
        if len(levels) == 1:
            units.append([line])
        else:
            units[-1].append(line)
    return preamble, units


def _lead_in(preamble):
    """Kalimat terakhir pengantar bila berakhir dengan titik dua (kalimat pembuka daftar)."""
    if not preamble:
        return ""
    last = re.split(r"(?<=[.!?])\s+", preamble[-1].strip())[-1]
    return last if last.rstrip().endswith(":") else ""


def _pack(blocks, first_head, carry, max_chars):
    """Kemas blok (tiap blok = list baris) secara berurutan tanpa memecah satu blok."""
    chunks, cur = [], list(first_head)
    used = sum(len(x) + 1 for x in cur)
    has_block = False
    for b in blocks:
        size = sum(len(x) + 1 for x in b)
        if has_block and used + size > max_chars:
            chunks.append("\n".join(cur))
            cur = [f"(lanjutan) {carry}"] if carry else []
            used = sum(len(x) + 1 for x in cur)
        cur += b
        used += size
        has_block = True
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_section(text, max_chars=2000):
    preamble, units = _units(text)
    if not units:
        return [text.strip()]
    carry = _lead_in(preamble)
    blocks = []
    for u in units:
        if sum(len(x) + 1 for x in u) <= max_chars or len(u) == 1:
            blocks.append(u)
        else:                                   # satu butir terlalu panjang: pecah di sub-butirnya
            parts = _pack([[x] for x in u[1:]], [u[0]], u[0], max_chars)
            blocks += [p.split("\n") for p in parts]
    return _pack(blocks, preamble, carry, max_chars)


def merge_intro(sections, max_intro=500):
    """sections: list[(heading, text)]. Pengantar pendek berakhiran ':' digabung ke bagian berikutnya."""
    out, pending = [], ""
    for heading, text in sections:
        if pending:
            text, pending = pending + "\n" + text, ""
        t = text.strip()
        if len(t) <= max_intro and t.endswith(":") and not _units(t)[1]:
            pending = t
            continue
        out.append((heading, text))
    if pending:
        out.append((sections[-1][0], pending))
    return out


def merge_siblings(sections, max_chars=2000, min_chars=250, sep=" > "):
    """Gabungkan bagian-bagian pendek yang bersaudara (induk judul sama) sampai batas ukuran.

    sections: list[(heading_path, text)] berurutan. Mengembalikan list[(heading_path, text)].
    - bagian digabung hanya dengan saudaranya yang bersebelahan, tidak pernah lintas induk
    - sisa di akhir kelompok yang masih < min_chars ditempelkan ke potongan sebelumnya
    - hasil gabungan memakai judul induk; judul tiap butir tetap ada di baris pertama teksnya
    """
    def parent(h):
        return h.rsplit(sep, 1)[0] if sep in h else h

    out, group = [], []          # group: list[(heading, text)] dengan induk yang sama

    def flush():
        if not group:
            return
        if len(group) == 1:
            out.append(group[0])
        else:
            packed, cur = [], []
            for _, t in group:
                if cur and sum(len(x) + 2 for x in cur) + len(t) > max_chars:
                    packed.append(cur)
                    cur = []
                cur.append(t)
            packed.append(cur)
            if len(packed) > 1 and sum(len(x) + 2 for x in packed[-1]) < min_chars:
                packed[-2] += packed.pop()
            out.extend((parent(group[0][0]), "\n\n".join(p)) for p in packed)
        group.clear()

    for h, t in sections:
        if group and parent(h) != parent(group[0][0]):
            flush()
        if len(t) > max_chars:   # bagian besar berdiri sendiri (dipotong oleh chunk_section)
            flush()
            out.append((h, t))
            continue
        group.append((h, t))
    flush()
    return out
