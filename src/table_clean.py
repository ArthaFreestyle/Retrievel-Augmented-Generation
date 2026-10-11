"""Mengubah tabel hasil pdfplumber menjadi teks "judul kolom: nilai" tanpa garis '|'.

Pemakaian:
    rows = page.extract_table()                      # list[list[str | None]]
    teks, catatan = table_to_text([rows], caption="Tabel 2.2. ...")

Tabel yang bersambung ke halaman berikutnya dikirim sekaligus:
    teks, catatan = table_to_text([rows_hal_23, rows_hal_24, rows_hal_25], caption=...)

Jika `teks` kosong, tabel tidak punya isi (mis. bagan arsiran) dan sebaiknya dibuang.
Jika `catatan` tidak kosong, bandingkan hasilnya dengan PDF asli.
"""
import re


def _clean(rows):
    rows = [["\n".join(re.sub(r"[^\S\n]+", " ", x).strip() for x in (c or "").split("\n")).strip() for c in r]
            for r in rows]
    rows = [r for r in rows if any(r)]
    n = max(len(r) for r in rows)
    return [r + [""] * (n - len(r)) for r in rows]


def _header_rows(rows):
    # baris kedua dianggap bagian judul bila sel pertamanya kosong dan tabelnya masih punya isi
    return 2 if len(rows) > 2 and not rows[1][0] else 1


def _ghost_merges(rows, h):
    """Pasangan kolom bersebelahan: satu hanya terisi di judul, satunya hanya terisi di isi."""
    def kind(j):
        head = any(r[j] for r in rows[:h])
        body = any(r[j] for r in rows[h:])
        return "F" if head and body else "H" if head else "D" if body else "E"

    kinds = [kind(j) for j in range(len(rows[0]))]
    merges, j = [], 0
    while j < len(kinds) - 1:
        if {kinds[j], kinds[j + 1]} == {"H", "D"}:
            merges.append(j)
            j += 2
        else:
            j += 1
    return merges


def _apply_merges(rows, merges):
    for j in reversed(merges):
        for r in rows:
            r[j] = r[j] or r[j + 1]
            del r[j + 1]
    return rows


def merged_cols(rows):
    """Jumlah kolom tabel pembuka setelah kolom-hantu digabung, dan jumlah kolom yang
    berisi data: potongan lanjutan di halaman berikutnya (tanpa kolom-hantu/kolom kosong)
    berjumlah kolom salah satunya."""
    first = _clean(rows)
    h = _header_rows(first)
    first = _apply_merges(first, _ghost_merges(first, h))
    used = [j for j in range(len(first[0])) if any(r[j] for r in first[h:])]
    return len(first[0]), len(used)


def _join_fragments(fragments, notes):
    first = _clean(fragments[0])
    h = _header_rows(first)
    raw_cols = len(first[0])
    merges = _ghost_merges(first, h)
    first = _apply_merges(first, merges)
    head, body = first[:h], first[h:]
    src = [0] * len(body)  # indeks potongan (halaman) asal tiap baris isi

    for n, frag in enumerate(fragments[1:], start=1):
        frag = _clean(frag)
        if len(frag[0]) == raw_cols and merges:
            frag = _apply_merges(frag, merges)
        if len(frag[0]) != len(head[0]):
            # potongan lanjutan tanpa kolom kosong (yang di tabel pembuka berisi hantu/arsiran)
            used = [j for j in range(len(head[0])) if any(r[j] for r in body)]
            if len(frag[0]) != len(used):
                notes.append("jumlah kolom potongan lanjutan berbeda, periksa manual")
                continue
            frag = [[r[used.index(j)] if j in used else "" for j in range(len(head[0]))] for r in frag]
        # judul yang diulang di halaman berikutnya (kadang ketempelan sisa sel dari halaman sebelumnya)
        if all(c.startswith(t) for c, t in zip(frag[0], head[0]) if t) and frag[0][0] == head[0][0]:
            for j, (c, t) in enumerate(zip(frag[0], head[0])):
                sisa = c[len(t):].strip()
                if sisa:
                    for r in reversed(body):
                        if r[j]:
                            r[j] = f"{r[j]} {sisa}"
                            break
            frag = frag[h:]
        body += frag
        src += [n] * len(frag)
    return head, body, src


def fix_table(fragments, drop=("No", "No."), ffill=True, with_pages=False):
    """fragments: list tabel (tiap tabel = list baris) yang merupakan satu tabel utuh.
    with_pages=True: kembalikan juga indeks potongan asal tiap baris isi (urut dengan body)."""
    notes = []
    head, body, src = _join_fragments(fragments, notes)

    # satukan baris judul; judul atas diwariskan ke kanan untuk sel gabungan
    if len(head) == 2:
        top, sub, parent = head[0][:], head[1], ""
        for j in range(len(top)):
            if top[j]:
                parent = top[j]
            elif sub[j]:
                top[j] = parent
        header = [(t + s) if t and re.match(r"[a-z]{1,2}\s", s) else " ".join(x for x in (t, s) if x)
                  for t, s in zip(top, sub)]
    else:
        header = head[0]
    header = [h.replace("\n", " ") for h in header]

    # buang kolom tanpa isi (kolom isian/arsiran) dan kolom nomor urut
    keep = [j for j in range(len(header)) if any(r[j] for r in body) and header[j] not in drop]
    kosong = [header[j] for j in range(len(header)) if not any(r[j] for r in body) and header[j]]
    if kosong:
        notes.append(f"{len(kosong)} kolom tanpa isi dibuang")
    header = [header[j] for j in keep]
    body = [[r[j] for j in keep] for r in body]
    src = [s for r, s in zip(body, src) if any(r)]
    body = [r for r in body if any(r)]

    # satu sel menampung nilai beberapa kelompok baris (kolom tanpa garis pemisah di PDF):
    # bila jumlah barisnya habis dibagi jumlah kelompok dan tiap bagian memuat angka, bagi rata berurutan
    starts = [i for i, r in enumerate(body) if r[0]]
    for j, name in enumerate(header):
        if not j or len(starts) < 3:
            continue
        k = 0
        while k < len(starts):
            a = starts[k]
            lines = body[a][j].split("\n") if body[a][j] else []
            m = k + 1
            while m < len(starts) and not body[starts[m]][j]:
                m += 1
            g = m - k
            if g >= 3 and len(lines) >= 4:
                per = len(lines) // g
                parts = [" ".join(lines[x * per:(x + 1) * per]) for x in range(g)]
                if len(lines) % g == 0 and all(re.search(r"\d", x) for x in parts):
                    for x, part in enumerate(parts):
                        body[starts[k + x]][j] = part
                    notes.append(f"kolom '{name}': satu sel dibagi rata ke {g} kelompok baris, cocokkan dengan PDF")
                else:
                    notes.append(f"kolom '{name}': satu sel berisi nilai beberapa baris, periksa manual")
            k = m
    body = [[c.replace("\n", " ") for c in r] for r in body]
    # "Lokal: maks. 30 juta Regional: maks. 40 juta" -> beri koma di antara pasangan label: nilai
    lab = re.compile(r"(?<=[^\s,;:]) (?=[A-Z][\w/-]*: )")
    body = [[lab.sub(", ", c) if len(re.findall(r"(?:^| )[A-Z][\w/-]*: ", c)) >= 2 else c for c in r] for r in body]

    # sel gabungan vertikal: baris lanjutan mewarisi sel di atasnya
    if ffill:
        for i in range(1, len(body)):
            if not body[i][0]:
                body[i] = [c or up for c, up in zip(body[i], body[i - 1])]
    return (header, body, notes, src) if with_pages else (header, body, notes)


def table_to_text(fragments, caption="", with_rows=False, **kw):
    """with_rows=True: kembalikan (teks, catatan, [(indeks_potongan, teks_baris), ...])."""
    header, body, notes, src = fix_table(fragments, with_pages=True, **kw)
    if len(header) < 2:
        empty = ("", notes + ["tabel tidak punya isi selain satu kolom, sebaiknya dibuang"])
        return empty + (([],) if with_rows else ())

    # baris berurutan yang hanya berbeda di satu kolom (bukan kolom pertama) dijadikan satu baris
    groups = []
    for r, frag_no in zip(body, src):
        if groups:
            base, j, vals = groups[-1][:3]
            diff = [k for k in range(len(r)) if r[k] != base[k]]
            if (len(diff) == 1 and diff[0] > 0 and j in (None, diff[0]) and r[diff[0]]
                    and "," not in r[diff[0]] and "," not in base[diff[0]]):
                groups[-1] = (base, diff[0], vals + [r[diff[0]]], groups[-1][3])
                continue
        groups.append((r, None, [], frag_no))

    lines, line_frags = [], []
    for base, j, vals, frag_no in groups:
        parts = []
        for k, (h, v) in enumerate(zip(header, base)):
            if k == j:
                v = ", ".join([v] + vals)
            if v:
                parts.append(f"{h}: {v}" if h else v)
        lines.append("; ".join(parts))
        line_frags.append(frag_no)
    text = "\n".join(x for x in [caption, *lines] if x)
    return (text, notes, list(zip(line_frags, lines))) if with_rows else (text, notes)
