"""Mendeteksi chunk formulir/format surat dan meringkasnya otomatis (tanpa ditulis tangan)."""
import re

LEADER = re.compile(r"\s*(?:[.…·_]\s?){4,}\s*")
ISIAN = "[diisi]"
SURAT = re.compile(r"<KOP|bertanda tangan di bawah ini|Meterai", re.I)
JUDUL = re.compile(r"^(Lampiran \d+\.[^\n]*)", re.M)
LABEL = re.compile(r"([A-Z][A-Za-z/().\- ]{2,40}?)\s*:?\s*\[diisi\]")


def tandai_isian(text):
    return re.sub(r" {2,}", " ", LEADER.sub(f" {ISIAN} ", text)).strip()


def is_form(text, min_isian=3, per_1000=4.0):
    """True bila chunk berupa formulir/format surat: banyak kolom isian atau berciri surat."""
    t = tandai_isian(text)
    n = t.count(ISIAN)
    padat = n >= min_isian and n * 1000 / max(len(t), 1) >= per_1000
    return padat or (bool(SURAT.search(t)) and n >= 2)


def ringkas_form(text, heading=""):
    """Satu kalimat pengganti: judul lampiran + daftar isian yang ditemukan."""
    t = re.sub(r"\n\s*\n\s*:", " :", tandai_isian(text))          # "Nama\n\n: [diisi]" -> "Nama : [diisi]"
    judul = JUDUL.findall(t) or ([heading] if heading else [])
    labels = []
    for m in LABEL.finditer(t):
        lab = re.sub(r"\s+", " ", m.group(1)).strip(" :.-")
        lab = lab.split("  ")[-1]
        if 2 < len(lab) <= 40 and lab.lower() not in {x.lower() for x in labels}:
            labels.append(lab)
    out = "; ".join(judul) if judul else "Formulir"
    out += " — templat isian, bukan ketentuan."
    if labels:
        out += " Isian: " + ", ".join(labels[:25]) + "."
    return out
